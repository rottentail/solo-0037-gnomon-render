"""Wavefront OBJ/MTL 导入:独立索引解析、非凸多边形三角化、材质映射与场景计划。

位置/UV/法线拥有独立索引空间,按 (v,vt,vn) 三元组拆分输出顶点,接缝不被抹平。
多边形按 Newell 法线投影后耳切,保留原始绕序;不取前三个点的扇形三角化。
无法表示的材质特性与缺失资源进入 diagnostics,导入仍可提交并被记录。
"""
import re
import numpy as np
from .errors import require
from .texture import decode_texture

MAX_FACE_CORNERS = 256
MAX_IMPORT_VERTICES = 20000
MAX_IMPORT_TRIANGLES = 30000
MAX_TOTAL_TRIANGLES = 60000
MAX_NODES = 100
DEGENERATE_AREA2 = 1e-12

# map 选项中唯一可表示的是 -clamp;其余选项改变采样但本服务没有对应字段。
_MAP_OPTION_ARITY = {"-s": 3, "-o": 3, "-t": 3, "-mm": 2, "-bm": 1, "-boost": 1,
                     "-texres": 1, "-clamp": 1, "-blendu": 1, "-blendv": 1,
                     "-imfchan": 1, "-type": 1, "-cc": 1}


def _diagnostic(diagnostics, severity, code, detail, subject=None):
    entry = {"severity": severity, "code": code, "detail": detail}
    if subject is not None:
        entry["subject"] = subject
    diagnostics.append(entry)


def _resolve_index(token, count, family, line):
    require(token != "", "obj_index", f"line {line}: empty {family} index")
    try:
        value = int(token)
    except ValueError:
        raise _domain("obj_index", f"line {line}: {family} index {token!r} is not an integer")
    require(value != 0, "obj_index", f"line {line}: OBJ indices are 1-based; 0 is invalid")
    resolved = value - 1 if value > 0 else count + value
    require(0 <= resolved < count, "obj_index",
            f"line {line}: {family} index {value} outside 1..{count}")
    return resolved


def _domain(code, detail):
    from .errors import RenderError
    return RenderError(code, detail)


def _strip_comment(line):
    return line.split("#", 1)[0].strip()


def _logical_lines(text):
    merged, pending = [], ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        merged.append(pending + line)
        pending = ""
    if pending:
        merged.append(pending)
    return merged


def parse_obj(text, diagnostics):
    positions, uvs, normals, faces = [], [], [], []
    mtllib, ignored, skipped = [], {}, {"p": 0, "l": 0}
    current_o, current_g, current_mtl = None, None, None
    for number, raw in enumerate(_logical_lines(text), 1):
        line = _strip_comment(raw)
        if not line:
            continue
        parts = line.split()
        keyword, args = parts[0].lower(), parts[1:]
        if keyword == "v":
            require(len(args) >= 3, "obj_parse", f"line {number}: v needs x y z")
            try:
                point = [float(args[0]), float(args[1]), float(args[2])]
            except ValueError:
                raise _domain("obj_parse", f"line {number}: v components must be numeric")
            positions.append(point)
        elif keyword == "vt":
            require(len(args) >= 1, "obj_parse", f"line {number}: vt needs at least u")
            try:
                uvs.append([float(args[0]), float(args[1]) if len(args) > 1 else 0.0])
            except ValueError:
                raise _domain("obj_parse", f"line {number}: vt components must be numeric")
        elif keyword == "vn":
            require(len(args) >= 3, "obj_parse", f"line {number}: vn needs x y z")
            try:
                vector = [float(args[0]), float(args[1]), float(args[2])]
            except ValueError:
                raise _domain("obj_parse", f"line {number}: vn components must be numeric")
            require(float(np.linalg.norm(vector)) > 1e-12, "obj_normal",
                    f"line {number}: vn must be nonzero")
            normals.append(vector)
        elif keyword == "f":
            require(len(args) >= 3, "obj_face",
                    f"line {number}: a face needs at least 3 corners, got {len(args)}")
            require(len(args) <= MAX_FACE_CORNERS, "obj_face_size",
                    f"line {number}: face has {len(args)} corners; limit is {MAX_FACE_CORNERS}")
            corners = []
            for token in args:
                fields = token.split("/")
                require(len(fields) <= 3, "obj_index",
                        f"line {number}: corner {token!r} has too many '/' separators")
                v = _resolve_index(fields[0], len(positions), "position", number)
                vt = (_resolve_index(fields[1], len(uvs), "uv", number)
                      if len(fields) > 1 and fields[1] != "" else None)
                vn = (_resolve_index(fields[2], len(normals), "normal", number)
                      if len(fields) > 2 and fields[2] != "" else None)
                corners.append((v, vt, vn))
            faces.append({"group": current_g or current_o or "default",
                          "material": current_mtl, "corners": corners, "line": number})
        elif keyword == "o":
            current_o = " ".join(args) if args else None
        elif keyword == "g":
            current_g = " ".join(args) if args else None
        elif keyword == "usemtl":
            current_mtl = " ".join(args) if args else None
        elif keyword == "mtllib":
            mtllib.extend(args)
        elif keyword in ("p", "l"):
            skipped[keyword] += 1
        elif keyword == "s":
            continue  # 平滑组不改变本服务的面积加权法线语义。
        else:
            ignored[keyword] = ignored.get(keyword, 0) + 1
    require(len(faces) > 0, "obj_empty", "OBJ text contains no faces")
    if skipped["p"] or skipped["l"]:
        _diagnostic(diagnostics, "info", "obj_primitives_skipped",
                    f"{skipped['p']} point and {skipped['l']} line statements are not "
                    "triangle geometry and were skipped")
    if ignored:
        summary = ", ".join(f"{name}×{count}" for name, count in sorted(ignored.items()))
        _diagnostic(diagnostics, "info", "obj_statements_ignored",
                    f"unsupported OBJ statements ignored: {summary}")
    if len(mtllib) > 1:
        _diagnostic(diagnostics, "info", "obj_mtllib_multiple",
                    f"OBJ declares {len(mtllib)} material libraries {mtllib}; materials resolve "
                    "from the single provided mtl_text")
    return {"positions": positions, "uvs": uvs, "normals": normals,
            "faces": faces, "mtllib": mtllib}


def _floats(args, count, line, keyword):
    try:
        return [float(item) for item in args[:count]]
    except ValueError:
        raise _domain("mtl_parse", f"line {line}: {keyword} components must be numeric")


def _parse_map(args, line, keyword):
    """返回 (文件名, 选项字典);文件名取最后一个 token,无法解析的选项进入选项字典。"""
    require(len(args) >= 1, "mtl_parse", f"line {line}: {keyword} needs a file name")
    options, index = {}, 0
    while index < len(args) - 1:
        token = args[index]
        if token.startswith("-"):
            arity = _MAP_OPTION_ARITY.get(token.lower(), 0)
            values = args[index + 1:index + 1 + arity]
            options[token.lower()] = values
            index += 1 + arity
        else:
            index += 1
    return args[-1], options


def parse_mtl(text, diagnostics):
    materials, order = {}, []
    current = None
    for number, raw in enumerate(_logical_lines(text), 1):
        line = _strip_comment(raw)
        if not line:
            continue
        parts = line.split()
        keyword, args = parts[0].lower(), parts[1:]
        if keyword == "newmtl":
            require(args, "mtl_parse", f"line {number}: newmtl needs a name")
            name = " ".join(args)
            if name in materials:
                _diagnostic(diagnostics, "warning", "mtl_duplicate",
                            f"line {number}: material {name!r} redefined; later definition wins",
                            subject=name)
            current = {"name": name, "kd": None, "map_kd": None, "clamp": None,
                       "alpha": 1.0, "unsupported": set(), "line": number}
            materials[name] = current
            order.append(name)
            continue
        if current is None:
            _diagnostic(diagnostics, "warning", "mtl_statement",
                        f"line {number}: {keyword} appears before any newmtl; ignored")
            continue
        try:
            if keyword == "kd":
                require(len(args) >= 3, "mtl_parse", f"line {number}: Kd needs r g b")
                current["kd"] = _floats(args, 3, number, "Kd")
            elif keyword in ("ka", "ks", "ke", "tf"):
                values = _floats(args, min(len(args), 3), number, keyword)
                if keyword == "ks" and any(abs(v) > 1e-9 for v in values):
                    current["unsupported"].add("specular Ks")
                if keyword == "ke" and any(abs(v) > 1e-9 for v in values):
                    current["unsupported"].add("emissive Ke")
            elif keyword == "ns":
                if any(abs(v) > 1e-9 for v in _floats(args, 1, number, "Ns")):
                    current["unsupported"].add("specular exponent Ns")
            elif keyword == "ni":
                values = _floats(args, 1, number, "Ni")
                if values and abs(values[0] - 1.0) > 1e-9:
                    current["unsupported"].add("refraction Ni")
            elif keyword == "d":
                current["alpha"] = _floats(args, 1, number, "d")[0]
            elif keyword == "tr":
                current["alpha"] = 1.0 - _floats(args, 1, number, "Tr")[0]
            elif keyword == "illum":
                values = _floats(args, 1, number, "illum")
                if values and values[0] > 2:
                    current["unsupported"].add(f"illum {int(values[0])} lighting model")
            elif keyword == "map_kd":
                filename, options = _parse_map(args, number, "map_Kd")
                current["map_kd"] = filename
                current["map_options"] = options
                clamp = options.get("-clamp", ["off"])
                current["clamp"] = clamp[0].lower() == "on" if clamp else None
            elif keyword in ("map_ka", "map_ks", "map_ke", "map_d"):
                current["unsupported"].add(f"{keyword} map")
            elif keyword in ("map_bump", "bump", "norm"):
                current["unsupported"].add("bump/normal map")
            elif keyword in ("disp", "decal", "refl"):
                current["unsupported"].add(f"{keyword} map")
            elif keyword == "sharpness":
                continue
            else:
                current["unsupported"].add(keyword)
        except IndexError:
            raise _domain("mtl_parse", f"line {number}: {keyword} has too few values")
    return {"materials": materials, "order": order}


def _newell_normal(points):
    normal = np.zeros(3)
    for index in range(len(points)):
        a, b = points[index], points[(index + 1) % len(points)]
        normal += [(a[1] - b[1]) * (a[2] + b[2]),
                   (a[2] - b[2]) * (a[0] + b[0]),
                   (a[0] - b[0]) * (a[1] + b[1])]
    return normal


def _triangulate_polygon(points):
    """耳切三角化。返回 (三角形局部索引列表, 退化丢弃数)。保留输入绕序。"""
    count = len(points)
    if count == 3:
        area2 = float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
        return ([(0, 1, 2)] if area2 > DEGENERATE_AREA2 else [], 0 if area2 > DEGENERATE_AREA2 else 1)
    normal = _newell_normal(points)
    if float(np.linalg.norm(normal)) <= 1e-30:
        return [], count - 2
    # 沿最大分量投影到 2D,保持一致的朝向判定。
    drop = int(np.argmax(np.abs(normal)))
    plane = [axis for axis in range(3) if axis != drop]
    flat = points[:, plane]
    extent = float(max(1.0, np.ptp(flat, axis=0).max()))
    tolerance = 1e-9 * extent * extent
    signed = 0.5 * float(np.sum(flat[:, 0] * np.roll(flat[:, 1], -1)
                                - np.roll(flat[:, 0], -1) * flat[:, 1]))
    if abs(signed) <= tolerance:
        return [], count - 2
    orientation = 1.0 if signed > 0 else -1.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    remaining = list(range(count))
    triangles, dropped = [], 0
    while len(remaining) > 3:
        clipped = False
        size = len(remaining)
        for slot in range(size):
            prev_i = remaining[slot - 1]
            cur_i = remaining[slot]
            next_i = remaining[(slot + 1) % size]
            turn = cross(flat[prev_i], flat[cur_i], flat[next_i]) * orientation
            if turn <= tolerance:
                # 共线顶点:作为零面积耳剪掉,三角形随后按退化丢弃。
                if turn >= -tolerance and not any(
                        cross(flat[prev_i], flat[cur_i], flat[other]) * orientation > tolerance
                        and cross(flat[cur_i], flat[next_i], flat[other]) * orientation > tolerance
                        and cross(flat[next_i], flat[prev_i], flat[other]) * orientation > tolerance
                        for other in remaining if other not in (prev_i, cur_i, next_i)):
                    remaining.pop(slot)
                    dropped += 1
                    clipped = True
                    break
                continue
            inside = any(
                cross(flat[prev_i], flat[cur_i], flat[other]) * orientation > tolerance
                and cross(flat[cur_i], flat[next_i], flat[other]) * orientation > tolerance
                and cross(flat[next_i], flat[prev_i], flat[other]) * orientation > tolerance
                for other in remaining if other not in (prev_i, cur_i, next_i))
            if not inside:
                a, b, c = points[prev_i], points[cur_i], points[next_i]
                area2 = float(np.linalg.norm(np.cross(b - a, c - a)))
                if area2 > DEGENERATE_AREA2:
                    triangles.append((prev_i, cur_i, next_i))
                else:
                    dropped += 1
                remaining.pop(slot)
                clipped = True
                break
        require(clipped, "obj_triangulation",
                "polygon is self-intersecting or degenerate and cannot be triangulated")
    a, b, c = (points[i] for i in remaining)
    if float(np.linalg.norm(np.cross(b - a, c - a))) > DEGENERATE_AREA2:
        triangles.append(tuple(remaining))
    else:
        dropped += 1
    return triangles, dropped


def _sanitize_key(name, used):
    key = re.sub(r"[^A-Za-z0-9_-]", "-", name).strip("-") or "node"
    if not key[0].isalpha():
        key = "n" + key
    base, key = key[:57], key[:57]
    suffix = 2
    while key in used:
        key = f"{base}-{suffix}"
        suffix += 1
    used.add(key)
    return key


def _auto_lens(points):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = float(np.linalg.norm(points - center, axis=1).max())
    radius = max(radius, 1e-3)
    require(radius <= 3e5, "obj_bounds",
            "model extents exceed auto-framing range; provide an explicit lens")
    distance = radius / float(np.tan(np.radians(22.5))) * 1.2
    direction = np.asarray([3.0, 2.0, 4.0])
    direction /= float(np.linalg.norm(direction))
    eye = center + direction * distance
    return {"projection": "perspective", "eye": eye.tolist(), "target": center.tolist(),
            "up": [0, 1, 0], "near": max(distance * 0.02, 0.001),
            "far": min(distance * 5 + radius * 4, 1e6), "fov": 45, "span": 4}


def _find_image(name, images):
    normalized = name.replace("\\", "/")
    if normalized in images:
        return normalized
    base = normalized.rsplit("/", 1)[-1]
    matches = [key for key in images if key.replace("\\", "/").rsplit("/", 1)[-1] == base]
    if len(matches) == 1:
        return matches[0]
    lowered = base.lower()
    matches = [key for key in images
               if key.replace("\\", "/").rsplit("/", 1)[-1].lower() == lowered]
    return matches[0] if len(matches) == 1 else None


def build_import(payload):
    """纯函数:把 OBJ/MTL/图片输入编译为网格、纹理、场景计划与诊断。不创建记录。"""
    diagnostics = []
    obj = parse_obj(payload["obj_text"], diagnostics)
    if obj["mtllib"] and not payload.get("mtl_text"):
        _diagnostic(diagnostics, "warning", "obj_mtllib_unprovided",
                    f"OBJ declares material libraries {obj['mtllib']} but no mtl_text was "
                    "submitted; faces fall back to the default surface")
    library = parse_mtl(payload["mtl_text"], diagnostics) if payload.get("mtl_text") else None
    materials = library["materials"] if library else {}
    images = payload.get("images") or {}

    groups, group_order = {}, []
    for face in obj["faces"]:
        key = (face["group"], face["material"])
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(face)

    mesh_inputs, mesh_plans, node_surfaces = [], [], {}
    texture_inputs, texture_bindings = {}, {}
    used_keys = {"root"}
    total_triangles = 0
    source_positions = np.asarray(obj["positions"], dtype=np.float64)

    for group_name, material_name in group_order:
        faces = groups[(group_name, material_name)]
        corner_map, out_positions, out_uvs, out_normals = {}, [], [], []
        triangles, dropped = [], 0
        has_uv = has_normal = None
        for face in faces:
            corners = face["corners"]
            face_uv = any(corner[1] is not None for corner in corners)
            face_normal = any(corner[2] is not None for corner in corners)
            if has_uv is None:
                has_uv, has_normal = face_uv, face_normal
            require(face_uv == has_uv and all((c[1] is not None) == has_uv for c in corners),
                    "obj_uv_partial",
                    f"group {group_name!r}: faces mix corners with and without vt; "
                    "UV must be all-or-nothing within an imported mesh")
            require(face_normal == has_normal
                    and all((c[2] is not None) == has_normal for c in corners),
                    "obj_normal_partial",
                    f"group {group_name!r}: faces mix corners with and without vn; "
                    "normals must be all-or-nothing within an imported mesh")
            local = []
            for corner in corners:
                if corner not in corner_map:
                    require(len(out_positions) < MAX_IMPORT_VERTICES, "obj_budget",
                            f"group {group_name!r} expands past {MAX_IMPORT_VERTICES} vertices; "
                            "split the OBJ group or its material assignments")
                    corner_map[corner] = len(out_positions)
                    out_positions.append(obj["positions"][corner[0]])
                    if has_uv:
                        out_uvs.append(obj["uvs"][corner[1]])
                    if has_normal:
                        out_normals.append(obj["normals"][corner[2]])
                local.append(corner_map[corner])
            polygon = source_positions[[corner[0] for corner in corners]]
            pieces, lost = _triangulate_polygon(polygon)
            dropped += lost
            triangles.extend((local[a], local[b], local[c]) for a, b, c in pieces)
        require(triangles, "obj_degenerate",
                f"group {group_name!r}: all faces are zero-area; nothing importable remains")
        require(len(triangles) <= MAX_IMPORT_TRIANGLES, "obj_budget",
                f"group {group_name!r} produces {len(triangles)} triangles; "
                f"per-mesh limit is {MAX_IMPORT_TRIANGLES}")
        total_triangles += len(triangles)
        require(total_triangles <= MAX_TOTAL_TRIANGLES, "obj_budget",
                f"import totals over {MAX_TOTAL_TRIANGLES} triangles; scene budget exceeded")
        if dropped:
            _diagnostic(diagnostics, "info", "obj_degenerate_dropped",
                        f"{dropped} zero-area triangles from collinear corners were dropped",
                        subject=group_name)

        node_key = _sanitize_key(f"{group_name}-{material_name or 'default'}", used_keys)
        label = f"{payload['label']}:{group_name}:{material_name or 'default'}"[:120]
        mesh_input = {"label": label, "positions": out_positions, "triangles": triangles}
        if has_normal:
            mesh_input["normals"] = out_normals
        if has_uv:
            mesh_input["uv"] = out_uvs
        mesh_inputs.append(mesh_input)

        material = materials.get(material_name) if material_name else None
        if material_name and material is None:
            _diagnostic(diagnostics, "warning", "mtl_unknown_material",
                        f"usemtl {material_name!r} has no definition in the provided library; "
                        "default surface applied", subject=material_name)
        surface, texture_label = _map_surface(material, material_name, has_uv,
                                              images, texture_inputs, diagnostics)
        node_surfaces[node_key] = surface
        if texture_label is not None:
            texture_bindings[node_key] = texture_label
        mesh_plans.append({"node": node_key, "label": label, "group": group_name,
                           "material": material_name, "vertices": len(out_positions),
                           "source_positions": len({c[0] for f in faces for c in f['corners']}),
                           "triangles": len(triangles), "uv": has_uv,
                           "normals": "explicit" if has_normal else "generated",
                           "degenerate_dropped": dropped})

    require(len(mesh_plans) + 1 <= MAX_NODES, "obj_budget",
            f"import needs {len(mesh_plans) + 1} nodes; scene limit is {MAX_NODES}")
    for name, image in images.items():
        if not any(plan["file"] == name for plan in
                   (t["plan"] for t in texture_inputs.values())):
            _diagnostic(diagnostics, "info", "image_unreferenced",
                        f"image {name!r} is not referenced by any used material", subject=name)

    nodes = [{"key": "root"}]
    mesh_bindings = {}
    for plan in mesh_plans:
        nodes.append({"key": plan["node"], "parent": "root",
                      "surface": node_surfaces[plan["node"]]})
        mesh_bindings[plan["node"]] = plan["label"]
    all_points = source_positions[[corner[0] for face in obj["faces"] for corner in face["corners"]]]
    scene = {"label": payload["label"], "nodes": nodes,
             "lens": payload.get("lens") or _auto_lens(all_points)}
    if payload.get("light"):
        scene["light"] = payload["light"]
    if payload.get("background") is not None:
        scene["background"] = payload["background"]

    plan = {"meshes": mesh_plans, "materials": _material_plans(materials, group_order),
            "textures": [t["plan"] for t in texture_inputs.values()],
            "mtllib_declared": obj["mtllib"],
            "stats": {"groups": len(mesh_plans), "faces": len(obj["faces"]),
                      "positions_in": len(obj["positions"]),
                      "vertices_out": sum(p["vertices"] for p in mesh_plans),
                      "triangles": total_triangles}}
    return {"scene": scene, "mesh_inputs": mesh_inputs,
            "texture_inputs": list(texture_inputs.values()),
            "mesh_bindings": mesh_bindings, "texture_bindings": texture_bindings,
            "diagnostics": diagnostics, "plan": plan}


def _map_surface(material, material_name, mesh_has_uv, images, texture_inputs, diagnostics):
    surface = {"color": [0.6, 0.6, 0.6], "texture": None, "wrap": "repeat",
               "alpha_cutoff": 0.5, "unlit": False, "double_sided": True}
    texture_label = None
    if material is None:
        return surface, None
    kd = material["kd"]
    if kd is not None:
        clamped = [min(1.0, max(0.0, value)) for value in kd]
        if clamped != kd:
            _diagnostic(diagnostics, "warning", "mtl_color_clamped",
                        f"Kd {kd} outside [0,1]; clamped", subject=material_name)
        surface["color"] = clamped
    alpha = material["alpha"]
    if alpha < 1.0 - 1e-9:
        _diagnostic(diagnostics, "warning", "mtl_transparency",
                    f"material opacity {alpha:g} cannot be represented (no alpha blending); "
                    "imported as opaque", subject=material_name)
    if material["unsupported"]:
        _diagnostic(diagnostics, "warning", "mtl_unsupported",
                    "not representable and ignored: "
                    + ", ".join(sorted(material["unsupported"])), subject=material_name)
    options = material.get("map_options") or {}
    ignored_options = sorted(name for name in options if name != "-clamp")
    if ignored_options:
        _diagnostic(diagnostics, "info", "mtl_map_option",
                    "map options not representable, texture applied unmodified: "
                    + ", ".join(ignored_options), subject=material_name)
    if material["clamp"]:
        surface["wrap"] = "clamp"
    filename = material["map_kd"]
    if filename:
        if not mesh_has_uv:
            _diagnostic(diagnostics, "warning", "mtl_texture_no_uv",
                        f"map_Kd {filename!r} dropped: the faces using this material have no "
                        "vt coordinates", subject=material_name)
        else:
            key = _find_image(filename, images)
            if key is None:
                _diagnostic(diagnostics, "warning", "mtl_missing_texture",
                            f"map_Kd file {filename!r} was not among the submitted images; "
                            "base color used instead", subject=material_name)
            else:
                label = f"map:{material_name}"[:120]
                if label not in texture_inputs:
                    try:
                        decode_texture({"label": label, "png_base64": images[key]})
                    except Exception as exc:
                        detail = getattr(exc, "detail", str(exc))
                        _diagnostic(diagnostics, "warning", "mtl_texture_invalid",
                                    f"map_Kd {filename!r} rejected: {detail}",
                                    subject=material_name)
                        return surface, None
                    texture_inputs[label] = {
                        "input": {"label": label, "png_base64": images[key]},
                        "plan": {"label": label, "material": material_name, "file": key}}
                surface["texture"] = None  # 提交时由服务替换为纹理记录 ID。
                texture_label = label
    return surface, texture_label


def _material_plans(materials, group_order):
    used = {material for _, material in group_order if material}
    plans = []
    for name, material in materials.items():
        plans.append({"name": name, "used": name in used,
                      "kd": material["kd"], "map_kd": material["map_kd"],
                      "alpha": material["alpha"],
                      "unsupported": sorted(material["unsupported"])})
    return plans
