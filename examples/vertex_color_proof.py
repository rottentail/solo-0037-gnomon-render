"""顶点色校样：斜拍、跨近平面成像与剖分不变性。

通过真实 RendererService 产出可复现的校样，回答三个问题：

1. 网格顶点色（可选，按 sRGB 导入）如何参与最终成像：
   线性(材质基色) * 线性(顶点色) * 线性(纹理) * 光照。
   缺少 vertex_colors 的旧网格等价于顶点色恒为 1，像素与旧版本完全一致。
2. 平面线性色带（片 A）：同一平面上的线性颜色场，换一条对角线剖分后
   色带不应挪动；透视校正插值在斜拍下仍与解析场一致。
3. 四角不共面的配色（片 B）：换剖分会产生真实的折痕差异，
   它只出现在片内对角线两侧、覆盖与轮廓完全不变，
   从而与裁切/插值造成的边界接缝区分开。

产物写入输出目录，metrics 为线性 RGB 量纲；脚本任一断言失败即以非零码退出。
"""
import argparse
import json
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from gnomon.channels import png_bytes, unpack_channels
from gnomon.color import linear_to_srgb, srgb_to_linear
from gnomon.errors import RenderError
from gnomon.samples import checker_texture
from gnomon.service import RendererService

SLASH = [[0, 1, 2], [0, 2, 3]]       # 对角线 1->3
BACKSLASH = [[0, 1, 3], [1, 2, 3]]   # 对角线 0->2


def band_field(x, y):
    """片 A：在场景线性颜色空间里沿平面连续变化的线性颜色场。"""
    return np.stack([0.50 + 0.30 * x + 0.10 * y,
                     0.50 - 0.20 * x + 0.20 * y,
                     0.40 + 0.15 * x - 0.10 * y], axis=-1)


def quad_mesh(label, corners_linear, triangles, with_uv=True, vertex_colors=None):
    """corners_linear 是四角的线性 RGB；导入时转成作者侧 sRGB 顶点色。"""
    colors = vertex_colors
    if colors is None:
        colors = linear_to_srgb(np.asarray(corners_linear, dtype=np.float64)).tolist()
    mesh = {"label": label,
            "positions": [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
            "triangles": triangles, "normals": [[0, 0, 1]] * 4,
            "vertex_colors": colors}
    if with_uv:
        mesh["uv"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
    return mesh


def render(service, mesh, lens, surface=None, pose=None, light=None, size=256, samples=1):
    node = {"key": "panel", "mesh": mesh, "surface": surface or {"unlit": True, "color": [1, 1, 1]}}
    if pose:
        node["pose"] = pose
    payload = {"label": "vertex-color-proof", "nodes": [node], "lens": lens}
    if light:
        payload["light"] = light
    scene = service.add_scene(payload)
    record = service.render({"scene": scene["id"], "width": size, "height": size, "samples": samples})
    return scene, record, unpack_channels(service.artifacts.read(record, "channels.npz"))


def difference_png(left, right):
    """红色强度=线性色最大通道差；黑=无差异。轮廓差异会显示为蓝色。"""
    delta = np.abs(left["color"] - right["color"]).max(axis=2)
    silhouette = ((left["object_id"] > 0) ^ (right["object_id"] > 0)).astype(float)
    heat = np.stack([np.clip(delta * 4, 0, 1), np.zeros_like(delta), silhouette], axis=2)
    return png_bytes(np.rint(heat * 255).astype(np.uint8))


def side_by_side(left_png, right_png, diff_png, names):
    images = [Image.open(BytesIO(raw)).convert("RGB") for raw in (left_png, right_png, diff_png)]
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 3 + 16, height + 28), (18, 23, 33))
    for index, image in enumerate(images):
        canvas.paste(image, (index * (width + 8), 24))
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def color_png(channels):
    from gnomon.color import encode_rgb
    return png_bytes(encode_rgb(channels["color"]))


def produce(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    oblique = {"eye": [2.6, 1.8, 3.4], "target": [0, 0, 0], "near": 0.1, "far": 30, "fov": 45}
    front = {"eye": [0, 0, 4], "target": [0, 0, 0], "near": 0.1, "far": 20, "fov": 45}
    corners = band_field(np.array([-1, 1, 1, -1]), np.array([-1, -1, 1, 1]))

    with tempfile.TemporaryDirectory(prefix="gnomon-vcolor-") as root:
        service = RendererService(root)
        try:
            # ---- 片 A：平面线性色带，两种对角线剖分 -----------------------
            band = {name: service.add_mesh(quad_mesh(f"色带-{name}", corners, tris))["id"]
                    for name, tris in (("slash", SLASH), ("backslash", BACKSLASH))}
            band_meshes = {name: service.catalog.get(mid, "mesh")["payload"] for name, mid in band.items()}

            def band_render(name, lens, pose=None, samples=1):
                return render(service, band[name], lens, pose=pose, size=256, samples=samples)[2]

            slash_front = band_render("slash", front)
            back_front = band_render("backslash", front)
            slash_tilt = band_render("slash", oblique, pose={"rotate": [0, 35, 0]}, samples=3)
            back_tilt = band_render("backslash", oblique, pose={"rotate": [0, 35, 0]}, samples=3)
            # 单采样时像素颜色与其世界位置同属一个样本点，可做解析场精确核对。
            slash_tilt_1s = band_render("slash", oblique, pose={"rotate": [0, 35, 0]}, samples=1)

            covered = slash_front["object_id"] > 0
            assert np.array_equal(covered, back_front["object_id"] > 0), "色带换剖分改变了覆盖轮廓"
            front_delta = np.abs(slash_front["color"] - back_front["color"])[covered]
            tilt_delta = np.abs(slash_tilt["color"] - back_tilt["color"])[covered]
            analytic = band_field(slash_front["world"][..., 0], slash_front["world"][..., 1])
            analytic_err = np.abs(slash_front["color"] - analytic)[covered]
            # 斜拍带姿态：把世界坐标逆变换回局部 (x,y) 再与解析场比较。
            # 多采样的颜色是全部子采样平均、世界位置只取最近子采样，
            # 因此解析场核对必须使用单采样成像；多采样结果只用于剖分不变性。
            theta = np.radians(35.0)
            local_x = np.cos(theta) * slash_tilt_1s["world"][..., 0] - np.sin(theta) * slash_tilt_1s["world"][..., 2]
            local_y = slash_tilt_1s["world"][..., 1]
            full_tilt = slash_tilt_1s["object_id"] > 0
            tilt_err = np.abs(slash_tilt_1s["color"] - band_field(local_x, local_y))[full_tilt]
            metrics["band_linear"] = {
                "front_max_linear_delta": float(front_delta.max()),
                "tilted_max_linear_delta": float(tilt_delta.max()),
                "front_analytic_max_error": float(analytic_err.max()),
                "tilted_analytic_max_error": float(tilt_err.max()),
                "silhouette_identical": bool(np.array_equal(covered, back_front["object_id"] > 0)),
                "assertion": "两种剖分与斜拍的线性差值均为浮点误差量级（阈值 1e-10）"}
            assert front_delta.max() < 1e-10 and tilt_delta.max() < 1e-10, "色带换剖分发生了挪动"
            assert analytic_err.max() < 1e-10 and tilt_err.max() < 1e-10, "透视校正插值偏离解析线性场"

            (output / "band-diagonal-proof.png").write_bytes(
                side_by_side(color_png(slash_tilt), color_png(back_tilt),
                             difference_png(slash_tilt, back_tilt),
                             ("slash", "backslash", "delta")))

            # ---- 片 B：四角不共面配色，换剖分应出现真实折痕 ---------------
            fold_corners = corners.copy()
            # 让第四角落在另外三角仿射补全之外：这不是同一个平面线性渐变。
            fold_corners[3] = np.clip(corners[3] + np.array([0.35, -0.25, 0.20]), 0, 1)
            fold = {name: service.add_mesh(quad_mesh(f"折痕-{name}", fold_corners, tris))["id"]
                    for name, tris in (("slash", SLASH), ("backslash", BACKSLASH))}
            # 单采样 + 较高分辨率，让采样中心能贴近共享外缘，量化差异如何向边缘衰减。
            fold_channels = {name: render(service, fold[name], front, size=512, samples=1)[2]
                             for name in fold}
            fold_cover = fold_channels["slash"]["object_id"] > 0
            assert np.array_equal(fold_cover, fold_channels["backslash"]["object_id"] > 0), \
                "折痕片的轮廓不应因剖分改变"
            fold_delta = np.abs(fold_channels["slash"]["color"] - fold_channels["backslash"]["color"]).max(axis=2)
            fold_world = fold_channels["slash"]["world"]
            edge_distance = np.minimum(
                np.minimum(np.abs(fold_world[..., 0] - 1), np.abs(fold_world[..., 0] + 1)),
                np.minimum(np.abs(fold_world[..., 1] - 1), np.abs(fold_world[..., 1] + 1)))
            near_edge = fold_cover & (edge_distance < 0.01)
            deep_interior = fold_cover & (edge_distance > 0.15)
            metrics["fold_nonplanar"] = {
                "max_linear_delta": float(fold_delta[fold_cover].max()),
                "mean_linear_delta": float(fold_delta[fold_cover].mean()),
                "changed_pixels_gt_2e-3": int((fold_delta[fold_cover] > 2e-3).sum()),
                "near_shared_edge_max_delta": float(fold_delta[near_edge].max()),
                "deep_interior_max_delta": float(fold_delta[deep_interior].max()),
                "silhouette_identical": True,
                "assertion": "轮廓一致；差异向共享边/角衰减到 1e-3 以下、在片内对角线两侧达到约 0.17，"
                             "这是换剖分的真实差异，而裁切/插值接缝只会出现在覆盖边界"}
            assert fold_delta[fold_cover].max() > 0.02, "非共面四角理应产生可观察的剖分差异"
            assert fold_delta[near_edge].max() < 2e-3, "共享边上的顶点色相同，不应出现接缝"

            (output / "fold-retriangulation-proof.png").write_bytes(
                side_by_side(color_png(fold_channels["slash"]), color_png(fold_channels["backslash"]),
                             difference_png(fold_channels["slash"], fold_channels["backslash"]),
                             ("slash", "backslash", "delta")))

            # ---- 跨近平面：裁切交点的顶点色仍属同一表面 --------------------
            near_lens = {"eye": [0, 0, 1.2], "target": [0, 0, 0], "near": 1.0, "far": 20, "fov": 60}
            # 单采样保证颜色与世界位置同点；多采样的颜色平均会混入被裁掉的子采样。
            near_scene, near_record, near_channels = render(
                service, band["slash"], near_lens, pose={"rotate": [0, 75, 0]}, size=256, samples=1)
            near_cover = near_channels["object_id"] > 0
            assert 0.05 < float(near_channels["coverage"].mean()) < 0.95, "近平面应切掉片的一部分"
            assert np.isfinite(near_channels["depth"][near_cover]).all(), "裁切后不得出现反投或无限深度"
            theta = np.radians(75.0)
            nx = np.cos(theta) * near_channels["world"][..., 0] - np.sin(theta) * near_channels["world"][..., 2]
            ny = near_channels["world"][..., 1]
            # 颜色与世界位置来自同一采样点，保留表面应连续贴合解析线性场。
            near_err = np.abs(near_channels["color"] - band_field(nx, ny))[near_cover]
            # 裁切边界两侧的覆盖是部分覆盖；边界内侧的颜色必须连续贴合解析场。
            edge_rows = np.where(near_channels["coverage"].mean(axis=1) < 0.999)[0]
            metrics["near_plane_clip"] = {
                "coverage": float(near_channels["coverage"].mean()),
                "analytic_max_error_inside": float(near_err.max()),
                "partial_coverage_rows": int(len(edge_rows)),
                "assertion": "近平面只改变覆盖与轮廓；保留表面的颜色仍连续，差异来自裁切而非插值接缝"}
            assert near_err.max() < 1e-10, "近平面裁切后的插值颜色偏离解析场"
            (output / "near-plane-clip-proof.png").write_bytes(color_png(near_channels))

            # ---- 顶点色与基色、PNG 纹理、光照同时成像（数值核对） ---------
            uniform = quad_mesh("组合-恒色", None, SLASH,
                                vertex_colors=[[0.5, 0.5, 0.5]] * 4)
            uniform_id = service.add_mesh(uniform)["id"]
            white_vc = service.add_mesh(quad_mesh("组合-白顶点", None, SLASH,
                                                  vertex_colors=[[1, 1, 1]] * 4))["id"]
            lin06 = float(srgb_to_linear(np.array([0.6]))[0])
            lin05 = float(srgb_to_linear(np.array([0.5]))[0])

            _, _, base_only = render(service, white_vc, front,
                                     surface={"unlit": True, "color": [0.6, 0.6, 0.6]}, size=64)
            _, _, base_times_vc = render(service, uniform_id, front,
                                         surface={"unlit": True, "color": [0.6, 0.6, 0.6]}, size=64)
            base_cover = base_only["object_id"] > 0
            got = base_times_vc["color"][base_cover]
            assert np.allclose(got, lin06 * lin05, atol=1e-12), "基色×顶点色的线性乘积不符"
            assert np.allclose(base_only["color"][base_cover], lin06, atol=1e-12)

            texture = service.add_texture(checker_texture())["id"]
            textured_surface = {"unlit": True, "color": [1, 1, 1], "texture": texture,
                                "wrap": "clamp", "alpha_cutoff": 0.5}
            _, _, tex_white = render(service, white_vc, front, surface=textured_surface, size=128)
            _, _, tex_half = render(service, uniform_id, front, surface=textured_surface, size=128)
            tex_cover = tex_white["object_id"] > 0
            ratio = tex_half["color"][tex_cover] / tex_white["color"][tex_cover]
            assert np.allclose(ratio, lin05, atol=1e-10), "顶点色应等比缩放纹理结果"

            light = {"toward": [0, 0, 1], "energy": 0.8, "ambient": 0.2}
            _, _, lit_white = render(service, white_vc, front,
                                     surface={"unlit": False, "color": [1, 1, 1]},
                                     light=light, size=64)
            _, _, lit_half = render(service, uniform_id, front,
                                    surface={"unlit": False, "color": [1, 1, 1]},
                                    light=light, size=64)
            lit_cover = lit_white["object_id"] > 0
            assert np.allclose(lit_half["color"][lit_cover], lin05 * lit_white["color"][lit_cover], atol=1e-12), \
                "顶点色与光照应在线性空间相乘"
            metrics["composition"] = {
                "base_linear": lin06, "vertex_linear": lin05,
                "base_times_vertex": lin06 * lin05,
                "texture_scale_unique": np.unique(np.round(ratio, 10)).tolist(),
                "lighting_preserves_ratio": True,
                "assertion": "最终线性颜色 = 线性基色 × 线性顶点色 × 线性纹理 × 环境+Lambert"}

            # ---- 输入校验：顶点色数量与取值范围 ---------------------------
            wrong_count = quad_mesh("坏-数量", corners, SLASH, vertex_colors=[[0.5, 0.5, 0.5]])
            out_of_range = quad_mesh("坏-范围", corners, SLASH,
                                     vertex_colors=[[0.5, 0.5, 1.2]] * 4)
            rejections = {}
            for name, payload in (("count", wrong_count), ("range", out_of_range)):
                try:
                    service.add_mesh(payload)
                    raise AssertionError(f"{name} 输入应当被拒绝")
                except RenderError as exc:
                    assert exc.code == "mesh_vertex_color", exc.code
                    rejections[name] = exc.code
            metrics["validation"] = {"rejected": rejections}

            # ---- 旧素材：没有顶点色时像素与旧行为完全一致 -----------------
            legacy = {"label": "旧网格无顶点色",
                      "positions": [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                      "triangles": SLASH, "normals": [[0, 0, 1]] * 4}
            legacy_id = service.add_mesh(legacy)["id"]
            assert service.catalog.get(legacy_id, "mesh")["payload"]["vertex_colors"] is None
            _, _, legacy_channels = render(service, legacy_id, front,
                                           surface={"unlit": True}, size=64)
            legacy_value = np.unique(np.round(legacy_channels["color"][legacy_channels["object_id"] > 0], 12))
            assert np.allclose(legacy_value, lin06, atol=1e-12), "旧素材成像发生变化"
            metrics["legacy_unchanged"] = {"linear_rgb": legacy_value.tolist(),
                                           "equals_surface_base": True}
            (output / "legacy-no-vertex-color.png").write_bytes(color_png(legacy_channels))

            (output / "proof-metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
            (output / "proof-scenes.json").write_text(
                json.dumps({"band_corners_linear": corners.tolist(),
                            "fold_corners_linear": fold_corners.tolist(),
                            "band_meshes": band_meshes,
                            "oblique_lens": oblique, "near_lens": near_lens},
                           ensure_ascii=False, indent=2, allow_nan=False))
        finally:
            service.close()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("run-output/vertex-color-proof"))
    arguments = parser.parse_args()
    results = produce(arguments.output)
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
