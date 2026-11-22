import numpy as np
from .camera import camera_matrices
from .clipping import clip_triangle, PLANES
from .color import srgb_to_linear
from .texture import texture_array
from .shading import shade
from .errors import require
from .mirror import reflection_matrix, plan_reflections


def allocate(width,height,samples,background,reflection=False):
    shape = (height*samples,width*samples)
    buffers = {"color":np.broadcast_to(srgb_to_linear(background),(*shape,3)).copy(),
               "depth":np.full(shape,np.inf),"normal":np.zeros((*shape,3)),
               "object_id":np.zeros(shape,dtype=np.int32),"world":np.zeros((*shape,3))}
    if reflection:
        buffers["reflection_id"] = np.zeros(shape,dtype=np.int32)
        buffers["reflection_world"] = np.zeros((*shape,3))
    return buffers


def edge(a,b,x,y):
    return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])


def sample_reflection(reflection,rgb,bx,by,region,samples,buffers):
    """Blend the mirrored view into mirror fragments and record its provenance.

    The reflection pass shares the main camera's NDC, so a fragment's reflection
    sits at its own full-frame position, rescaled into the reflection buffer.
    """
    kx,ky = reflection["scale"]
    ox,oy = reflection["origin"]
    gx = (region[0]*samples+bx+0.5)/samples*kx-ox-0.5
    gy = float((region[1]*samples+by+0.5)/samples*ky-oy-0.5)
    source = reflection["color"]
    sh,sw = source.shape[:2]
    x0 = np.clip(np.floor(gx).astype(np.int64),0,sw-1)
    x1 = np.minimum(x0+1,sw-1)
    y0 = min(max(int(np.floor(gy)),0),sh-1)
    y1 = min(y0+1,sh-1)
    fx = (gx-np.floor(gx))[:,None]
    fy = gy-np.floor(gy)
    sampled = ((1-fx)*(1-fy)*source[y0,x0]+fx*(1-fy)*source[y0,x1]
               +(1-fx)*fy*source[y1,x0]+fx*fy*source[y1,x1])
    weight = reflection["reflectance"]
    rgb = rgb*(1-weight)+sampled*weight
    xi = np.clip(np.floor(gx+0.5).astype(np.int64),0,sw-1)
    yi = min(max(int(np.floor(gy+0.5)),0),sh-1)
    buffers["reflection_id"][by,bx] = reflection["object_id"][yi,xi]
    buffers["reflection_world"][by,bx] = reflection["world"][yi,xi]
    return rgb


def raster_triangle(vertex,object_id,surface,texture,light,buffers,width,height,region,samples,reflection=None):
    clip = vertex[:,:4]
    if np.any(clip[:,3] <= 1e-12):
        return 0
    ndc = clip[:,:3]/clip[:,3,None]
    screen = np.column_stack([(ndc[:,0]+1)*width*samples/2,(1-ndc[:,1])*height*samples/2])
    signed_area = edge(screen[0],screen[1],screen[2,0],screen[2,1])
    if abs(signed_area) < 1e-10:
        return 0
    # Front faces are CCW in the image's bottom-left coordinate convention.
    if not surface["double_sided"] and signed_area >= 0:
        return 0
    if signed_area < 0:
        vertex = vertex[[0,2,1]]
        clip,ndc,screen = clip[[0,2,1]],ndc[[0,2,1]],screen[[0,2,1]]
        signed_area = -signed_area
    rx,ry,rw,rh = region
    low = np.maximum(np.floor(screen.min(axis=0)).astype(int),[rx*samples,ry*samples])
    high = np.minimum(np.ceil(screen.max(axis=0)).astype(int),[(rx+rw)*samples,(ry+rh)*samples])
    if np.any(high <= low):
        return 0
    count = 0
    # Bounded scanline workspace; framebuffer memory remains the dominant cost.
    for py in range(low[1],high[1]):
        px = np.arange(low[0],high[0])
        xx,yy = px+0.5,py+0.5
        edges = np.stack([edge(screen[1],screen[2],xx,yy),edge(screen[2],screen[0],xx,yy),edge(screen[0],screen[1],xx,yy)],axis=1)
        inside = np.ones(len(px),dtype=bool)
        for i,(a,b) in enumerate([(screen[1],screen[2]),(screen[2],screen[0]),(screen[0],screen[1])]):
            dx,dy = b-a
            top_left = dy < 0 or (dy == 0 and dx > 0)
            inside &= (edges[:,i] > 1e-9) | ((np.abs(edges[:,i])<=1e-9) & top_left)
        bary = edges/signed_area
        z = bary @ ndc[:,2]
        by,bx = py-ry*samples,px-rx*samples
        previous = buffers["depth"][by,bx]
        # Equal-depth fragments retain scene traversal order.
        win = inside & (z < previous-1e-12)
        if not np.any(win):
            continue
        bx,z,bary = bx[win],z[win],bary[win]
        weights = bary/clip[:,3]
        weights /= weights.sum(axis=1,keepdims=True)
        attributes = weights @ vertex[:,4:]
        rgb,keep,normals = shade(surface,attributes[:,3:6],attributes[:,6:8],texture,light)
        bx,z,attributes,rgb,normals = bx[keep],z[keep],attributes[keep],rgb[keep],normals[keep]
        if reflection is not None and len(bx):
            rgb = sample_reflection(reflection,rgb,bx,by,region,samples,buffers)
        elif "reflection_id" in buffers and len(bx):
            # A nearer non-mirror surface replaces whatever the mirror left here.
            buffers["reflection_id"][by,bx] = 0
            buffers["reflection_world"][by,bx] = 0
        buffers["depth"][by,bx] = z
        buffers["color"][by,bx] = rgb
        buffers["normal"][by,bx] = normals
        buffers["world"][by,bx] = attributes[:,:3]
        buffers["object_id"][by,bx] = object_id
        count += len(bx)
    return count


def render_reflections(compiled,objects,request,view,projection,region):
    """Render one mirrored pass per visible mirror and collect pickable metadata."""
    budget = request.get("reflection") or {}
    scale = budget.get("resolution_scale",1.0)
    enabled = budget.get("enabled",True)
    width,height = request["width"],request["height"]
    eye = np.asarray(compiled["scene"]["lens"]["eye"],dtype=np.float64)
    plans = plan_reflections(objects,eye,view,projection,width,height,scale,region) if enabled else {}
    reflections,entries = {},{}
    for obj in objects:
        spec = obj.get("mirror")
        if spec is None:
            continue
        entry = {"node":obj["key"],"reflectance":spec["reflectance"],
                 "plane":{"normal":list(spec["normal"]),"offset":spec["offset"]},"reflection":None}
        entries[str(obj["id"])] = entry
        plan = plans.get(obj["id"])
        if plan is None:
            continue
        excluded = set(spec["exclude"])|{obj["key"]}
        included = None if spec["include"] is None else set(spec["include"])-{obj["key"]}
        sub_request = {"width":plan["width"],"height":plan["height"],"samples":1,"region":plan["region"]}
        sub_buffers,sub_meta = render_samples(compiled,sub_request,
                                              view=view @ reflection_matrix(spec["normal"],spec["offset"]),
                                              projection=projection,
                                              keep_planes=[(spec["normal"],spec["offset"])],
                                              include_keys=included,exclude_keys=excluded,mirrors=False)
        reflections[obj["id"]] = {"color":sub_buffers["color"],"object_id":sub_buffers["object_id"],
                                  "world":sub_buffers["world"],"scale":plan["scale"],
                                  "origin":(plan["region"][0],plan["region"][1]),
                                  "reflectance":spec["reflectance"]}
        entry["reflection"] = {"resolution":[plan["width"],plan["height"]],"region":plan["region"],
                               "scale":plan["scale"],"samples":1,"excluded":sorted(excluded),
                               "included":None if included is None else sorted(included),
                               "stats":sub_meta["stats"]}
    metadata = {"mirrors":entries,"reflection":{"enabled":bool(enabled),"resolution_scale":scale}}
    return reflections,metadata


def render_samples(compiled,request,*,view=None,projection=None,keep_planes=(),include_keys=None,exclude_keys=(),mirrors=True):
    width,height,samples = request["width"],request["height"],request["samples"]
    region = request.get("region") or (0,0,width,height)
    region = tuple(region)
    _,_,rw,rh = region
    estimated = rw*rh*samples*samples*128
    require(estimated <= 160*1024*1024,"render_budget","sample buffers exceed 160 MiB budget")
    require(compiled["triangles"] <= 60000,"triangle_budget","visible scene exceeds 60000 triangles")
    scene = compiled["scene"]
    if view is None or projection is None:
        view,projection = camera_matrices(scene["lens"],width/height)
    objects = [obj for obj in compiled["objects"]
               if (include_keys is None or obj["key"] in include_keys) and obj["key"] not in exclude_keys]
    reflections,mirror_metadata = {},{}
    if mirrors and any(obj.get("mirror") for obj in objects):
        reflections,mirror_metadata = render_reflections(compiled,objects,request,view,projection,region)
    buffers = allocate(rw,rh,samples,scene["background"],reflection=bool(reflections))
    textures = {key:texture_array(value["payload"]) for key,value in compiled["resources"].items() if value["kind"]=="texture"}
    planes = PLANES
    if keep_planes:
        inverse = np.linalg.inv(projection @ view).T
        planes = np.vstack([PLANES,*[inverse @ np.array([*normal,offset],dtype=np.float64) for normal,offset in keep_planes]])
    stats = {"input_triangles":sum(len(obj["triangles"]) for obj in objects),"clipped_triangles":0,"written_samples":0,"estimated_bytes":estimated}
    for obj in objects:
        positions = obj["positions"]
        homogeneous = np.column_stack([positions,np.ones(len(positions))])
        clip = homogeneous @ (projection @ view).T
        uv = obj["uv"] if obj["uv"] is not None else np.zeros((len(positions),2))
        packed = np.column_stack([clip,positions,obj["normals"],uv])
        texture = textures.get(obj["surface"]["texture"])
        reflection = reflections.get(obj["id"])
        for face in obj["triangles"]:
            for triangle in clip_triangle(packed[face],planes):
                stats["clipped_triangles"] += 1
                stats["written_samples"] += raster_triangle(triangle,obj["id"],obj["surface"],texture,scene["light"],buffers,width,height,region,samples,reflection)
    metadata = {"view":view.tolist(),"projection":projection.tolist(),"region":list(region),"stats":stats}
    metadata.update(mirror_metadata)
    return buffers,metadata
