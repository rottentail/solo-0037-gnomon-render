import numpy as np
from .camera import camera_matrices
from .clipping import clip_triangle
from .color import srgb_to_linear
from .texture import texture_array
from .shading import shade
from .errors import require


def allocate(width,height,samples,background):
    shape = (height*samples,width*samples)
    return {"color":np.broadcast_to(srgb_to_linear(background),(*shape,3)).copy(),
            "depth":np.full(shape,np.inf),"normal":np.zeros((*shape,3)),
            "object_id":np.zeros(shape,dtype=np.int32),"world":np.zeros((*shape,3))}


def edge(a,b,x,y):
    return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])


def raster_triangle(vertex,object_id,surface,texture,light,buffers,width,height,region,samples):
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
        buffers["depth"][by,bx] = z
        buffers["color"][by,bx] = rgb
        buffers["normal"][by,bx] = normals
        buffers["world"][by,bx] = attributes[:,:3]
        buffers["object_id"][by,bx] = object_id
        count += len(bx)
    return count


def render_samples(compiled,request):
    width,height,samples = request["width"],request["height"],request["samples"]
    region = request.get("region") or (0,0,width,height)
    _,_,rw,rh = region
    estimated = rw*rh*samples*samples*128
    require(estimated <= 160*1024*1024,"render_budget","sample buffers exceed 160 MiB budget")
    require(compiled["triangles"] <= 60000,"triangle_budget","visible scene exceeds 60000 triangles")
    scene = compiled["scene"]
    view,projection = camera_matrices(scene["lens"],width/height)
    buffers = allocate(rw,rh,samples,scene["background"])
    textures = {key:texture_array(value["payload"]) for key,value in compiled["resources"].items() if value["kind"]=="texture"}
    stats = {"input_triangles":compiled["triangles"],"clipped_triangles":0,"written_samples":0,"estimated_bytes":estimated}
    for obj in compiled["objects"]:
        positions = obj["positions"]
        homogeneous = np.column_stack([positions,np.ones(len(positions))])
        clip = homogeneous @ (projection @ view).T
        uv = obj["uv"] if obj["uv"] is not None else np.zeros((len(positions),2))
        packed = np.column_stack([clip,positions,obj["normals"],uv])
        texture = textures.get(obj["surface"]["texture"])
        for face in obj["triangles"]:
            for triangle in clip_triangle(packed[face]):
                stats["clipped_triangles"] += 1
                stats["written_samples"] += raster_triangle(triangle,obj["id"],obj["surface"],texture,scene["light"],buffers,width,height,region,samples)
    return buffers,{"view":view.tolist(),"projection":projection.tolist(),"region":list(region),"stats":stats}
