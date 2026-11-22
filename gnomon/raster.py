import numpy as np
from .camera import camera_matrices
from .clipping import clip_triangle
from .color import srgb_to_linear
from .texture import texture_array
from .shading import shade
from .decal import projection_frame, apply_decal
from .errors import require


def allocate(width,height,samples,background):
    shape = (height*samples,width*samples)
    return {"color":np.broadcast_to(srgb_to_linear(background),(*shape,3)).copy(),
            "depth":np.full(shape,np.inf),"normal":np.zeros((*shape,3)),
            "object_id":np.zeros(shape,dtype=np.int32),"world":np.zeros((*shape,3)),
            "decal":np.zeros(shape)}


def edge(a,b,x,y):
    return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])


def scan_triangle(vertex, double_sided, width, height, region, samples):
    """Clip-space setup shared by the material pass and the decal pass."""
    clip = vertex[:,:4]
    if np.any(clip[:,3] <= 1e-12):
        return None
    ndc = clip[:,:3]/clip[:,3,None]
    screen = np.column_stack([(ndc[:,0]+1)*width*samples/2,(1-ndc[:,1])*height*samples/2])
    signed_area = edge(screen[0],screen[1],screen[2,0],screen[2,1])
    if abs(signed_area) < 1e-10:
        return None
    # Front faces are CCW in the image's bottom-left coordinate convention.
    if not double_sided and signed_area >= 0:
        return None
    if signed_area < 0:
        vertex = vertex[[0,2,1]]
        clip,ndc,screen = clip[[0,2,1]],ndc[[0,2,1]],screen[[0,2,1]]
        signed_area = -signed_area
    rx,ry,rw,rh = region
    low = np.maximum(np.floor(screen.min(axis=0)).astype(int),[rx*samples,ry*samples])
    high = np.minimum(np.ceil(screen.max(axis=0)).astype(int),[(rx+rw)*samples,(ry+rh)*samples])
    if np.any(high <= low):
        return None
    return vertex,clip,ndc,screen,signed_area,(rx,ry),low,high


def fragments(setup,samples):
    """Yield per-scanline fragment batches with depth, buffer coords and attributes."""
    vertex,clip,ndc,screen,signed_area,(rx,ry),low,high = setup
    for py in range(low[1],high[1]):
        px = np.arange(low[0],high[0])
        xx,yy = px+0.5,py+0.5
        edges = np.stack([edge(screen[1],screen[2],xx,yy),edge(screen[2],screen[0],xx,yy),edge(screen[0],screen[1],xx,yy)],axis=1)
        inside = np.ones(len(px),dtype=bool)
        for i,(a,b) in enumerate([(screen[1],screen[2]),(screen[2],screen[0]),(screen[0],screen[1])]):
            dx,dy = b-a
            top_left = dy < 0 or (dy == 0 and dx > 0)
            inside &= (edges[:,i] > 1e-9) | ((np.abs(edges[:,i])<=1e-9) & top_left)
        if not np.any(inside):
            continue
        bary = edges[inside]/signed_area
        z = bary @ ndc[:,2]
        by,bx = py-ry*samples,px[inside]-rx*samples
        weights = bary/clip[:,3]
        weights /= weights.sum(axis=1,keepdims=True)
        attributes = weights @ vertex[:,4:]
        yield by,bx,z,attributes


def raster_triangle(vertex,object_id,surface,texture,light,buffers,width,height,region,samples):
    setup = scan_triangle(vertex,surface["double_sided"],width,height,region,samples)
    if setup is None:
        return 0
    count = 0
    for by,bx,z,attributes in fragments(setup,samples):
        previous = buffers["depth"][by,bx]
        # Equal-depth fragments retain scene traversal order.
        win = z < previous-1e-12
        if not np.any(win):
            continue
        bx,z,attributes = bx[win],z[win],attributes[win]
        # fragments() already strips the 4 clip columns, so interpolated
        # attributes are world(0:3), normal(3:6), uv(6:8).
        rgb,keep,normals = shade(surface,attributes[:,3:6],attributes[:,6:8],texture,light)
        bx,z,attributes,rgb,normals = bx[keep],z[keep],attributes[keep],rgb[keep],normals[keep]
        buffers["depth"][by,bx] = z
        buffers["color"][by,bx] = rgb
        buffers["normal"][by,bx] = normals
        buffers["world"][by,bx] = attributes[:,:3]
        buffers["object_id"][by,bx] = object_id
        count += len(bx)
    return count


def raster_decal_triangle(vertex,object_id,decal,frame,texture,buffers,width,height,region,samples):
    # Rasterize both windings: the world-space facing test below is the single
    # authority for back-face rejection, so mirrored winding cannot hide faces.
    setup = scan_triangle(vertex,True,width,height,region,samples)
    if setup is None:
        return 0
    count = 0
    for by,bx,z,attributes in fragments(setup,samples):
        previous = buffers["depth"][by,bx]
        same_surface = buffers["object_id"][by,bx] == object_id
        # No penetration: only a fragment of the owning node that itself won
        # the depth buffer may receive the emblem. Subsamples won by another
        # object (occluders, coincident surfaces) clear any stale coverage so
        # the resolved pixel cannot report an emblem on the blocker.
        blocked = ~same_surface
        if np.any(blocked):
            buffers["decal"][by,bx[blocked]] = 0.0
        reachable = same_surface & np.isfinite(previous) & (np.abs(z-previous) <= 1e-10)
        if not np.any(reachable):
            continue
        bx,z,attributes = bx[reachable],z[reachable],attributes[reachable]
        # fragments() strips the clip columns: world(0:3), normal(3:6), uv(6:8).
        world,normals = attributes[:,:3],attributes[:,3:6]
        color,mask,_ = apply_decal(decal,frame,texture,world,normals,buffers["color"][by,bx])
        if not np.any(mask > 0):
            continue
        covered = mask > 0
        bx,color,mask = bx[covered],color[covered],mask[covered]
        buffers["color"][by,bx] = color
        buffers["decal"][by,bx] = np.maximum(buffers["decal"][by,bx],mask)
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
    stats = {"input_triangles":compiled["triangles"],"clipped_triangles":0,"written_samples":0,
             "decal_triangles":0,"decal_samples":0,"estimated_bytes":estimated}
    for obj in compiled["objects"]:
        positions = obj["positions"]
        homogeneous = np.column_stack([positions,np.ones(len(positions))])
        clip = homogeneous @ (projection @ view).T
        uv = obj["uv"] if obj["uv"] is not None else np.zeros((len(positions),2))
        # Packed layout after the 4 clip columns: world(3) at 4:7, normal(3)
        # at 7:10, uv(2) at 10:12.
        packed = np.column_stack([clip,positions,obj["normals"],uv])
        texture = textures.get(obj["surface"]["texture"])
        for face in obj["triangles"]:
            for triangle in clip_triangle(packed[face]):
                stats["clipped_triangles"] += 1
                stats["written_samples"] += raster_triangle(triangle,obj["id"],obj["surface"],texture,scene["light"],buffers,width,height,region,samples)
    # Second pass: projected overlays. The depth buffer from the material pass
    # supplies occlusion, so hidden and back-facing surfaces cannot be reached.
    for obj in compiled["objects"]:
        decal = obj.get("decal")
        if decal is None:
            continue
        frame = projection_frame(decal)
        decal_texture = textures[decal["texture"]]
        positions = obj["positions"]
        homogeneous = np.column_stack([positions,np.ones(len(positions))])
        clip = homogeneous @ (projection @ view).T
        uv = obj["uv"] if obj["uv"] is not None else np.zeros((len(positions),2))
        packed = np.column_stack([clip,positions,obj["normals"],uv])
        for face in obj["triangles"]:
            for triangle in clip_triangle(packed[face]):
                stats["decal_triangles"] += 1
                stats["decal_samples"] += raster_decal_triangle(triangle,obj["id"],decal,frame,decal_texture,buffers,width,height,region,samples)
    return buffers,{"view":view.tolist(),"projection":projection.tolist(),"region":list(region),"stats":stats}
