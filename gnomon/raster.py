import numpy as np
from .camera import camera_matrices
from .clipping import clip_triangle
from .color import srgb_to_linear
from .texture import texture_array
from .shading import shade
from .errors import require

# Per sub-sample we keep, at most, the nearest MAX_TRANSLUCENT_LAYERS
# translucent fragments in front of (or without) an opaque winner. Layers
# beyond this cap are dropped and counted; see render metadata overflow stats.
MAX_TRANSLUCENT_LAYERS = 8
# Conservative bookkeeping per stored layer record (int16 ids/depth/rgb/alpha).
LAYER_RECORD_BYTES = 96
# Opaque-only framebuffer footprint estimate, retained from the baseline.
BASE_SAMPLE_BYTES = 128
BUFFER_BUDGET = 160*1024*1024
DEPTH_EPSILON = 1e-12


def sample_buffer_bytes(rw, rh, samples, has_translucent):
    subpixels = rw*rh*samples*samples
    per_sample = BASE_SAMPLE_BYTES
    if has_translucent:
        per_sample += MAX_TRANSLUCENT_LAYERS*LAYER_RECORD_BYTES
    return subpixels*per_sample


def allocate(width,height,samples,background,has_translucent):
    shape = (height*samples,width*samples)
    buffers = {"color":np.broadcast_to(srgb_to_linear(background),(*shape,3)).copy(),
               "depth":np.full(shape,np.inf),"normal":np.zeros((*shape,3)),
               "object_id":np.zeros(shape,dtype=np.int32),"world":np.zeros((*shape,3)),
               "background":srgb_to_linear(background),"has_translucent":has_translucent}
    if has_translucent:
        # Fixed-capacity A-buffer slots kept in nearest-first depth order;
        # id 0 and infinite depth mark an unused slot.
        buffers["layer_count"] = np.zeros(shape,dtype=np.int16)
        buffers["layer_depth"] = np.full((*shape,MAX_TRANSLUCENT_LAYERS),np.inf)
        buffers["layer_id"] = np.zeros((*shape,MAX_TRANSLUCENT_LAYERS),dtype=np.int16)
        buffers["layer_rgb"] = np.zeros((*shape,MAX_TRANSLUCENT_LAYERS,3))
        buffers["layer_alpha"] = np.zeros((*shape,MAX_TRANSLUCENT_LAYERS))
        buffers["layer_world"] = np.zeros((*shape,MAX_TRANSLUCENT_LAYERS,3))
        buffers["layer_normal"] = np.zeros((*shape,MAX_TRANSLUCENT_LAYERS,3))
    return buffers


def edge(a,b,x,y):
    return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0])


def raster_triangle(vertex,object_id,surface,texture,light,buffers,width,height,region,samples,translucent):
    clip = vertex[:,:4]
    if np.any(clip[:,3] <= 1e-12):
        return 0,0
    ndc = clip[:,:3]/clip[:,3,None]
    screen = np.column_stack([(ndc[:,0]+1)*width*samples/2,(1-ndc[:,1])*height*samples/2])
    signed_area = edge(screen[0],screen[1],screen[2,0],screen[2,1])
    if abs(signed_area) < 1e-10:
        return 0,0
    # Front faces are CCW in the image's bottom-left coordinate convention.
    if not surface["double_sided"] and signed_area >= 0:
        return 0,0
    if signed_area < 0:
        vertex = vertex[[0,2,1]]
        clip,ndc,screen = clip[[0,2,1]],ndc[[0,2,1]],screen[[0,2,1]]
        signed_area = -signed_area
    rx,ry,rw,rh = region
    low = np.maximum(np.floor(screen.min(axis=0)).astype(int),[rx*samples,ry*samples])
    high = np.minimum(np.ceil(screen.max(axis=0)).astype(int),[(rx+rw)*samples,(ry+rh)*samples])
    if np.any(high <= low):
        return 0,0
    written = overflow = 0
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
        if not np.any(inside):
            continue
        bary = edges/signed_area
        z = bary @ ndc[:,2]
        by,bx = py-ry*samples,px-rx*samples
        if translucent:
            # Translucent fragments never replace the opaque winner, but any
            # fragment in front of it (or with no winner behind) is stored;
            # depth-sorted compositing decides the final color.
            win = inside & (z < buffers["depth"][by,bx]+DEPTH_EPSILON)
            if not np.any(win):
                continue
            bx,z,bary = bx[win],z[win],bary[win]
            weights = bary/clip[:,3]
            weights /= weights.sum(axis=1,keepdims=True)
            attributes = weights @ vertex[:,4:]
            rgb,alpha,normals = shade(surface,attributes[:,3:6],attributes[:,6:8],texture,light)
            alive = alpha > 0
            bx,z,attributes,rgb,alpha,normals = (bx[alive],z[alive],attributes[alive],
                                                 rgb[alive],alpha[alive],normals[alive])
            if len(bx) == 0:
                continue
            # Sorted insertion into the fixed-capacity per-cell layer slots.
            # Each winning cell receives at most one fragment per scanline.
            counts = buffers["layer_count"][by,bx]
            occupied = np.arange(MAX_TRANSLUCENT_LAYERS)[None,:] < counts[:,None]
            depths = buffers["layer_depth"][by,bx]
            # Equal-depth layers keep traversal order: new fragments go after
            # existing ones at the same depth.
            after = occupied & (depths <= z[:,None]-DEPTH_EPSILON)
            positions = after.sum(axis=1)
            full = counts >= MAX_TRANSLUCENT_LAYERS
            accepted = ~full
            overflow += int(full.sum())
            if not np.any(accepted):
                continue
            bx,z = bx[accepted],z[accepted]
            positions,counts = positions[accepted],counts[accepted]
            rgb,alpha = rgb[accepted],alpha[accepted]
            attributes,normals = attributes[accepted],normals[accepted]
            for index in range(len(bx)):
                row,col,pos = by,int(bx[index]),int(positions[index])
                # Rotate the cell's slots upward, then overwrite the insertion
                # point. Explicit per-cell indexing avoids overlapping whole-row
                # slice aliasing.
                if pos < MAX_TRANSLUCENT_LAYERS-1:
                    buffers["layer_depth"][row,col,pos+1:] = buffers["layer_depth"][row,col,pos:-1]
                    buffers["layer_id"][row,col,pos+1:] = buffers["layer_id"][row,col,pos:-1]
                    buffers["layer_rgb"][row,col,pos+1:] = buffers["layer_rgb"][row,col,pos:-1]
                    buffers["layer_alpha"][row,col,pos+1:] = buffers["layer_alpha"][row,col,pos:-1]
                    buffers["layer_world"][row,col,pos+1:] = buffers["layer_world"][row,col,pos:-1]
                    buffers["layer_normal"][row,col,pos+1:] = buffers["layer_normal"][row,col,pos:-1]
                buffers["layer_depth"][row,col,pos] = z[index]
                buffers["layer_id"][row,col,pos] = object_id
                buffers["layer_rgb"][row,col,pos] = rgb[index]
                buffers["layer_alpha"][row,col,pos] = alpha[index]
                buffers["layer_world"][row,col,pos] = attributes[index,:3]
                buffers["layer_normal"][row,col,pos] = normals[index]
                buffers["layer_count"][row,col] = counts[index]+1
            written += len(bx)
            continue
        # Equal-depth fragments retain scene traversal order.
        win = inside & (z < buffers["depth"][by,bx]-DEPTH_EPSILON)
        if not np.any(win):
            continue
        bx,z,bary = bx[win],z[win],bary[win]
        weights = bary/clip[:,3]
        weights /= weights.sum(axis=1,keepdims=True)
        attributes = weights @ vertex[:,4:]
        rgb,alpha,normals = shade(surface,attributes[:,3:6],attributes[:,6:8],texture,light)
        keep = alpha > 0
        bx,z,attributes,rgb,normals = bx[keep],z[keep],attributes[keep],rgb[keep],normals[keep]
        buffers["depth"][by,bx] = z
        buffers["color"][by,bx] = rgb
        buffers["normal"][by,bx] = normals
        buffers["world"][by,bx] = attributes[:,:3]
        buffers["object_id"][by,bx] = object_id
        written += len(bx)
    return written,overflow


def render_samples(compiled,request):
    width,height,samples = request["width"],request["height"],request["samples"]
    region = request.get("region") or (0,0,width,height)
    _,_,rw,rh = region
    has_translucent = compiled["has_translucent"]
    estimated = sample_buffer_bytes(rw,rh,samples,has_translucent)
    require(estimated <= BUFFER_BUDGET,"render_budget","sample buffers exceed 160 MiB budget")
    require(compiled["triangles"] <= 60000,"triangle_budget","visible scene exceeds 60000 triangles")
    scene = compiled["scene"]
    view,projection = camera_matrices(scene["lens"],width/height)
    buffers = allocate(rw,rh,samples,scene["background"],has_translucent)
    textures = {key:texture_array(value["payload"]) for key,value in compiled["resources"].items() if value["kind"]=="texture"}
    stats = {"input_triangles":compiled["triangles"],"clipped_triangles":0,
             "written_samples":0,"estimated_bytes":estimated,
             "translucent_layers_max":MAX_TRANSLUCENT_LAYERS,
             "translucent_overflow_samples":0}
    # Pass 1 writes opaque/cutout surfaces exactly as the baseline nearest-depth
    # path. Pass 2 inserts translucent fragments into the bounded layer slots.
    # Final compositing is depth-sorted, so scene node order cannot fabricate
    # the color result.
    for pass_index in range(2):
        translucent_pass = bool(pass_index)
        if translucent_pass and not has_translucent:
            continue
        for obj in compiled["objects"]:
            if (obj["surface"]["opacity"] < 1.0) != translucent_pass:
                continue
            positions = obj["positions"]
            homogeneous = np.column_stack([positions,np.ones(len(positions))])
            clip = homogeneous @ (projection @ view).T
            uv = obj["uv"] if obj["uv"] is not None else np.zeros((len(positions),2))
            packed = np.column_stack([clip,positions,obj["normals"],uv])
            texture = textures.get(obj["surface"]["texture"])
            for face in obj["triangles"]:
                for triangle in clip_triangle(packed[face]):
                    stats["clipped_triangles"] += 1
                    written,overflow = raster_triangle(triangle,obj["id"],obj["surface"],texture,
                                                       scene["light"],buffers,width,height,region,samples,
                                                       translucent_pass)
                    stats["written_samples"] += written
                    stats["translucent_overflow_samples"] += overflow
    return buffers,{"view":view.tolist(),"projection":projection.tolist(),"region":list(region),"stats":stats}
