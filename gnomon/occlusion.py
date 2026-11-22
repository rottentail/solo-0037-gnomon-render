"""Screen-space ambient occlusion over the resolved nearest-surface channels.

The estimator works in world space on the resolved depth layer: each covered
pixel receives a hemisphere of sample offsets around its (camera-facing)
normal, every sample is projected through the render camera, and the stored
nearest surface at that pixel is tested as a potential occluder. Two rules
keep the result meaningful for illustrators:

- Occlusion is decided by view-space depth, which is projection-independent,
  so orthographic and perspective lenses share the same geometry test.
- A world-space range check weights occluders by true distance to the sample
  point, so surfaces that merely overlap in projection but lie farther than
  the detail radius never read as contact.

The per-pixel rotation noise is a hash of the full-frame pixel coordinate, so
results are reproducible and independent of the rendered region.
"""
import numpy as np


def kernel_offsets(count):
    # Deterministic hemisphere kernel (z+). Lengths are biased toward the
    # origin so tight contact detail is sampled more densely than the rim.
    rng = np.random.default_rng(20240617)
    directions = rng.normal(size=(count,3))
    directions[:,2] = np.abs(directions[:,2])
    directions /= np.linalg.norm(directions,axis=1,keepdims=True)
    scale = 0.1+0.9*rng.random(count)**2
    return directions*scale[:,None]


def pixel_noise(fx,fy):
    # Frame-aligned hash: identical for the same full-frame pixel regardless
    # of which region was rendered.
    def frac(value):
        return value-np.floor(value)
    return (frac(np.sin(fx*12.9898+fy*78.233)*43758.5453),
            frac(np.sin(fx*39.3468+fy*11.1351)*24634.6345))


def occlusion_margin(lens,width,height,radius,bias):
    # Pixels outside the requested region that occlusion sampling can read.
    # Perspective: a sample within `reach` of a receiver at depth >= near
    # projects at most cot*reach/(near-reach) NDC units away; beyond that the
    # only safe margin is the whole frame.
    reach = radius+bias
    if lens["projection"]=="perspective":
        near = lens["near"]
        if reach >= near:
            return width,height
        cot = 1/np.tan(np.radians(lens["fov"])/2)
        mx = int(np.ceil((cot/(width/height))*reach/(near-reach)*width/2))+1
        my = int(np.ceil(cot*reach/(near-reach)*height/2))+1
    else:
        span = lens["span"]
        mx = int(np.ceil(reach*width/(span*(width/height))))+1
        my = int(np.ceil(reach*height/span))+1
    return min(mx,width),min(my,height)


def expand_region(region,margin,width,height):
    x,y,w,h = region
    mx,my = margin
    x0,y0 = max(0,x-mx),max(0,y-my)
    x1,y1 = min(width,x+w+mx),min(height,y+h+my)
    return (x0,y0,x1-x0,y1-y0)


def compute_occlusion(world,normal,covered,view,projection,projection_kind,width,height,origin,params):
    h,w = world.shape[:2]
    visibility = np.ones((h,w))
    if not covered.any():
        return visibility
    radius = params["radius"]
    bias = params["bias"] if params["bias"] is not None else 0.025*radius
    epsilon = 1e-3*radius
    rotation = view[:3,:3]
    translation = view[:3,3]
    # Occluder view depth per pixel; background never occludes.
    occluder_depth = -(world @ rotation.T + translation)[...,2]
    occluder_depth = np.where(covered,occluder_depth,np.inf)
    points = world[covered]
    normals = normal[covered]
    # Flip normals toward the camera so the back face of a double-sided sheet
    # does not occlude itself.
    if projection_kind=="orthographic":
        toward = np.tile(rotation[2],(len(points),1))
    else:
        toward = -rotation.T @ translation-points
        length = np.linalg.norm(toward,axis=1,keepdims=True)
        toward = np.divide(toward,length,out=np.zeros_like(toward),where=length>1e-12)
    facing = np.einsum("ij,ij->i",normals,toward)
    normals = np.where(facing[:,None]<0,-normals,normals)
    # Per-pixel kernel rotation from frame-aligned noise.
    yy,xx = np.indices((h,w))
    h1,h2 = pixel_noise((xx+origin[0])[covered].astype(np.float64),(yy+origin[1])[covered].astype(np.float64))
    random_vec = np.column_stack([np.cos(2*np.pi*h1),np.sin(2*np.pi*h1),2*h2-1])
    tangent = random_vec-normals*np.einsum("ij,ij->i",random_vec,normals)[:,None]
    tangent_len = np.linalg.norm(tangent,axis=1,keepdims=True)
    axis = np.where(np.abs(normals[:,2:3])<0.9,np.array([0.0,0.0,1.0]),np.array([1.0,0.0,0.0]))
    fallback = np.cross(normals,axis)
    fallback /= np.linalg.norm(fallback,axis=1,keepdims=True)
    tangent = np.where(tangent_len>1e-9,tangent/np.where(tangent_len>1e-9,tangent_len,1),fallback)
    bitangent = np.cross(normals,tangent)
    kernel = kernel_offsets(params["samples"])
    base = points+normals*bias
    view_projection = projection @ view
    ox,oy = origin
    occlusion = np.zeros(len(points))
    for kx,ky,kz in kernel:
        sample = base+(tangent*kx+bitangent*ky+normals*kz)*radius
        sample_depth = -(sample @ rotation.T + translation)[:,2]
        clip = np.column_stack([sample,np.ones(len(sample))]) @ view_projection.T
        wclip = clip[:,3]
        valid = wclip>1e-9
        ndc = clip[:,:3]/np.where(valid,wclip,1.0)[:,None]
        bx = np.floor((ndc[:,0]+1)*width/2).astype(np.int64)-ox
        by = np.floor((1-ndc[:,1])*height/2).astype(np.int64)-oy
        valid &= (bx>=0)&(bx<w)&(by>=0)&(by<h)
        if not valid.any():
            continue
        idx = np.nonzero(valid)[0]
        blocked = sample_depth[idx] > occluder_depth[by[idx],bx[idx]]+epsilon
        if not blocked.any():
            continue
        hit = idx[blocked]
        distance = np.linalg.norm(world[by[hit],bx[hit]]-sample[hit],axis=1)
        # Range check: only geometry within the detail radius reads as contact.
        occlusion[hit] += np.clip(1-distance/radius,0,1)
    visibility[covered] = 1-params["strength"]*np.clip(occlusion/len(kernel),0,1)
    return visibility
