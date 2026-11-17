import numpy as np
from .math3d import unit
from .errors import require


def camera_matrices(lens, aspect):
    eye, target = np.asarray(lens["eye"]), np.asarray(lens["target"])
    forward = unit(target-eye)
    side = unit(np.cross(forward,np.asarray(lens["up"])))
    up = np.cross(side,forward)
    view = np.eye(4)
    view[:3,:3] = np.stack([side,up,-forward])
    view[:3,3] = -view[:3,:3] @ eye
    n, f = lens["near"], lens["far"]
    if lens["projection"] == "perspective":
        cot = 1/np.tan(np.radians(lens["fov"])/2)
        proj = np.array([[cot/aspect,0,0,0],[0,cot,0,0],[0,0,-(f+n)/(f-n),-2*f*n/(f-n)],[0,0,-1,0]])
    else:
        h = lens["span"]
        proj = np.array([[2/(h*aspect),0,0,0],[0,2/h,0,0],[0,0,-2/(f-n),-(f+n)/(f-n)],[0,0,0,1]])
    return view,proj


def unproject(x,y,depth,width,height,view,projection):
    ndc = np.array([2*(x+0.5)/width-1,1-2*(y+0.5)/height,depth,1])
    point = np.linalg.inv(projection @ view) @ ndc
    require(abs(point[3]) > 1e-12,"unproject","point at infinity")
    return (point[:3]/point[3]).tolist()
