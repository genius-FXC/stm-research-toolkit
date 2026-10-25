"""Quarter-turn display coordinates; original sampling coordinates remain intact."""
import numpy as np
def rotate_points(xy,nx,ny,turns):
    p=np.asarray(xy);x,y=p[...,0],p[...,1]
    return np.stack([(x,y),(ny-1-y,x),(nx-1-x,ny-1-y),(y,nx-1-x)][turns%4],axis=-1)
