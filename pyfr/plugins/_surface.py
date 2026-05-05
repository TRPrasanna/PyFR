import numpy as np


def cross_fluxpts(r, f):
    """Compute r x f for arrays shaped (ndims, nfpts, neles)."""
    if r.shape != f.shape:
        raise ValueError('Cross-product operands must have matching shapes')

    if r.shape[0] == 2:
        return (r[0]*f[1] - r[1]*f[0])[None, ...]
    elif r.shape[0] == 3:
        return np.cross(r, f, axisa=0, axisb=0, axisc=0)
    else:
        raise ValueError('Cross products require two or three dimensions')
