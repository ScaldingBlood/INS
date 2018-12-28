import numpy as np

def cross_product(m):
    return np.matrix([
        [0, -m[2], m[1]],
        [m[2], 0, -m[0]],
        [-m[1], m[0], 0]
    ])

def array2matrix(a):
    return np.matrix([a[0], a[1], a[2]]).T