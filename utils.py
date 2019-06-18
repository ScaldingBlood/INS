import numpy as np
from numpy.linalg import norm
from numpy import array, cross, dot, float64, hypot, zeros
from math import acos, atan2, cos, pi, sin, asin, sqrt


def cross_product(m):
    return np.matrix([
        [0, -m[2], m[1]],
        [m[2], 0, -m[0]],
        [-m[1], m[0], 0]
    ])

def array2matrix(a):
    return np.matrix([a[0], a[1], a[2]]).T

def R_2vect(R, vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)

    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
    return R

def vec_2q(vector_orig, vector_fin):

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin)) /2

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)

    # Calculate the q.
    return np.mat([ca, sa*x, sa*y, sa*z]).T

def multiple_q(q1, q2):
    res = np.mat([
    q1[0, 0] * q2[0, 0] - q1[1, 0] * q2[1, 0] - q1[2, 0] * q2[2, 0] - q1[3, 0] * q2[3, 0],
    q1[0, 0] * q2[1, 0] + q1[1, 0] * q2[0, 0] + q1[2, 0] * q2[3, 0] - q1[3, 0] * q2[2, 0],
    q1[0, 0] * q2[2, 0] + q1[2, 0] * q2[0, 0] - q1[1, 0] * q2[3, 0] + q1[3, 0] * q2[1, 0],
    q1[0, 0] * q2[3, 0] + q1[3, 0] * q2[0, 0] - q1[2, 0] * q2[1, 0] + q1[1, 0] * q2[2, 0]]).T
    res = res / np.linalg.norm(res)
    return res


def calculate_angle(arr1, arr2):
    arr1 = arr1 / np.linalg.norm(arr1)
    arr2 = arr2 / np.linalg.norm(arr2)
    return acos(sum(arr1 * arr2)) / pi * 180

def q2R(q):
    return np.matrix([
        [1 - 2 * q[2, 0] * q[2, 0] - 2 * q[3, 0] * q[3, 0],
         2 * q[1, 0] * q[2, 0] - 2 * q[0, 0] * q[3, 0],
         2 * q[1, 0] * q[3, 0] + 2 * q[0, 0] * q[2, 0]],
        [2 * q[1, 0] * q[2, 0] + 2 * q[0, 0] * q[3, 0],
         1 - 2 * q[1, 0] * q[1, 0] - 2 * q[3, 0] * q[3, 0],
         2 * q[2, 0] * q[3, 0] - 2 * q[0, 0] * q[1, 0]],
        [2 * q[1, 0] * q[3, 0] - 2 * q[0, 0] * q[2, 0],
         2 * q[2, 0] * q[3, 0] + 2 * q[0, 0] * q[1, 0],
         1 - 2 * q[1, 0] * q[1, 0] - 2 * q[2, 0] * q[2, 0]]
    ])

def vec2q(vec):
    norm = np.linalg.norm(vec)
    vec = vec / norm if norm > 0 else [0, 0, 0]
    q = np.mat([cos(norm / 2), vec[0] * sin(norm / 2), vec[1] * sin(norm / 2),
                      vec[2] * sin(norm / 2)]).T
    return q

def q2ang(q):
    return [atan2(2 * (q[0, 0] * q[1, 0] + q[2, 0] * q[3, 0]),
                (1 - 2 * (q[1, 0] * q[1, 0] + q[2, 0] * q[2, 0]))) * 180 / pi,
     asin(2 * (q[0, 0] * q[2, 0] - q[1, 0] * q[3, 0])) * 180 / pi,
     atan2(2 * (q[0, 0] * q[3, 0] + q[1, 0] * q[2, 0]),
                (1 - 2 * (q[2, 0] * q[2, 0] + q[3, 0] * q[3, 0]))) * 180 / pi]

def angle2q(ang):
    a = ang[0] / 2
    b = ang[1] / 2
    c = ang[2] / 2
    return np.mat([cos(a) * cos(b) * cos(c) + sin(a) * sin(b) * sin(c),
                   sin(a) * cos(b) * cos(c) - cos(a) * sin(b) * sin(c),
                   cos(a) * sin(b) * cos(c) + sin(a) * cos(b) * sin(c),
                   cos(a) * cos(b) * sin(c) - sin(a) * sin(b) * cos(c)]).T


if __name__ == '__main__':
    r = np.eye(3)
    a = [1, 2, 3]
    a = a / np.linalg.norm(a)
    print(a)
    b = [5, 2, 4]
    b = b / np.linalg.norm(b)
    print(b)
    q = vec_2q(a ,b)
    print(q2R(q) * np.mat([a[0], a[1], a[2]]).T)
    invq = -q
    invq[0, 0] = -invq[0, 0]
    print(multiple_q(multiple_q(q, np.mat([0, a[0], a[1], a[2]]).T), invq))

    vec1 = [10, -30, -50]
    vec2 = [30, 10, -50]
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    print(vec_2q(vec1, vec2))
    print(q2R(vec_2q(vec1, vec2)))
    print(q2R(vec_2q(vec1, vec2)) * np.mat([10, -30, -50]).T)
    print(q2ang(vec_2q(vec1, vec2)))

    vec3 = [-16, 26, -50]
    R = q2R(angle2q([0, 0, 1 / 6 * pi])).T
    print(R * np.mat(vec3).T)