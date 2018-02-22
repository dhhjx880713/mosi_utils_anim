import numpy as np
from math import acos, cos, sin, atan
from ..external.transformations import quaternion_about_axis, quaternion_inverse, quaternion_multiply, quaternion_from_axis_angle

def exponential(q):
    """ https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation
        chapter 2 of lee 2000
    """
    v = np.array([q[1], q[2], q[3]])
    theta = np.linalg.norm(v)
    return (cos(theta) + v/ theta * sin(theta))# * np.exp(q[0])

def log(q):
    """ https://math.stackexchange.com/questions/1030737/exponential-function-of-quaternion-derivation
      chapter 2 of lee 2000
    """
    v = np.array([q[1], q[2], q[3]])
    w = q[0]
    if w == 0:
        return np.pi/2 * v
    elif 0 < abs(w) < 1:
        theta = np.linalg.norm(v)
        return v / theta * atan(theta/w)
    else: # w == 1
        return 0

def dist(q1, q2):
    """  chapter 2 of lee 2000"""
    return np.linalg.norm(log(quaternion_multiply(quaternion_inverse(q1), q2)))

def rotate_vector_by_quaternion(v, q):
    vq = [0, v[0], v[1], v[2]]
    v_prime = quaternion_multiply(q, quaternion_multiply(vq , quaternion_inverse(q)))[1:]
    v_prime /= np.linalg.norm(v_prime)
    return v_prime


def apply_conic_constraint(q, ref_q, axis, k):
    """ lee 2000 p. 48"""
    q0 = ref_q
    w = axis
    rel_q = quaternion_multiply(quaternion_inverse(q0), q)
    rel_q /np.linalg.norm(rel_q)
    w_prime = rotate_vector_by_quaternion(w, ref_q)
    phi = acos(np.dot(w, w_prime))
    if 0 < phi > k: # apply
        delta_phi = k - phi
        v = np.cross(w, w_prime)
        delta_q = quaternion_about_axis(delta_phi, v)
        q = quaternion_multiply(delta_q, q)
    return q


def apply_axial_constraint(q, ref_q, axis, k1, k2):
    """ lee 2000 p. 48"""
    q0 = ref_q
    w = axis
    rel_q = quaternion_multiply(quaternion_inverse(q0), q)
    rel_q /np.linalg.norm(rel_q)
    w_prime = rotate_vector_by_quaternion(w, rel_q)
    v = np.cross(w, w_prime)
    phi = acos(np.dot(w, w_prime))

    #remove rotation around v
    # psi_q = exp(-phi*v)
    inv_phi_q = quaternion_about_axis(-phi, v)
    psi_q = quaternion_multiply(inv_phi_q, ref_q)

    # get angle around w from psi_q
    w_prime2 = rotate_vector_by_quaternion(w, psi_q)
    w_prime2 /= np.linalg.norm(w_prime2)
    psi = acos(np.dot(w, w_prime2))

    if k1 < psi > k2: # apply
        delta_psi1 = k1 - psi
        delta_psi2 = k2 - psi
        delta_psi = min(delta_psi1, delta_psi2)
        delta_q = quaternion_about_axis(delta_psi, w)
        q = quaternion_multiply(delta_q, q)
    return q


def apply_spherical_constraint(q, ref_q, axis, k):
    """ lee 2000 p. 48"""
    q0 = ref_q
    w = axis
    rel_q = quaternion_multiply(quaternion_inverse(q0), q)
    rel_q /np.linalg.norm(rel_q)
    w_prime = rotate_vector_by_quaternion(w, rel_q)
    v = np.cross(w, w_prime)
    log_rel_q = log(rel_q)
    phi = acos(np.dot(w, w_prime))
    if 2*np.linalg.norm(log_rel_q) > k: # apply
        delta_phi = k - phi
        delta_q = quaternion_about_axis(delta_phi, v)
        q = quaternion_multiply(delta_q, q)
    return q