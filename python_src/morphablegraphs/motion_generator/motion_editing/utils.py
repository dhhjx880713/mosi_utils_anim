import numpy as np
import math
from ...external.transformations import quaternion_multiply, quaternion_inverse
from ...animation_data.utils import euler_to_quaternion

LEN_QUATERNION = 4
LEN_TRANSLATION = 3


def normalize(v):
    return v/np.linalg.norm(v)


def quaternion_from_axis_angle(axis, angle):
    q = [1,0,0,0]
    if np.linalg.norm(axis) > 0:
        q[1] = axis[0] * math.sin(angle / 2)
        q[2] = axis[1] * math.sin(angle / 2)
        q[3] = axis[2] * math.sin(angle / 2)
        q[0] = math.cos(angle / 2)
        q = normalize(q)
    return q


def exp_map_to_quaternion(e):
    angle = np.linalg.norm(e)
    axis = e / angle
    q = quaternion_from_axis_angle(axis, angle)
    return q


def convert_exp_frame_to_quat_frame(skeleton, e):
    src_offset = 0
    dest_offset = 0
    n_joints = len(skeleton.animated_joints)
    q = np.zeros(n_joints*4)
    for node in skeleton.animated_joints:
        e_i = e[src_offset:src_offset+3]
        q[dest_offset:dest_offset+4] = exp_map_to_quaternion(e_i)
        src_offset += 3
        dest_offset += 4
    return q


def add_quat_frames(skeleton, q_frame1, q_frame2):
    src_offset = 0
    dest_offset = 3
    new_quat_frame = np.zeros(len(q_frame1))
    new_quat_frame[:3] = q_frame1[:3]
    for node in skeleton.animated_joints:
        new_q = quaternion_multiply(q_frame1[dest_offset:dest_offset + 4], q_frame2[src_offset:src_offset + 4])
        new_quat_frame[dest_offset:dest_offset+4] = new_q
        dest_offset += 4
        src_offset += 4
    return new_quat_frame


def get_3d_rotation_between_vectors(a, b):
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s ==0:
        return np.eye(3)
    c = np.dot(a,b)
    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    v_x_2 = np.dot(v_x,v_x)
    r = np.eye(3) + v_x + (v_x_2* (1-c/s**2))
    return r

def quaternion_from_vector_to_vector(a, b):
    "src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another"
    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    return q/ np.linalg.norm(q)


def convert_euler_to_quat(euler_frame, joints):
    quat_frame = euler_frame[:3].tolist()
    offset = 3
    step = 3
    for joint in joints:
        e = euler_frame[offset:offset+step]
        q = euler_to_quaternion(e)
        quat_frame += list(q)
        offset += step
    return np.array(quat_frame)


def normalize_quaternion(q):
    return quaternion_inverse(q) / np.dot(q, q)
