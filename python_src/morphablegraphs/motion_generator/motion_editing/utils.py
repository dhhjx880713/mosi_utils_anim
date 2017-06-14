import numpy as np
import math
from ...external.transformations import quaternion_multiply, quaternion_inverse, quaternion_matrix, quaternion_from_matrix
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


def get_average_joint_position(skeleton, frames, joint_name, start_frame, end_frame):
    temp_positions = []
    for idx in xrange(start_frame, end_frame):
        frame = frames[idx]
        pos = skeleton.nodes[joint_name].get_global_position(frame)
        temp_positions.append(pos)
    return np.mean(temp_positions, axis=0)


def get_average_joint_direction(skeleton, frames, joint_name, child_joint_name, start_frame, end_frame,ground_height=0):
    temp_dirs = []
    for idx in xrange(start_frame, end_frame):
        frame = frames[idx]
        pos1 = skeleton.nodes[joint_name].get_global_position(frame)
        pos2 = skeleton.nodes[child_joint_name].get_global_position(frame)
        pos2[1] = ground_height
        joint_dir = pos2 - pos1
        joint_dir /= np.linalg.norm(joint_dir)
        temp_dirs.append(joint_dir)
    return np.mean(temp_dirs, axis=0)


def to_local_cos(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = skeleton.nodes[node_name].get_global_matrix(frame)[:3,:3]
    #pm[:3, 3] = [0, 0, 0]
    inv_pm = np.linalg.inv(pm)
    r = quaternion_matrix(q)[:3,:3]
    lr = np.dot(inv_pm, r)[:3,:3]
    q = quaternion_from_matrix(lr)
    return q

def get_dir_on_plane(x, n):
    axb = np.cross(x,n)
    d = np.cross(n, normalize(axb))
    d = normalize(d)
    return d

def project2(x,n):
    """ get direction on plane based on cross product and then project onto the direction """
    d = get_dir_on_plane(x, n)
    return project_on_line(x, d)

def project_vec3(x, n):
    """" project vector on normal of plane and then substract from vector to get projection on plane """
    w = project_on_line(x, n)
    v = x-w
    return v

def project(x, n):
    """ http://www.euclideanspace.com/maths/geometry/elements/plane/lineOnPlane/"""
    l = np.linalg.norm(x)
    a = normalize(x)
    b = normalize(n)
    axb = np.cross(a,b)
    bxaxb = np.cross(b, axb)
    return l * bxaxb

def project_on_line(x, v):
    """https://en.wikipedia.org/wiki/Scalar_projection"""
    s = np.dot(x, v) / np.dot(v, v)
    return s * v

def project_onto_plane(x, n):
    """https://stackoverflow.com/questions/17915475/how-may-i-project-vectors-onto-a-plane-defined-by-its-orthogonal-vector-in-pytho"""
    nl = np.linalg.norm(n)
    d = np.dot(x, n) / nl
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]


def project_vec_on_plane(vec, n):
    """https://math.stackexchange.com/questions/633181/formula-to-project-a-vector-onto-a-plane"""
    n = normalize(n)
    d = np.dot(vec, n)
    return vec - np.dot(d, n)


def distance_from_point_to_line(p1, p2, vec):
    proj = p2+project_on_line(p1, vec)
    return np.linalg.norm(proj - p1)


def limb_projection(p1, center, n):
    #s1 = np.dot(p1, n) / np.dot(p1, p1)
    #proj_p1 = p1 - s1*n

    #s2 = np.dot(p2, n) / np.dot(p2, p2)
    #proj_p2 = p2 - s2 * n
    proj_p1 = project_vec3(p1, n)
    proj_center = project_vec3(center, n)

    d = np.linalg.norm(proj_p1-proj_center)
    #print proj_p1, proj_p2, s1, s2, proj_p1, proj_p2, d, n
    #print "d", d
    return d
