# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 20:11:35 2015

@author: Erik, Han, Markus
"""
import os
import numpy as np
from math import sqrt, radians, sin, cos, isnan

from copy import deepcopy
import glob
from .bvh import BVHReader, BVHWriter
from scipy import stats  # linear regression
from .quaternion_frame import QuaternionFrame
from ..external.transformations import quaternion_matrix, euler_from_matrix, \
    quaternion_from_matrix, euler_matrix, \
    quaternion_multiply, \
    quaternion_about_axis, \
    rotation_matrix, \
    quaternion_conjugate


from .constants import *


def rotation_order_to_string(rotation_order):
    r_order_string = "r"
    for c in rotation_order:
        if c == "Xrotation":
            r_order_string += "x"
        elif c == "Yrotation":
            r_order_string += "y"
        elif c == "Zrotation":
            r_order_string += "z"
    return r_order_string


def extract_root_positions_from_frames(frames):
    roots = []
    for i in range(len(frames)):
        position = np.array([frames[i][0], frames[i][1], frames[i][2]])
        roots.append(position)
    return np.array(roots)


def get_arc_length_from_points(points):
    """
    Note: accuracy depends on the granulariy of points
    """
    points = np.asarray(points)
    arc_length = 0.0
    last_p = None
    for p in points:
        if last_p is not None:
            delta = p - last_p
            arc_length += np.linalg.norm(delta)
        last_p = p
    return arc_length


def get_step_length(frames):
    """

    :param frames: a list of euler or quaternion frames
    :return step_len: travelled distance of root joint
    """
    root_points = extract_root_positions(frames)
    step_len = get_arc_length_from_points(root_points)
    return step_len


def inverse_pose_transform(translation, euler_angles):
    inv_translation = -translation[0],-translation[1],-translation[2]
    inv_euler_angles = -euler_angles[0],-euler_angles[1],-euler_angles[2]
    return inv_translation, inv_euler_angles


def euler_to_quaternion(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER, filter_value=True):
    """Convert euler angles to quaternion vector [qw, qx, qy, qz]

    Parameters
    ----------
    * euler_angles: list of floats
    \tA list of ordered euler angles in degrees
    * rotation_order: Iteratable
    \t a list that specifies the rotation axis corresponding to the values in euler_angles
    * filter_values: Bool
    \t enforce a unique rotation representation

    """
    # convert euler angle from degree into radians
    return euler_to_quaternion_rad(np.deg2rad(euler_angles), rotation_order, filter_value)


def euler_to_quaternion_rad(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER, filter_value=True):
    """Convert euler angles to quaternion vector [qw, qx, qy, qz]

    Parameters
    ----------
    * euler_angles: list of floats
    \tA list of ordered euler angles in radian
    * rotation_order: Iteratable
    \t a list that specifies the rotation axis corresponding to the values in euler_angles
    * filter_values: Bool
    \t enforce a unique rotation representation

    """
    # convert euler angles into rotation matrix, then convert rotation matrix
    # into quaternion
    assert len(euler_angles) == 3, ('The length of euler angles should be 3!')

    rotmat = euler_to_matrix(euler_angles, rotation_order)
    q = quaternion_from_matrix(rotmat)
    # filter the quaternion http://physicsforgames.blogspot.de/2010/02/quaternions.html
    if filter_value:
        dot = np.sum(q)
        if dot < 0:
            q = -q
    return q[0], q[1], q[2], q[3]

def matrix_3x3_to_4x4(m):
    """
    Transforms a list of matrices from 3x3 rotational matrix to 4x4 homogeneous transformation matrix
    :param m: np.array(n, 3, 3)
    :return: np.array(n, 4, 4)
    """
    shape = list(m.shape)
    shape[-1] = 4
    shape[-2] = 4
    s = np.zeros(shape)
    s[:, :3,:3] = m
    s[:, 3,3] = 1
    return s


def quaternion_to_euler_smooth(q, euler_ref, rotation_order=DEFAULT_ROTATION_ORDER):
    """

    :param q: list of floats, [qw, qx, qy, qz]
    :param euler_ref: reference euler angle in degree, pick the euler value which is closer to reference rotation
    :param rotation_order: list of str
    :return: euler angles in degree
    """
    len_angle = len(euler_ref)
    euler_angles = quaternion_to_euler(q, rotation_order)
    for i in range(len_angle):
        if euler_ref[i] < 0:
            diff1 = abs(euler_ref[i] - euler_angles[i])
            diff2 = abs(euler_angles[i] - 360 - euler_ref[i])
            if diff2 < diff1:
                euler_angles[i] -= 360
        else:
            diff1 = abs(euler_ref[i] - euler_angles[i])
            diff2 = abs(euler_angles[i] + 360 -euler_ref[i])
            if diff2 < diff1:
                euler_angles[i] += 360
    return euler_angles

def quaternion_to_euler_rad(q, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    Parameters
    ----------
    * q: list of floats
    \tQuaternion vector with form: [qw, qx, qy, qz]

    Return
    ------
    * euler_angles: list
    \tEuler angles in radian with specified order
    """
    q = np.asarray(q)
    # normalize quaternion vector
    q /= np.linalg.norm(q)
    # create rotation matrix from quaternion
    Rq = quaternion_matrix(q)
    # map rotation order
    r_order_string = rotation_order_to_string(rotation_order)
    return euler_from_matrix(Rq, r_order_string)


def quaternion_to_euler(q, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    Parameters
    ----------
    * q: list of floats
    \tQuaternion vector with form: [qw, qx, qy, qz]

    Return
    ------
    * euler_angles: list
    \tEuler angles in degree with specified order
    """
    return np.rad2deg(quaternion_to_euler_rad(q, rotation_order)).tolist()


def convert_euler_frames_to_quaternion_frames_legacy(bvhreader, euler_frames, filter_joints=True, animated_joints=None):
    """
    :param bvhreader: a BVHReader instance to store skeleton information
    :param euler_frames: a list of euler frames
    :return: a list of quaternion frames
    """
    quat_frames = []
    last_quat_frame = None
    for frame in euler_frames:
        quat_frame = QuaternionFrame(bvhreader, frame, ignore_bip_joints=filter_joints, animated_joints=animated_joints)
        if last_quat_frame is not None:
            for joint_name in list(bvhreader.node_names.keys()):
                if joint_name in list(quat_frame.keys()):
                    quat_frame[joint_name] = check_quat(
                        quat_frame[joint_name],
                        last_quat_frame[joint_name])
        last_quat_frame = quat_frame
        root_translation = frame[0:3]
        quat_frame_values = [root_translation, ] + list(quat_frame.values())
        quat_frame_values = convert_quat_frame_value_to_array(quat_frame_values)
        quat_frames.append(quat_frame_values)
    return quat_frames


def euler_to_matrix(euler_angles, rotation_order):
    # convert euler angles into rotation matrix
    assert len(euler_angles) == 3, ('The length of euler angles should be 3!')
    rotmat = None
    if rotation_order[0] == 'Xrotation':
        if rotation_order[1] == 'Yrotation':
            rotmat = euler_matrix(euler_angles[0],
                                  euler_angles[1],
                                  euler_angles[2],
                                  axes='rxyz')
        elif rotation_order[1] == 'Zrotation':
            rotmat = euler_matrix(euler_angles[0],
                                  euler_angles[1],
                                  euler_angles[2],
                                  axes='rxzy')
    elif rotation_order[0] == 'Yrotation':
        if rotation_order[1] == 'Xrotation':
            rotmat = euler_matrix(euler_angles[0],
                                  euler_angles[1],
                                  euler_angles[2],
                                  axes='ryxz')
        elif rotation_order[1] == 'Zrotation':
            rotmat = euler_matrix(euler_angles[0],
                                  euler_angles[1],
                                  euler_angles[2],
                                  axes='ryzx')
    elif rotation_order[0] == 'Zrotation':
        if rotation_order[1] == 'Xrotation':
            rotmat = euler_matrix(euler_angles[0],
                                  euler_angles[1],
                                  euler_angles[2],
                                  axes='rzxy')
        elif rotation_order[1] == 'Yrotation':
            rotmat = euler_matrix(euler_angles[0],
                                  euler_angles[1],
                                  euler_angles[2],
                                  axes='rzyx')
    else:
        raise ValueError('Unknown rotation order')
    return rotmat


def extract_euler_angles(bvh_reader, euler_frame, node_name):
    channels = bvh_reader.node_names[node_name]["channels"]
    euler_angles = []
    rotation_order = []
    for ch in channels:
        if ch.lower().endswith("rotation"):
            idx = bvh_reader.node_channels.index((node_name, ch))
            rotation_order.append(ch)
            euler_angles.append(euler_frame[idx])
    return euler_angles, rotation_order


def convert_euler_frame_to_quat_frame(bvh_reader, euler_frame, prev_quat_frame, animated_joints=None):
    if animated_joints is None:
        animated_joints = list(bvh_reader.get_animated_joints())
    n_dims = len(animated_joints)* 4 +3
    quat_frame = np.zeros(n_dims)
    quat_frame[:3] = euler_frame[:3]
    dst = 3
    for node_name in animated_joints:
        euler_angles, rotation_order = extract_euler_angles(bvh_reader, euler_frame, node_name)
        q = euler_to_quaternion(euler_angles, rotation_order, True)
        if prev_quat_frame is not None:
            q = check_quat(q, prev_quat_frame[dst:dst + 4])
        quat_frame[dst:dst + 4] = q
        dst+=4
    return quat_frame


def convert_euler_frames_to_quaternion_frames(bvh_reader, euler_frames, animated_joints=None):
    """
    :param bvh_reader: a BVHReader instance to store skeleton information
    :param euler_frames: a list of euler frames
    :return: a list of quaternion frames
    """
    quat_frames = []
    prev_quat_frame = None
    for frame in euler_frames:
        quat_frame = convert_euler_frame_to_quat_frame(bvh_reader, frame, prev_quat_frame, animated_joints)
        quat_frames.append(quat_frame)
        prev_quat_frame = quat_frame
    return quat_frames


def convert_quat_frame_value_to_array(quat_frame_values):
    n_channels = len(quat_frame_values)
    quat_frame_value_array = []
    for item in quat_frame_values:
        if not isinstance(item, list):
            item = list(item)
        quat_frame_value_array += item
    return quat_frame_value_array


def check_quat(test_quat, ref_quat):
    """check if quat needs to be flipped or not
    """
    test_quat = np.asarray(test_quat)
    ref_quat = np.asarray(ref_quat)
    dot = np.dot(test_quat, ref_quat)
    if dot < 0:
        test_quat = - test_quat
    return test_quat.tolist()


def convert_quaternion_frames_to_euler_frames(quaternion_frames):
    """Returns an nparray of Euler frames

    Parameters
    ----------

    :param quaternion_frames:
     * quaternion_frames: List of quaternion frames
    \tQuaternion frames that shall be converted to Euler frames

    Returns
    -------

    * euler_frames: numpy array
    \tEuler frames
    """

    def gen_4_tuples(it):
        """Generator of n-tuples from iterable"""

        return list(zip(it[0::4], it[1::4], it[2::4], it[3::4]))

    def get_euler_frame(quaternionion_frame):
        """Converts a quaternion frame into an Euler frame"""

        euler_frame = list(quaternionion_frame[:3])
        for quaternion in gen_4_tuples(quaternionion_frame[3:]):
            euler_frame += quaternion_to_euler(quaternion)

        return euler_frame

    euler_frames = list(map(get_euler_frame, quaternion_frames))

    return np.array(euler_frames)


def convert_quaternion_to_euler(quaternion_frames):
    """Returns an nparray of Euler frames

    Parameters
    ----------

     * quaternion_frames: List of quaternion frames
    \tQuaternion frames that shall be converted to Euler frames

    Returns
    -------

    * euler_frames: numpy array
    \tEuler frames
    """

    def gen_4_tuples(it):
        """Generator of n-tuples from iterable"""

        return list(zip(it[0::4], it[1::4], it[2::4], it[3::4]))

    def get_euler_frame(quaternionion_frame):
        """Converts a quaternion frame into an Euler frame"""

        euler_frame = list(quaternionion_frame[:3])
        for quaternion in gen_4_tuples(quaternionion_frame[3:]):
            euler_frame += quaternion_to_euler(quaternion)

        return euler_frame

    euler_frames = list(map(get_euler_frame, quaternion_frames))

    return np.array(euler_frames)


def euler_substraction(theta1, theta2):
    ''' compute the angular distance from theta2 to theta1, positive value is anti-clockwise, negative is clockwise
    @param theta1, theta2: angles in degree
    '''
    theta1 %= 360
    theta2 %= 360
    if theta1 > 180:
        theta1 -= 360
    elif theta1 < -180:
        theta1 += 360

    if theta2 > 180:
        theta2 -= 360
    elif theta2 < - 180:
        theta2 += 360

    theta = theta1 - theta2
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    if theta > 180 or theta < - 180:
        raise ValueError(' exception value')
    return theta


def get_cartesian_coordinates_from_quaternion(skeleton, node_name, quaternion_frame, return_gloabl_matrix=False):
    """Returns cartesian coordinates for one node at one frame. Modified to
     handle frames with omitted values for joints starting with "Bip"

    Parameters
    ----------
    * node_name: String
    \tName of node
     * skeleton: Skeleton
    \tBVH data structure read from a file

    """
    if skeleton.node_names[node_name]["level"] == 0:
        root_frame_position = quaternion_frame[:3]
        root_node_offset = skeleton.node_names[node_name]["offset"]

        return [t + o for t, o in
                zip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(skeleton.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list

        offsets = [skeleton.node_names[nodename]["offset"]
                   for nodename in chain_names]
        root_position = quaternion_frame[:3].flatten()
        offsets[0] = [r + o for r, o in zip(root_position, offsets[0])]

        j_matrices = []
        count = 0
        for node_name in chain_names:
            index = skeleton.node_name_frame_map[node_name] * 4 + 3
            j_matrix = quaternion_matrix(quaternion_frame[index: index + 4])
            j_matrix[:, 3] = offsets[count] + [1]
            j_matrices.append(j_matrix)
            count += 1

        global_matrix = np.identity(4)
        for j_matrix in j_matrices:
            global_matrix = np.dot(global_matrix, j_matrix)
        if return_gloabl_matrix:
            return global_matrix
        else:
            point = np.array([0, 0, 0, 1])
            point = np.dot(global_matrix, point)
            return point[:3].tolist()


def get_cartesian_coordinates_for_plam_quaternion(
        skeleton,
        quat_frame,
        node_name='Left'):
    if node_name == 'Left':
        node_name = 'LeftHand'
        Finger2 = 'Bip01_L_Finger2'
        Finger21 = 'Bip01_L_Finger21'
        Finger2_offset = [9.55928, -0.145352, -0.186424]
        Finger2_angles = [-0.0, 0.0, 0.0]
        Finger21_offset = [3.801407, 0.0, 0.0]
        Finger21_angles = [-0.0, 0.0, 0.0]
    elif node_name == 'Right':
        node_name = 'RightHand'
        Finger2 = 'Bip01_R_Finger2'
        Finger21 = 'Bip01_R_Finger21'
        Finger2_offset = [9.559288, 0.145353, -0.186417]
        Finger2_angles = [-0.0, 0.0, 0.0]
        Finger21_offset = [3.801407, 0.0, 0.0]
        Finger21_angles = [-0.0, 0.0, 0.0]
    else:
        raise ValueError('Unknown node name!')
    global_matrix = get_cartesian_coordinates_from_quaternion(
        skeleton,
        node_name,
        quat_frame,
        return_gloabl_matrix=True)
    quat_finger2 = euler_to_quaternion(Finger2_angles)
    transmat_finger2 = quaternion_matrix(quat_finger2)
    transmat_finger2[:3, 3] = Finger2_offset[:]
    global_matrix = np.dot(global_matrix, transmat_finger2)
    quat_finger21 = euler_to_quaternion(Finger21_angles)
    transmat_finger21 = quaternion_matrix(quat_finger21)
    transmat_finger21[:3, 3] = Finger21_offset[:]
    global_matrix = np.dot(global_matrix, transmat_finger21)
    return global_matrix[:3, 3]


def convert_quaternion_frame_to_cartesian_frame(skeleton, quat_frame):
    """
    Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
    """
    cartesian_frame = []
    for node_name in skeleton.node_names:
        # ignore Bip joints and end sites
        if not node_name.startswith(
                "Bip") and "children" in list(skeleton.node_names[node_name].keys()):
            cartesian_frame.append(
                get_cartesian_coordinates_from_quaternion(
                    skeleton,
                    node_name,
                    quat_frame))  # get_cartesian_coordinates2

    return cartesian_frame


def align_point_clouds_2D(a, b, weights):
    '''
     Finds aligning 2d transformation of two equally sized point clouds.
     from Motion Graphs paper by Kovar et al.
     Parameters
     ---------
     *a: list
     \t point cloud
     *b: list
     \t point cloud
     *weights: list
     \t weights of correspondences
     Returns
     -------
     *theta: float
     \t angle around y axis in radians
     *offset_x: float
     \t
     *offset_z: float

     '''
    if len(a) != len(b):
        raise ValueError("two point cloud should have the same number points: "+str(len(a))+","+str(len(b)))
    n_points = len(a)
    numerator_left = 0
    denominator_left = 0
    weighted_sum_a_x = 0
    weighted_sum_b_x = 0
    weighted_sum_a_z = 0
    weighted_sum_b_z = 0
    sum_of_weights = 0.0
    #    if not weights:
    #        weight = 1.0/n_points # todo set weight base on joint level
    for index in range(n_points):
        numerator_left += weights[index] * (a[index][0] * b[index][2] -
                                            b[index][0] * a[index][2])
        denominator_left += weights[index] * (a[index][0] * b[index][0] +
                                              a[index][2] * b[index][2])
        sum_of_weights += weights[index]
        weighted_sum_a_x += weights[index] * a[index][0]
        weighted_sum_b_x += weights[index] * b[index][0]
        weighted_sum_a_z += weights[index] * a[index][2]
        weighted_sum_b_z += weights[index] * b[index][2]
    numerator_right = 1.0 / sum_of_weights * \
        (weighted_sum_a_x * weighted_sum_b_z -
         weighted_sum_b_x * weighted_sum_a_z)
    numerator = numerator_left - numerator_right
    denominator_right = 1.0 / sum_of_weights * \
        (weighted_sum_a_x * weighted_sum_b_x +
         weighted_sum_a_z * weighted_sum_b_z)
    denominator = denominator_left - denominator_right
    theta = np.arctan2(numerator, denominator)
    offset_x = (weighted_sum_a_x - weighted_sum_b_x *
                np.cos(theta) - weighted_sum_b_z * np.sin(theta)) / sum_of_weights
    offset_z = (weighted_sum_a_z + weighted_sum_b_x *
                np.sin(theta) - weighted_sum_b_z * np.cos(theta)) / sum_of_weights

    return theta, offset_x, offset_z


def rotate_around_y_axis(point, theta):
    """
    source https://www.siggraph.org/education/materials/HyperGraph/modeling/mod_tran/2drota.htm
    Parameters
    ---------
    *point: list
    \t coordinates
    *theta: float
    \t angle in radians
    """
    transfomed_point = point[:]
    transfomed_point[0] = point[0] * cos(theta) - point[2] * sin(theta)
    transfomed_point[2] = point[2] * cos(theta) + point[0] * sin(theta)
    return transfomed_point


def transform_point_by_quaternion(point, quaternion, offset, origin=None):
    """
    http://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    homogenous_point = np.append([0], point)
    tmp_q = quaternion_multiply(quaternion, homogenous_point)
    temp_q = quaternion_multiply(tmp_q, quaternion_conjugate(quaternion))
    new_point = [temp_q[i+1] + offset[i] + origin[i] for i in range(3)]
    return new_point


def transform_point_by_quaternion_faster(point, quaternion, offset, origin=None):
    """
    http://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    t = 2 * np.cross(quaternion[1:], point)
    new_point = np.array(point) + quaternion[0] * t + np.cross(quaternion[1:], t)
    new_point = [new_point[i] + offset[i] + origin[i] for i in range(3)]
    return new_point


def transform_point_by_quaternion_faster2(point, quaternion, offset, origin=None):
    """
    http://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    u = quaternion[1:]
    s = quaternion[0]
    new_point = 2.0 * np.dot(u, point) * u + (s*s - np.dot(u, u)) * point + 2.0 * s * np.cross(u, point)
    new_point = [new_point[i] + offset[i] + origin[i] for i in range(3)]
    return new_point


def euler_angles_to_rotation_matrix(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER):
        # generate rotation matrix based on rotation order
    assert len(euler_angles) == 3, ('The length of rotation angles should be 3')
    if round(euler_angles[0], 3) == 0 and round(euler_angles[2], 3) == 0:
        # rotation about y axis
        R = rotation_matrix(np.deg2rad(euler_angles[1]), [0, 1, 0])
    else:
        euler_angles = np.deg2rad(euler_angles)
        R = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes=rotation_order_to_string(rotation_order))
    return R


def create_transformation_matrix(translation, euler_angles, rotation_order=DEFAULT_ROTATION_ORDER):
    #m = np.eye(4,4)
    m = euler_angles_to_rotation_matrix(euler_angles, rotation_order)
    m[:3,3] = translation
    return m


def transform_point(point, euler_angles, offset, origin=None, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    rotate point around y axis and translate it by an offset
    Parameters
    ---------
    *point: list
    \t coordinates
    *angles: list of floats
    \tRotation angles in degrees
    *offset: list of floats
    \tTranslation
    """
    assert len(point) == 3, ('the point should be a list of length 3')
    # translate point to original point
    point = np.asarray(point)
    if origin is not None:
        origin = np.asarray(origin)
        point = point - origin
    if not isinstance(point, list):
        point = list(point)
    point.append(1.0)
    R = euler_angles_to_rotation_matrix(euler_angles, rotation_order)
    rotated_point = np.dot(R, point)
    if origin is not None:
        rotated_point[:3] += origin
    #print rotated_point,
    transformed_point = rotated_point[:3] + offset
    return transformed_point.tolist()


def transform_euler_frame(euler_frame, angles, offset, rotation_order=None):
    """
    Calls transform_point for the root parameters and adds theta to the y rotation
    channel of the frame.

    The offset of root is transformed by transform_point
    The orientation of root is rotated by Rotation matrix

    Parameters
    ---------
    *euler_frame: np.ndarray
    \t the parameters of a single frame
    *angles: list of floats
    \tRotation angles in degrees
    *offset: np.ndarray
    \tTranslation
    """
    if rotation_order is None:
        rotation_order = ["Xrotation", "Yrotation", "Zrotation"]
    transformed_frame = euler_frame[:]
    transformed_frame[:3] = transform_point(euler_frame[:3], angles, offset)
    if rotation_order[0] == 'Xrotation':
        if rotation_order[1] == 'Yrotation':
            R = euler_matrix(np.deg2rad(angles[0]),
                             np.deg2rad(angles[1]),
                             np.deg2rad(angles[2]),
                             axes='rxyz')
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='rxyz')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'rxyz'))
        elif rotation_order[1] == 'Zrotation':
            R = euler_matrix(np.deg2rad(angles[0]),
                             np.deg2rad(angles[1]),
                             np.deg2rad(angles[2]),
                             axes='rxzy')
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='rxzy')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'rxzy'))
    elif rotation_order[0] == 'Yrotation':
        if rotation_order[1] == 'Xrotation':
            R = euler_matrix(np.deg2rad(angles[0]),
                             np.deg2rad(angles[1]),
                             np.deg2rad(angles[2]),
                             axes='ryxz')
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='ryxz')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'ryxz'))
        elif rotation_order[1] == 'Zrotation':
            R = euler_matrix(np.deg2rad(angles[0]),
                             np.deg2rad(angles[1]),
                             np.deg2rad(angles[2]),
                             axes='ryzx')
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='ryzx')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'ryzx'))
    elif rotation_order[0] == 'Zrotation':
        if rotation_order[1] == 'Xrotation':
            R = euler_matrix(np.deg2rad(angles[0]),
                             np.deg2rad(angles[1]),
                             np.deg2rad(angles[2]),
                             axes='rzxy')
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='rzxy')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'rzxy'))
        elif rotation_order[1] == 'Yrotation':
            R = euler_matrix(np.deg2rad(angles[0]),
                             np.deg2rad(angles[1]),
                             np.deg2rad(angles[2]),
                             axes='rzyx')
            OR = euler_matrix(np.deg2rad(euler_frame[3]),
                              np.deg2rad(euler_frame[4]),
                              np.deg2rad(euler_frame[5]),
                              axes='rzyx')
            rotmat = np.dot(R, OR)
            eul_angles = np.rad2deg(euler_from_matrix(rotmat, 'rzyx'))
    transformed_frame[3:6] = eul_angles
    return transformed_frame


def transform_euler_frames(euler_frames, angles, offset):
    """ Applies a transformation on the root joint of a list euler frames.
    Parameters
    ----------
    *euler_frames: np.ndarray
    \tList of frames where the rotation is represented as euler angles in degrees.
    *angles: list of floats
    \tRotation angles in degrees
    *offset:  np.ndarray
    \tTranslation
    """
    transformed_euler_frames = []
    for frame in euler_frames:
        transformed_euler_frames.append(
            transform_euler_frame(frame, angles, offset))
    return np.array(transformed_euler_frames)

'''
def transform_quaternion_frame(quat_frame,
                               angles,
                               offset,
                               origin=None,
                               rotation_order=None):
    """
    Calls transform_point for the root parameters and adds theta to the y
    rotation channel of the frame.
    Parameters
    ---------
    *quat_frame: np.ndarray
    \t the parameters of a single frame
    *angles: list of floats
    \tRotation angles in degrees
    *offset: np.ndarray
    \tTranslation
    """

    if rotation_order is None:
        rotation_order = ["Xrotation", "Yrotation", "Zrotation"]
    transformed_frame = quat_frame[:]
    #    original_point_copy = deepcopy(original_point)
    #    transformed_frame[:3] = transform_point(quat_frame[:3], [0, 0, 0], offset)
    # q = euler_to_quaternion(angles, rotation_order) # replace
    if round(angles[0], 3) == 0 and round(angles[2], 3) == 0:
        q = quaternion_about_axis(np.deg2rad(angles[1]), [0, 1, 0])
    else:
        q = euler_to_quaternion(angles, rotation_order)

    transformed_frame[:3] = transform_point(quat_frame[:3], angles, offset, origin)
    #transformed_frame[:3] = transform_point_by_quaternion_faster2(quat_frame[:3], q, offset, origin)
    oq = quat_frame[3:7]
    rotated_q = quaternion_multiply(q, oq)
    transformed_frame[3:7] = rotated_q
    return transformed_frame
'''

def transform_quaternion_frames(quat_frames, angles, offset, rotation_order=None):
    """ Applies a transformation on the root joint of a list quaternion frames.
    Parameters
    ----------
    *quat_frames: np.ndarray
    \tList of frames where the rotation is represented as euler angles in degrees.
    *angles: list of floats
    \tRotation angles in degrees
    *offset:  np.ndarray
    \tTranslation
    """
    if rotation_order is None:
        rotation_order = DEFAULT_ROTATION_ORDER
    offset = np.array(offset)
    if round(angles[0], 3) == 0 and round(angles[2], 3) == 0:
        rotation_q = quaternion_about_axis(np.deg2rad(angles[1]), [0, 1, 0])
    else:
        rotation_q = euler_to_quaternion(angles, rotation_order)
    rotation_matrix = euler_angles_to_rotation_matrix(angles, rotation_order)[:3, :3]
    for frame in quat_frames:
        ot = frame[:3]
        oq = frame[3:7]
        frame[:3] = np.dot(rotation_matrix, ot) + offset
        frame[3:7] = quaternion_multiply(rotation_q, oq)
    return quat_frames


def smooth_quaternion_frames(quaternion_frames, discontinuity, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    quaternion_frames: list
    \tA list of quaternion frames
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """
    n_joints = (len(quaternion_frames[0]) - 3) / 4
    # smooth quaternion
    n_frames = len(quaternion_frames)
    for i in range(n_joints):
        for j in range(n_frames - 1):
            q1 = np.array(quaternion_frames[j][3 + i * 4: 3 + (i + 1) * 4])
            q2 = np.array(quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
            if np.dot(q1, q2) < 0:
                quaternion_frames[
                    j + 1][3 + i * 4:3 + (i + 1) * 4] = -quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    d = float(discontinuity)
    w = float(window)
    smoothing_factors = []
    for f in range(n_frames):
        value = 0.0
        if d - w <= f < d:
            tmp = (f - d + w) / w
            value = 0.5 * tmp ** 2
        elif d <= f <= d + w:
            tmp = (f - d + w) / w
            value = -0.5 * tmp ** 2 + 2 * tmp - 2
        smoothing_factors.append(value)
    smoothing_factors = np.array(smoothing_factors)
    new_quaternion_frames = []
    for i in range(len(quaternion_frames[0])):
        current_value = quaternion_frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        new_value = current_value + (magnitude * smoothing_factors)
        new_quaternion_frames.append(new_value)
    new_quaternion_frames = np.array(new_quaternion_frames).T
    return new_quaternion_frames


def smooth_quaternion_frames_partially(quaternion_frames, joint_parameter_indices, discontinuity, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    quaternion_frames: list
    \tA list of quaternion frames
    parameters_indices: list
    \tThe list of joint parameter indices that should be smoothed
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """
    n_joints = (len(quaternion_frames[0]) - 3) / 4
    # smooth quaternion
    n_frames = len(quaternion_frames)
    for i in range(n_joints):
         for j in range(n_frames - 1):
             q1 = np.array(quaternion_frames[j][3 + i * 4: 3 + (i + 1) * 4])
             q2 = np.array(quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
             if np.dot(q1, q2) < 0:
                quaternion_frames[
                    j + 1][3 + i * 4:3 + (i + 1) * 4] = -quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    transition_frame = float(discontinuity)
    window_size = float(window)
    smoothing_factors = []
    for frame_number in range(n_frames):
        value = 0.0
        if transition_frame - window_size <= frame_number < transition_frame:
            tmp = (frame_number - transition_frame + window_size) / window_size
            value = 0.5 * tmp ** 2
        elif transition_frame <= frame_number <= transition_frame + window_size:
            tmp = (frame_number - transition_frame + window_size) / window_size
            value = -0.5 * tmp ** 2 + 2 * tmp - 2
        smoothing_factors.append(value)
    #smoothing_factors = np.array(smoothing_factors)
    transition_frame = int(transition_frame)
    #new_quaternion_frames = []
    magnitude_vector = np.array(quaternion_frames[transition_frame]) - quaternion_frames[transition_frame-1]
    #print "magnitude ", magnitude_vector
    for i in joint_parameter_indices:
        joint_parameter_index = i*4+3
        for frame_index in range(n_frames):
            quaternion_frames[frame_index][joint_parameter_index] = quaternion_frames[frame_index][joint_parameter_index]+magnitude_vector[joint_parameter_index] * smoothing_factors[frame_index]
    #
    # for i in xrange(len(quaternion_frames[0])):
    #     joint_parameter_value = quaternion_frames[:, i]
    #     if i in joint_parameter_indices:
    #
    #         magnitude = joint_parameter_value[transition_frame] - joint_parameter_value[transition_frame - 1]
    #         print "smooth", i, magnitude, smoothing_factors
    #         new_value = joint_parameter_value + (magnitude * smoothing_factors)
    #         new_quaternion_frames.append(new_value)
    #     else:
    #         new_quaternion_frames.append(joint_parameter_value)
    # new_quaternion_frames = np.array(new_quaternion_frames).T
    #return quaternion_frames


def smooth_motion(euler_frames, discontinuity, window=20):
    """ Smooth a function around the given discontinuity frame

    Parameters
    ----------
    motion : AnimationController
        The motion to be smoothed
        ATTENTION: This function changes the values of the motion
    discontinuity : int
        The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
        The smoothing window
    Returns
    -------
    None.
    """
    d = float(discontinuity)
    s = float(window)

    smoothing_faktors = []
    for f in range(len(euler_frames)):
        value = 0.0
        if d - s <= f < d:
            tmp = ((f - d + s) / s)
            value = 0.5 * tmp ** 2
        elif d <= f <= d + s:
            tmp = ((f - d + s) / s)
            value = -0.5 * tmp ** 2 + 2 * tmp - 2

        smoothing_faktors.append(value)

    smoothing_faktors = np.array(smoothing_faktors)
    new_euler_frames = []
    for i in range(len(euler_frames[0])):
        current_value = euler_frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        if magnitude > 180:
            magnitude -= 360
        elif magnitude < -180:
            magnitude += 360
        new_value = current_value + (magnitude * smoothing_faktors)
        new_euler_frames.append(new_value)
    new_euler_frames = np.array(new_euler_frames).T
    return new_euler_frames


def smoothly_concatenate(euler_frames_a, euler_frames_b, window_size=20):
    euler_frames = np.concatenate((euler_frames_a, euler_frames_b), axis=0)
    euler_frames = smooth_motion(euler_frames, len(euler_frames_a), window_size)
    return euler_frames


def smoothly_concatenate_quaternion_frames(quaternion_frames_a, quaternion_frames_b, window_size=20):

    quaternion_frames = np.concatenate((quaternion_frames_a, quaternion_frames_b), axis=0)
    d = len(quaternion_frames_a)
    quaternion_frames = smooth_quaternion_frames(quaternion_frames, d, window_size)
    return quaternion_frames



def get_2d_pose_transform(quaternion_frames, frame_number, param_range=(3,7)):
    dir_vec = pose_orientation_quat(quaternion_frames[frame_number], param_range)
    angle = get_rotation_angle(dir_vec, [0, -1])
    offset_x = quaternion_frames[frame_number][0]
    offset_z = quaternion_frames[frame_number][2]
    return [offset_x, 0.0, offset_z], [0,angle,0]


def fast_quat_frames_transformation(quaternion_frames_a,
                                    quaternion_frames_b, param_range=(3,7)):
    #param_range = (7,11)
    dir_vec_a = pose_orientation_quat(quaternion_frames_a[-1],param_range)
    dir_vec_b = pose_orientation_quat(quaternion_frames_b[0],param_range)
    #print "dir_vector a", dir_vec_a

    #print "dir_vector b", dir_vec_b

    #find_aligning_transformation
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = quaternion_frames_a[-1][0] - quaternion_frames_b[0][0]
    offset_z = quaternion_frames_a[-1][2] - quaternion_frames_b[0][2]
    offset = [offset_x, 0.0, offset_z]
    return angle, offset



def fast_euler_frames_transformation(euler_frames_a,
                                     euler_frames_b):
    dir_vec_a = pose_orientation_euler(euler_frames_a[-1])
    dir_vec_b = pose_orientation_euler(euler_frames_b[0])
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = euler_frames_a[-1][0] - euler_frames_b[0][0]
    offset_z = euler_frames_a[-1][2] - euler_frames_b[0][2]
    offset = [offset_x, 0.0, offset_z]
    return angle, offset


def get_rotation_angle(point1, point2):
    """
    estimate the rotation angle from point2 to point1
    point1, point2 are normalized points
    rotate point2 to be the same as point1

    Parameters
    ----------
    *point1, point2: list or numpy array
    \tUnit 2d points

    Return
    ------
    *rotation_angle: float (in degree)
    \tRotation angle from point2 to point1
    """
    theta1 = point_to_euler_angle(point1)
    theta2 = point_to_euler_angle(point2)
    rotation_angle = euler_substraction(theta2, theta1)
    return rotation_angle


def point_to_euler_angle(vec):
    '''
    @brief: covert a 2D point vec = (cos, sin) to euler angle (in degree)
    The output range is [-180, 180]
    '''
    vec = np.array(vec)
    theta = np.rad2deg(np.arctan2(vec[1], vec[0]))
    return theta


def fast_quat_frames_alignment(quaternion_frames_a,
                               quaternion_frames_b,
                               smooth=True,
                               smoothing_window=DEFAULT_SMOOTHING_WINDOW_SIZE,
                               param_range=(3, 7)):
    """implement a fast frame alignment based on orientation and offset of root
       of last frame of first motion and first frame of second motion
    """

    angle, offset = fast_quat_frames_transformation(quaternion_frames_a, quaternion_frames_b, param_range)
    transformed_frames = transform_quaternion_frames(quaternion_frames_b,
                                                     [0, angle, 0],
                                                     offset)
    # concatenate the quaternion_frames_a and transformed_framess
    if smooth:
        quaternion_frames = smoothly_concatenate_quaternion_frames(
            quaternion_frames_a,
            transformed_frames,
            window_size=smoothing_window)
    else:
        quaternion_frames = np.concatenate((quaternion_frames_a,
                                            transformed_frames))
    return quaternion_frames


def fast_euler_frames_alignment(euler_frames_a,
                                euler_frames_b,
                                smooth=True,
                                smoothing_window=DEFAULT_SMOOTHING_WINDOW_SIZE):
    angle, offset = fast_euler_frames_transformation(
        euler_frames_a, euler_frames_b)
    transformed_frames = transform_euler_frames(euler_frames_b,
                                                [0, angle, 0],
                                                offset)
    # concatenate the quaternion_frames_a and transformed_framess
    if smooth:
        euler_frames = smoothly_concatenate(
            euler_frames_a,
            transformed_frames,
            window_size=smoothing_window)
    else:
        quaternion_frames = np.concatenate((euler_frames_a,
                                            transformed_frames))
    return quaternion_frames


def calculate_point_cloud_distance(a, b, w=None):
    """
    calculates the distance between two point clouds with equal length and
    corresponding distances
    """
    assert len(a) == len(b)
    if w is None:
        w = np.ones(len(a))
    distance = 0
    n_points = len(a)
    for i in range(n_points):
        d = [a[i][0] - b[i][0], a[i][1] - b[i][1], a[i][2] - b[i][2]]
        distance += sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) * w[i]
    return distance / n_points


def rotate_and_translate_point(p, theta, offset_x, offset_z):
    """rotate and translate a 3d point, first rotation them translation
       theta is in radians
    """
    rotation_angles = [0, theta, 0]
    rotation_angles = np.rad2deg(rotation_angles)
    offset = [offset_x, 0, offset_z]
    transformed_point = transform_point(p, rotation_angles, offset)
    return transformed_point


def transform_point_cloud(point_cloud, theta, offset_x, offset_z):
    """
    transforms points in a point cloud by a rotation around y and a translation
    along x and z
    """
    transformed_point_cloud = []
    for p in point_cloud:
        if p is not None:
            transformed_point_cloud.append(
                rotate_and_translate_point(p, theta, offset_x, offset_z))
    return transformed_point_cloud




def quat_distance(quat_a, quat_b):
    # normalize quaternion vector first
    quat_a = np.asarray(quat_a)
    quat_b = np.asarray(quat_b)
    quat_a /= np.linalg.norm(quat_a)
    quat_b /= np.linalg.norm(quat_b)
    rotmat_a = quaternion_matrix(quat_a)
    rotmat_b = quaternion_matrix(quat_b)
    diff_mat = rotmat_a - rotmat_b
    tmp = np.ravel(diff_mat)
    diff = np.linalg.norm(tmp)
    return diff


def calculate_weighted_frame_distance_quat(quat_frame_a,
                                           quat_frame_b,
                                           weights):
    assert len(quat_frame_a) == len(quat_frame_b) and \
        len(quat_frame_a) == (len(weights) - 2) * LEN_QUAT + LEN_ROOT_POS
    diff = 0
    for i in range(len(weights) - 2):
        quat1 = quat_frame_a[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
        quat2 = quat_frame_b[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
        tmp = quat_distance(quat1, quat2)*weights[i]
        diff += tmp
    return diff




def extract_root_positions(frames):
    """

    :param frames: a list of euler or quaternion frames
    :return roots_2D: a list of root position in 2D space
    """
    roots_2D = []
    for i in range(len(frames)):
        position_2D = [frames[i][0], frames[i][2]]
        roots_2D.append(position_2D)

    return roots_2D


def pose_orientation_quat(quaternion_frame, param_range=(3,7), ref_offset=[0, 0, 1, 1]):
    """Estimate pose orientation from root orientation
    """
    ref_offset = np.array(ref_offset)
    q = quaternion_frame[param_range[0]:param_range[1]]
    rotmat = quaternion_matrix(q)
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0], rotated_point[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec

def pose_orientation_euler(euler_frame):
    ref_offset = np.array([0, 0, 1, 1])
    rot_angles = euler_frame[3:6]
    rot_angles_rad = np.deg2rad(rot_angles)
    rotmat = euler_matrix(rot_angles_rad[0],
                          rot_angles_rad[1],
                          rot_angles_rad[2],
                          'rxyz')
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0], rotated_point[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec


def get_trajectory_dir_from_2d_points(points):
    """Estimate the trajectory heading

    Parameters
    *\Points: numpy array
    Step 1: fit the points with a 2d straight line
    Step 2: estimate the direction vector from first and last point
    """
    points = np.asarray(points)
    dir_vector = points[-1] - points[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(*points.T)
    if isnan(slope):
        orientation_vec1 = np.array([0, 1])
        orientation_vec2 = np.array([0, -1])
    else:
        orientation_vec1 = np.array([slope, 1])
        orientation_vec2 = np.array([-slope, -1])
    if np.dot(
            orientation_vec1,
            dir_vector) > np.dot(
            orientation_vec2,
            dir_vector):
        orientation_vec = orientation_vec1
    else:
        orientation_vec = orientation_vec2
    orientation_vec = orientation_vec / np.linalg.norm(orientation_vec)
    return orientation_vec


def get_dir_from_2d_points(points):
    points = np.asarray(points)
    dir = points[-1] - points[2]
    dir = dir/np.linalg.norm(dir)
    return dir


def align_quaternion_frames_only_last_frame(quat_frames, prev_frames=None, transformation=None):
    """Concatenate and align quaternion frames based on previous frames or
        given transformation

    Parameters
    ----------
    * new_frames: list
         A list of quaternion frames
    * prev_frames: list
        A list of quaternion frames
   *  transformation: dict
       Contains translation and orientation in Euler angles
    Returns:
    --------
    * transformed_frames: np.ndarray
        Quaternion frames resulting from the back projection of s,
        transformed to fit to prev_frames.

    """
    # find alignment transformation or use given transformation
    if prev_frames is not None:
        angle, offset = fast_quat_frames_transformation(prev_frames, quat_frames)
        transformed_frames = transform_quaternion_frames([quat_frames[-1]],
                                                         [0, angle, 0],
                                                         offset)
        return transformed_frames
    elif prev_frames is None and transformation is not None:
        # align frames
        transformed_frames = transform_quaternion_frames([quat_frames[-1]], transformation["orientation"], transformation["position"])
        return transformed_frames
    else:
        return [quat_frames[-1]]


def align_quaternion_frames(quat_frames, prev_frames=None, transformation=None, param_range=(3,7)):
    """align quaternion frames based on previous frames or
        given transformation

    Parameters
    ----------
    * new_frames: list
         A list of quaternion frames
    * prev_frames: list
        A list of quaternion frames
   *  transformation: dict
       Contains translation and orientation in Euler angles
    Returns:
    --------
    * transformed_frames: np.ndarray
        Quaternion frames resulting from the back projection of s,
        transformed to fit to prev_frames.

    """
    # find alignment transformation or use given transformation
    if prev_frames is not None:
        angle, offset = fast_quat_frames_transformation(prev_frames, quat_frames, param_range)
        return transform_quaternion_frames(quat_frames, [0, angle, 0], offset)
    elif prev_frames is None and transformation is not None:
        # align frames
        return transform_quaternion_frames(quat_frames, transformation["orientation"], transformation["position"])
    else:
        return quat_frames


def rotate_euler_frames(euler_frames,
                        frame_idx,
                        ref_orientation):
    """
    Rotate a list of euler frames using the same rotation angle
    :param euler_frames: a list of euler frames to be rotated
    :param frame_idx: frame which uses to calculate rotation anlge
    :param ref_orientation: reference orientation for alignment
    :return rotated_frames: a list of rotated euler frames
    """
    test_ori = pose_orientation_euler(euler_frames[frame_idx])
    rot_angle = get_rotation_angle(ref_orientation, test_ori)
    translation = np.array([0, 0, 0])
    rotated_frames = transform_euler_frames(euler_frames,
                                            [0, rot_angle, 0],
                                            translation)
    return rotated_frames


def upvector_correction(filefolder,
                        frame_idx,
                        ref_upvector,
                        save_folder):
    """
    Correct the up vector of bvh files in given file folder
    :param filefolder:
    :param ref_upvector:
    :return:
    """
    if not filefolder.endswith(os.sep):
        filefolder += os.sep
    if not save_folder.endswith(os.sep):
        save_folder += os.sep
    bvhfiles = glob.glob(filefolder+'*.bvh')
    for item in bvhfiles:
        bvhreader = BVHReader(item)
        rotated_frames = rotate_euler_frames_about_x_axis(bvhreader.frames,
                                                          frame_idx,
                                                          ref_upvector)
        save_filename = save_folder + bvhreader.filename
        BVHWriter(save_filename, bvhreader, rotated_frames, bvhreader.frame_time, False)


def rotate_euler_frames_about_x_axis(euler_frames,
                                     frame_idx,
                                     ref_upvector):
    sample_upvector = pose_up_vector_euler(euler_frames[frame_idx])
    rot_angle = get_rotation_angle(ref_upvector, sample_upvector)
    translation = np.array([0, 0, 0])
    rotated_frames = transform_euler_frames(euler_frames,
                                            [rot_angle, 0, 0],
                                            translation)
    return rotated_frames


def is_vertical_pose_euler(euler_frame):
    ref_vec = np.array([1, 0, 0, 1])
    rot_angles = np.deg2rad(euler_frame[3:6])
    rotmat = euler_matrix(rot_angles[0],
                          rot_angles[1],
                          rot_angles[2],
                          'rxyz')
    rotated_vec = np.dot(rotmat, ref_vec)
    if rotated_vec[1] > 0.99:
        return True
    else:
        return False


def is_vertical_pose_quat(quat_frame):
    quat_hip = np.asarray(quat_frame[3:7])
    ref_vec = np.array([1, 0, 0, 1])
    rotmat = quaternion_matrix(quat_hip)
    rotated_vec = np.dot(rotmat, ref_vec)
    if rotated_vec[1] > 0.99:
        return True
    else:
        return False


def pose_up_vector_euler(euler_frame):
    """
    Estimate up vector of given euler frame. Assume the pose is aligned in negative Z axis, so the projection of
    3D up vector into XOY plane (Y is vertical axis) is taken into consideration.
    :param euler_frame:
    :return:
    """
    ref_offset = np.array([1, 0, 0, 1])
    rot_angles = euler_frame[3:6]
    rot_angles_rad = np.deg2rad(rot_angles)
    rotmat = euler_matrix(rot_angles_rad[0],
                          rot_angles_rad[1],
                          rot_angles_rad[2],
                          'rxyz')
    rotated_point = np.dot(rotmat, ref_offset)
    up_vec = np.array([rotated_point[0], rotated_point[1]])
    up_vec /= np.linalg.norm(up_vec)
    return up_vec


def pose_up_vector_quat(quat_frame):
    """
    Estimate up vector of given quaternion frame. Assume the pose is aligned in negative Z axis, so the projection of
    3D up vector into XOY plane (Y is vertical axis) is taken into consideration.
    :param quat_frame:
    :return:
    """
    ref_offset = np.array([1, 0, 0, 1])
    rotmat = quaternion_matrix(quat_frame[3:7])
    rotated_point = np.dot(rotmat, ref_offset)
    up_vec = np.array([rotated_point[0], rotated_point[2]])
    up_vec /= np.linalg.norm(up_vec)
    return up_vec


def quaternion_rotate_vector(q, vector):
    """ src: http://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
    Args:
        q:
        vector:

    Returns:

    """
    tmp_result = quaternion_multiply(q, vector)
    return quaternion_multiply(tmp_result, quaternion_conjugate(q))[1:]


def quaternion_from_vector_to_vector(a, b):
    """src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    http://wiki.ogre3d.org/Quaternion+and+Rotation+Primer"""

    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    if np.dot(q,q) != 0:
        return q/ np.linalg.norm(q)
    else:
        idx = np.nonzero(a)[0]
        q = np.array([0, 0, 0, 0])
        q[1 + ((idx + 1) % 2)] = 1 # [0, 0, 1, 0] for a rotation of 180 around y axis
        return q

def quaternion_rotation_from_vector_to_vector_frames(a,b):
    """
    Generates quaternions for a set of vectors to a single vector.

    :param a: np.array( n, 3)
    :param b: np.array( 3)
    :return: quaternion rotations: np.array(n, 4)
    """
    v = np.cross(a, [b])
    w = np.sqrt((np.linalg.norm(a, axis=-1) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.append( np.reshape(w, (-1, 1)), v, axis=1)

    dotted = lambda q: np.dot(q,q) != 0

    # np.dot(q,q) != 0 condition:
    condition = list(map(dotted, q))

    # if np.dot(q, q) != 0:
    q[condition] = q[condition] / np.reshape(np.linalg.norm(q, axis=1), (-1,1))

    # else:
    nonzeros = lambda a: np.nonzero(a)[0][0]
    not_conditions = list(map(lambda q: not q, condition))
    idx = list(map(nonzeros, a[not_conditions]))


    elseCases = np.zeros((len(idx), 4))
    for i in range(len(idx)):
        elseCases[i][int(1 + ((idx[i] + 1) % 2))] = 1
    q[not_conditions] = elseCases

    return q

def matrix_rotation_from_vector_to_vector_frames(a, b):
    """
    Generates rotation matrices for a set of vectors to a single vector.

    :param a: np.array( n, 3)
    :param b: np.array( 3)
    :return: homogeneous rotation matrices: np.array(n, 4, 4)
    """
    q = quaternion_rotation_from_vector_to_vector_frames(a, b)
    qm = np.array(list(map(quaternion_matrix, q)))
    return qm

def quaternion_angle_axis_frames(q):
    """
    Transforms a set of quaternion rotations to angle axis representation.

    :param q: np.array(n, 4, 4)
    :return: angles: np.array(n, 1)
    :return: axis: np.array(n, 3,3)
    """
    q /= np.reshape(np.linalg.norm(q, axis=1), (-1,1))
    s = np.sqrt(1 - (q[:,0] ** 2))
    s[s==0] = 0.001
    angles = 2.0 * np.arccos(q[:,0])
    axis = q[:,1:] / s[..., np.newaxis]
    return (angles, axis)


def convert_euler_frame_to_reduced_euler_frame(bvhreader, euler_frame):
    if type(euler_frame) != list:
        euler_frame = list(euler_frame)
    reduced_euler_frame = euler_frame[:3]
    for node_name in bvhreader.node_names:
        if not node_name.startswith('Bip') and 'EndSite' not in node_name:
            x_idx = bvhreader.node_channels.index((node_name, 'Xrotation'))
            reduced_euler_frame += euler_frame[x_idx: x_idx + LEN_EULER]
    return reduced_euler_frame


def convert_euler_frames_to_reduced_euler_frames(bvhreader, euler_frames):
    reduced_euler_frames = []
    for euler_frame in euler_frames:
        reduced_euler_frames.append(convert_euler_frame_to_reduced_euler_frame(bvhreader, euler_frame))
    return reduced_euler_frames


def point_rotation_by_quaternion(point, q):
    """
    Rotate a 3D point by quaternion
    :param point: [x, y, z]
    :param q: [qw, qx, qy, qz]
    :return rotated_point: [x, y, ]
    """
    r = [0, point[0], point[1], point[2]]
    q_conj = [q[0], -1*q[1], -1*q[2], -1*q[3]]
    return quaternion_multiply(quaternion_multiply(q, r), q_conj)[1:]

def quaternion_normalized(q):
	"""
	normalizes a set of quaternions
	:param q: np.array(n, 4)
	:return: np.array(n, 4)
	"""
	sqs = np.sum(q**2, axis=-1)
	q[sqs > 0.0001] =q[sqs > 0.0001] * 1 / sqs[sqs > 0.0001]
	q[sqs <= 0.0001] = [1.0, 0.0, 0.0, 0.0]
	#q = q / np.linalg.norm(q, axis=-1)
	#if np.sum(q) < 0:
	#	q = -q
	return q


def quaternions_from_matrix(ts):
    """
	Transforms a set of quaternions from a matrix. This code was taken from holdens implementation.

	Remark: For some reason, this code does not accurately transform quaternions from matrices. For some reason, the singularity (q = -q) is not accurately solved.

    :param ts: np.array(n, d, 3, 3)
    :return: np.array(n, d, 4)
    """
    d0, d1, d2 = ts[..., 0,0], ts[..., 1,1], ts[..., 2,2]

    q0 = (d0 + d1 + d2 + 1.0) / 4.0
    q1 = (d0 - d1 - d2 + 1.0) / 4.0
    q2 = (-d0 + d1 - d2 + 1.0) / 4.0
    q3 = (-d0 - d1 + d2 + 1.0) / 4.0

    q0 = np.sqrt(q0.clip(0, None))
    q1 = np.sqrt(q1.clip(0, None))
    q2 = np.sqrt(q2.clip(0, None))
    q3 = np.sqrt(q3.clip(0, None))

    c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
    c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
    c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
    c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)

    q1[c0] *= np.sign(ts[c0, 2, 1] - ts[c0, 1, 2])
    q2[c0] *= np.sign(ts[c0, 0, 2] - ts[c0, 2, 0])
    q3[c0] *= np.sign(ts[c0, 1, 0] - ts[c0, 0, 1])

    q0[c1] *= np.sign(ts[c1, 2, 1] - ts[c1, 1, 2])
    q2[c1] *= np.sign(ts[c1, 1, 0] + ts[c1, 0, 1])
    q3[c1] *= np.sign(ts[c1, 0, 2] + ts[c1, 2, 0])

    q0[c2] *= np.sign(ts[c2, 0, 2] - ts[c2, 2, 0])
    q1[c2] *= np.sign(ts[c2, 1, 0] + ts[c2, 0, 1])
    q3[c2] *= np.sign(ts[c2, 2, 1] + ts[c2, 1, 2])

    q0[c3] *= np.sign(ts[c3, 1, 0] - ts[c3, 0, 1])
    q1[c3] *= np.sign(ts[c3, 2, 0] + ts[c3, 0, 2])
    q2[c3] *= np.sign(ts[c3, 2, 1] + ts[c3, 1, 2])

    qs = np.empty((ts.shape[0], ts.shape[1], 4))
    qs[..., 0] = q0
    qs[..., 1] = q1
    qs[..., 2] = q2
    qs[..., 3] = q3

    #qs = np.reshape(qs, (qs.shape[0], qs.shape[1], 4))

    return qs