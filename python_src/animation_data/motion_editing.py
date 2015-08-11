# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 20:11:35 2015

@author: Erik, Han, Markus
"""

import numpy as np
from math import sqrt, radians, sin, cos, isnan
import fk3
from itertools import izip
from copy import deepcopy
import time
from scipy import stats  # linear regression
from external.transformations import quaternion_matrix, euler_from_matrix, \
    quaternion_from_matrix, euler_matrix, \
    quaternion_multiply

DEFAULT_SMOOTHING_WINDOW_SIZE = 20
fk_funcs = [
    fk3.one_joint_fk,
    fk3.two_joints_fk,
    fk3.three_joints_fk,
    fk3.four_joints_fk,
    fk3.five_joints_fk,
    fk3.six_joints_fk,
    fk3.seven_joints_fk,
    fk3.eight_joints_fk,
]

fk_func_jacs = [
    fk3.one_joint_fk_jacobian,
    fk3.two_joints_fk_jacobian,
    fk3.three_joints_fk_jacobian,
    fk3.four_joints_fk_jacobian,
    fk3.five_joints_fk_jacobian,
    fk3.six_joints_fk_jacobian,
    fk3.seven_joints_fk_jacobian,
    fk3.eight_joints_fk_jacobian,
]


def euler_to_quaternion(euler_angles, rotation_order=['Xrotation', 'Yrotation', 'Zrotation'],
                        filter_value=True):
    """Convert euler angles to quaternion vector [qw, qx, qy, qz]

    Parameters
    ----------
    * euler_angles: list of floats
    \tA list of ordered euler angles in degress
    * rotation_order: Iteratable
    \t a list that specifies the rotation axis corresponding to the values in euler_angles
    * filter_values: Bool
    \t enforce a unique rotation representation    

    """
    # convert euler angles into rotation matrix, then convert rotation matrix
    # into quaternion
    assert len(euler_angles) == 3, ('The length of euler angles should be 3!')
    # convert euler angle from degree into radians
    euler_angles = np.deg2rad(euler_angles)
    if rotation_order[0] == 'Xrotation':
        if rotation_order[1] == 'Yrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rxyz')
        elif rotation_order[1] == 'Zrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rxzy')
    elif rotation_order[0] == 'Yrotation':
        if rotation_order[1] == 'Xrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='ryxz')
        elif rotation_order[1] == 'Zrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='ryzx')
    elif rotation_order[0] == 'Zrotation':
        if rotation_order[1] == 'Xrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rzxy')
        elif rotation_order[1] == 'Yrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rzyx')
    # convert rotation matrix R into quaternion vector (qw, qx, qy, qz)
    q = quaternion_from_matrix(R)
    # filter the quaternion see
    # http://physicsforgames.blogspot.de/2010/02/quaternions.html
    if filter_value:
        dot = np.sum(q)
        if dot < 0:
            q = -q
    return q[0], q[1], q[2], q[3]


def quaternion_to_euler(q, rotation_order=['Xrotation',
                                           'Yrotation',
                                           'Zrotation']):
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
    q = np.asarray(q)
    # normalize quaternion vector
    q = q / np.linalg.norm(q)
    # create rotation matrix from quaternion
    Rq = quaternion_matrix(q)
    # map rotation order
    if rotation_order[0] == 'Xrotation':
        if rotation_order[1] == 'Yrotation':
            euler_angles = np.rad2deg(euler_from_matrix(Rq, 'rxyz'))
        elif rotation_order[1] == 'Zrotation':
            euler_angles = np.rad2deg(euler_from_matrix(Rq, 'rxzy'))
    elif rotation_order[0] == 'Yrotation':
        if rotation_order[1] == 'Xrotation':
            euler_angles = np.rad2deg(euler_from_matrix(Rq, 'ryxz'))
        elif rotation_order[1] == 'Zrotation':
            euler_angles = np.rad2deg(euler_from_matrix(Rq, 'ryzx'))
    elif rotation_order[0] == 'Zrotation':
        if rotation_order[1] == 'Xrotation':
            euler_angles = np.rad2deg(euler_from_matrix(Rq, 'rzxy'))
        elif rotation_order[1] == 'Yrotation':
            euler_angles = np.rad2deg(euler_from_matrix(Rq, 'rzyx'))
    return euler_angles.tolist()


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

        return zip(it[0::4], it[1::4], it[2::4], it[3::4])

    def get_euler_frame(quaternionion_frame):
        """Converts a quaternion frame into an Euler frame"""

        euler_frame = list(quaternionion_frame[:3])
        for quaternion in gen_4_tuples(quaternionion_frame[3:]):
            euler_frame += quaternion_to_euler(quaternion)

        return euler_frame

    euler_frames = map(get_euler_frame, quaternion_frames)

    return np.array(euler_frames)


def euler_substraction(theta1, theta2):
    '''
    @brief: compute the angular distance from theta1 to theta2, positive value is anti-clockwise, negative is clockwise
    @param theta1, theta2: angles in degree
    '''
    theta1 = theta1 % 360
    theta2 = theta2 % 360
    if theta1 > 180:
        theta1 = theta1 - 360
    elif theta1 < -180:
        theta1 = theta1 + 360

    if theta2 > 180:
        theta2 = theta2 - 360
    elif theta2 < - 180:
        theta2 = theta2 + 360

    theta = theta1 - theta2
    if theta > 180:
        theta = theta - 360
    elif theta < -180:
        theta = theta + 360
    if theta > 180 or theta < - 180:
        raise ValueError(' exception value')
    return theta


def get_cartesian_coordinates_from_quaternion(skeleton,
                                              node_name,
                                              quaternion_frame):
    """Returns cartesian coordinates for one node at one frame. Modified to
     handle frames with omitted values for joints starting with "Bip"

    Parameters
    ----------

    * node_name: String
    \tName of node
     * skeleton: Skeleton
    \tBVH data structure read from a file
    * node_name_map: dict
    \tA map from node name to index in the euler frame

    """
    if skeleton.node_names[node_name]["level"] == 0:
        root_frame_position = quaternion_frame[:3]
        root_node_offset = skeleton.node_names[node_name]["offset"]

        return [t + o for t, o in
                izip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(skeleton.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list

        offsets = [skeleton.node_names[nodename]["offset"]
                   for nodename in chain_names]
        root_position = quaternion_frame[:3].flatten()
        offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]

        j_matrices = []
        count = 0
        for node_name in chain_names:
            index = skeleton.node_name_map[node_name] * 4 + 3
            j_matrix = quaternion_matrix(quaternion_frame[index: index + 4])
            j_matrix[:, 3] = offsets[count] + [1]
            j_matrices.append(j_matrix)
            count += 1

        global_matrix = np.identity(4)
        for j_matrix in j_matrices:
            global_matrix = np.dot(global_matrix, j_matrix)

        point = np.array([0, 0, 0, 1])
        point = np.dot(global_matrix, point)
        return point[:3].tolist()


def get_cartesian_coordinates_from_euler_full_skeleton(bvh_reader,
                                                       skeleton,
                                                       node_name,
                                                       euler_frame):
    """Return cartesian coordinates for one node at one frame, include the 
       skipped joints starting with "Bip"
    """
    if bvh_reader.node_names[node_name]["level"] == 0:
        root_frame_position = euler_frame[:3]
        root_node_offset = bvh_reader.node_names[node_name]["offset"]

        return [t + o for t, o in
                izip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(skeleton.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list

        eul_angles = []
        index = 0
        for nodename in chain_names:
            indeces = []
            for channel in bvh_reader.node_names[nodename]["channels"]:
                if channel.endswith("rotation"):
                    indeces.append(
                        bvh_reader.node_channels.index((nodename, channel)))
            eul_angles.append(euler_frame[indeces])
            index += 1

        # print chain_names,
        # bvh_reader.node_names.keys().index("RightShoulder")*3 +
        # 3,len(euler_frame)
        rad_angles = (map(radians, eul_angle) for eul_angle in eul_angles)

        thx, thy, thz = map(list, zip(*rad_angles))

        offsets = [bvh_reader.node_names[nodename]["offset"]
                   for nodename in chain_names]

        # Add root offset to frame offset list
        root_position = euler_frame[:3]
        offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]

        ax, ay, az = map(list, izip(*offsets))

        # f_idx identifies the kinematic forward transform function
        # This does not lead to a negative index because the root is
        # handled separately

        f_idx = len(ax) - 2
        # print "f",f_idx
        if len(ax) - 2 < len(fk_funcs):
            return fk_funcs[f_idx](ax, ay, az, thx, thy, thz)
        else:
            return [0, 0, 0]


def get_cartesian_coordinates_from_euler(skeleton, node_name, euler_frame):
    """Returns cartesian coordinates for one node at one frame. Modified to
     handle frames with omitted values for joints starting with "Bip"

    Parameters
    ----------

    * node_name: String
    \tName of node
     * skeleton: Skeleton
    \t skeleton structure read from a file
    * frame_number: Integer
    \tAnimation frame number that gets extracted
    * node_name_map: dict
    \tA map from node name to index in the euler frame

    """
    # print len(euler_frame),node_name

    if skeleton.node_names[node_name]["level"] == 0:
        root_frame_position = euler_frame[:3]
        root_node_offset = skeleton.node_names[node_name]["offset"]

        return [t + o for t, o in
                izip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(skeleton.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list

#        if node_name == "Head_EndSite":
#                print chain_names
        eul_angles = []
        for nodename in chain_names:
            index = skeleton.node_name_map[nodename] * 3 + 3
            eul_angles.append(euler_frame[index:index + 3])

        # print chain_names,
        # bvh_reader.node_names.keys().index("RightShoulder")*3 +
        # 3,len(euler_frame)
        rad_angles = (map(radians, eul_angle) for eul_angle in eul_angles)

        thx, thy, thz = map(list, zip(*rad_angles))

        offsets = [skeleton.node_names[nodename]["offset"]
                   for nodename in chain_names]

        # Add root offset to frame offset list
        root_position = euler_frame[:3]
        offsets[0] = [r + o for r, o in izip(root_position, offsets[0])]

        ax, ay, az = map(list, izip(*offsets))

        # f_idx identifies the kinematic forward transform function
        # This does not lead to a negative index because the root is
        # handled separately

        f_idx = len(ax) - 2
        # print "f",f_idx
        if len(ax) - 2 < len(fk_funcs):
            return fk_funcs[f_idx](ax, ay, az, thx, thy, thz)
        else:
            return [0, 0, 0]


def convert_euler_frame_to_cartesian_frame(skeleton, euler_frame):
    """
    converts euler frames to cartesian frames by calling get_cartesian_coordinates for each joint
    """
    # print euler_frame.shape
    cartesian_frame = []
    for node_name in skeleton.node_names:
        # ignore Bip joints and end sites
        if not node_name.startswith("Bip") and "children" in skeleton.node_names[node_name].keys():
            cartesian_frame.append(
                get_cartesian_coordinates_from_euler(skeleton, node_name, euler_frame))

    return cartesian_frame


def convert_quaternion_frame_to_cartesian_frame(skeleton, quat_frame):
    """
    Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
    """
    cartesian_frame = []
    for node_name in skeleton.node_names:
        # ignore Bip joints and end sites
        if not node_name.startswith("Bip") and "children" in skeleton.node_names[node_name].keys():
            cartesian_frame.append(get_cartesian_coordinates_from_quaternion(
                skeleton, node_name, quat_frame))  # get_cartesian_coordinates2

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
        raise ValueError("two point cloud should have the same number points")
    n_points = len(a)
    numerator_left = 0
    numerator_right = 0
    denominator_left = 0
    denominator_right = 0
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
    #theta = np.arctan(numerator/denominator)
    #theta = math.atan(numerator/denominator)
    offset_x = 0
    offset_z = 0
    offset_x = (weighted_sum_a_x - weighted_sum_b_x *
                np.cos(theta) - weighted_sum_b_z * np.sin(theta)) / sum_of_weights
    offset_z = (weighted_sum_a_z + weighted_sum_b_x *
                np.sin(theta) - weighted_sum_b_z * np.cos(theta)) / sum_of_weights

    return theta, offset_x, offset_z


def convert_euler_frames_to_cartesian_frames(skeleton, euler_frames):
    """
    converts to euler frames to cartesian frames
    """

    cartesian_frames = []
    for euler_frame in euler_frames:
        cartesian_frames.append(
            convert_euler_frame_to_cartesian_frame(skeleton, euler_frame))
    return np.array(cartesian_frames)


def find_aligning_transformation(skeleton, euler_frames_a, euler_frames_b):
    """
    performs alignment of the point clouds based on the poses at the end of
    euler_frames_a and the start of euler_frames_b
    Returns the rotation around y axis in radians, x offset and z offset
    """
    point_cloud_a = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_a[-1])
    point_cloud_b = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_b[0])
    weights = skeleton.get_joint_weights()
    theta, offset_x, offset_z = align_point_clouds_2D(
        point_cloud_a, point_cloud_b, weights)
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


def transform_point(point,
                    euler_angles,
                    offset,
                    origin=None,
                    rotation_order=["Xrotation",
                                    "Yrotation",
                                    "Zrotation"]):
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
    assert len(point) == 3,  ('the point should be a list of length 3')
    # translate point to original point
    point = np.asarray(point)
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    if type(point) is not list:
        point = list(point)
    point.append(1.0)
    # generate rotation matrix based on rotation order
    assert len(
        euler_angles) == 3, ('The length of rotation angles should be 3')
    euler_angles = np.deg2rad(euler_angles)
    if rotation_order[0] == 'Xrotation':
        if rotation_order[1] == 'Yrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rxyz')
        elif rotation_order[1] == 'Zrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rxzy')
    elif rotation_order[0] == 'Yrotation':
        if rotation_order[1] == 'Xrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='ryxz')
        elif rotation_order[1] == 'Zrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='ryzx')
    elif rotation_order[0] == 'Zrotation':
        if rotation_order[1] == 'Xrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rzxy')
        elif rotation_order[1] == 'Yrotation':
            R = euler_matrix(euler_angles[0],
                             euler_angles[1],
                             euler_angles[2],
                             axes='rzyx')
    rotated_point = np.dot(R, point)
    if origin is not None:

        rotated_point[:3] += origin
#    print rotated_point[:3]
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

#    print quat_frame
    if rotation_order is None:
        rotation_order = ["Xrotation", "Yrotation", "Zrotation"]
    transformed_frame = quat_frame[:]
#    original_point_copy = deepcopy(original_point)
    transformed_frame[:3] = transform_point(quat_frame[:3],
                                            angles,
                                            offset,
                                            origin=origin)
#    transformed_frame[:3] = transform_point(quat_frame[:3], [0, 0, 0], offset)
    q = euler_to_quaternion(angles, rotation_order)
    oq = quat_frame[3:7]
    rotated_q = quaternion_multiply(q, oq)
    transformed_frame[3:7] = rotated_q
    return transformed_frame


def transform_quaternion_frames(quat_frames, angles, offset):
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
    quat_frames_copy = deepcopy(quat_frames)
    offset = np.asarray(offset)
    original_point = quat_frames[0][:3]

    transformed_quat_frames = []
    for frame in quat_frames_copy:
        transformed_quat_frames.append(transform_quaternion_frame(frame,
                                                                  angles,
                                                                  offset,
                                                                  original_point))
    return np.array(transformed_quat_frames)


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
    for i in xrange(n_joints):
        for j in xrange(n_frames - 1):
            q1 = np.array(quaternion_frames[j][3 + i * 4: 3 + (i + 1) * 4])
            q2 = np.array(quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
            if np.dot(q1, q2) < 0:
                quaternion_frames[
                    j + 1][3 + i * 4:3 + (i + 1) * 4] = -quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    d = float(discontinuity)
    w = float(window)
    smoothing_factors = []
    for f in xrange(n_frames):
        value = 0.0
        if d - w <= f < d:
            tmp = (f - d + w) / w
            value = 0.5 * tmp**2
        elif d <= f <= d + w:
            tmp = (f - d + w) / w
            value = -0.5 * tmp**2 + 2 * tmp - 2
        smoothing_factors.append(value)
    smoothing_factors = np.array(smoothing_factors)
    new_quaternion_frames = []
    for i in xrange(len(quaternion_frames[0])):
        current_value = quaternion_frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        new_value = current_value + (magnitude * smoothing_factors)
        new_quaternion_frames.append(new_value)
    new_quaternion_frames = np.array(new_quaternion_frames).T
    return new_quaternion_frames


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
    for f in xrange(len(euler_frames)):
        value = 0.0
        if d - s <= f < d:
            tmp = ((f - d + s) / s)
            value = 0.5 * tmp**2
        elif d <= f <= d + s:
            tmp = ((f - d + s) / s)
            value = -0.5 * tmp**2 + 2 * tmp - 2

        smoothing_faktors.append(value)

    smoothing_faktors = np.array(smoothing_faktors)
    new_euler_frames = []
    for i in xrange(len(euler_frames[0])):
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


def smoothly_concatenate(euler_frames_a, euler_frames_b,  window_size=20):
    euler_frames = np.concatenate((euler_frames_a, euler_frames_b), axis=0)
    euler_frames = smooth_motion(euler_frames, d, window_size)
    return euler_frames


def smoothly_concatenate_quaternion_frames(quaternion_frames_a,
                                           quaternion_frames_b,
                                           window_size=20):
    quaternion_frames = np.concatenate((quaternion_frames_a,
                                        quaternion_frames_b), axis=0)
    d = len(quaternion_frames_a)
    quaternion_frames = smooth_quaternion_frames(quaternion_frames,
                                                 d,
                                                 window_size)
    return quaternion_frames


def shift_euler_frames_to_ground(euler_frames,
                                 ground_contact_joint,
                                 skeleton):
    """
    shift all euler frames of motion to ground, which means the y-axis for
    gound contact joint should be 0
    Step 1: apply forward kinematic to compute global position for ground
            contact joint for each frame
    Setp 2: find the offset from ground contact joint to ground, and shift
            corresponding frame based on offset
    """
    tmp_frames = deepcopy(euler_frames)
    for frame in tmp_frames:
        contact_point_position = get_cartesian_coordinates_from_euler(skeleton,
                                                                      ground_contact_joint,
                                                                      frame)
        offset_y = contact_point_position[1]
        # shift root position by offset_y
        frame[1] = frame[1] - offset_y
    return tmp_frames


def align_frames(skeleton,
                 euler_frames_a,
                 euler_frames_b,
                 smooth=True):
    """
    calls find_aligning_transformation and concatenates the frames based on the
    resulting transformation
     Parameters
    ----------
    *skeleton: Skeleton
    \tUsed to extract hierarchy information.
    *euler_frames_a: np.ndarray
    \List of frames where the rotation is represented as euler angles in degrees.
    *euler_frames_b: np.ndarray
    \List of frames where the rotation is represented as euler angles in degrees.
    *node_name_map: dict
    \tDictionary that maps joint names to their order in the bvh file ignoring "Bip" joints
    *smooth: bool
    \t Sets whether or not smoothing is supposed to be applied on the at the transition.
     Returns
    -------
    *aligned_frames : np.ndarray
    \tAligned and optionally smoothed motion
    """
    theta, offset_x, offset_z = find_aligning_transformation(skeleton, euler_frames_a,
                                                             euler_frames_b)

    # apply 2d transformation
    offset = np.array([offset_x, 0, offset_z])
    angles = [0, np.degrees(theta), 0]
    euler_frames_b = transform_euler_frames(euler_frames_b, angles,
                                            offset)

    # concatenate frames and optionally apply smoothing
    if smooth:
        euler_frames = smoothly_concatenate(euler_frames_a, euler_frames_b)
    else:
        euler_frames = np.concatenate((euler_frames_a, euler_frames_b), axis=0)
    return euler_frames


def align_quaternion_frames(skeleton,
                            quaternion_frames_a,
                            quaternion_frames_b,
                            node_name_map=None,
                            smooth=True):
    # find translation and rotation by using last frame from quaternion_frames_a
    # and the first frame from quaternion_frame_b
    # convert quaternion frame to euler frame
    last_euler_frame = np.ravel(quaternion_frames_a[-1])
    first_euler_frame = np.ravel(quaternion_frames_b[0])
    point_cloud_a = convert_quaternion_frame_to_cartesian_frame(
        skeleton, last_euler_frame)
    point_cloud_b = convert_quaternion_frame_to_cartesian_frame(
        skeleton, first_euler_frame)

    theta, offset_x, offset_z = align_point_clouds_2D(
        point_cloud_a, point_cloud_b, skeleton.joint_weights)
    rotation_angle = [0, np.rad2deg(theta), 0]
    translation = [offset_x, 0, offset_z]
    transformed_quaternion_frames_b = transform_quaternion_frames(
        quaternion_frames_b, rotation_angle, translation)
    if smooth:
        quaternion_frames = smoothly_concatenate_quaternion_frames(quaternion_frames_a,
                                                                   transformed_quaternion_frames_b,
                                                                   window_size=20)
    else:
        quaternion_frames = np.concatenate((quaternion_frames_a,
                                            transformed_quaternion_frames_b))
    return quaternion_frames


def fast_quat_frames_transformation(quaternion_frames_a,
                                    quaternion_frames_b):
    dir_vec_a = pose_orientation(quaternion_frames_a[-1])
    dir_vec_b = pose_orientation(quaternion_frames_b[0])
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = quaternion_frames_a[-1][0] - quaternion_frames_b[0][0]
    offset_z = quaternion_frames_a[-1][2] - quaternion_frames_b[0][2]
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
    theta1 = fromPointToEulerAngle(point1)
    theta2 = fromPointToEulerAngle(point2)
    rotation_angle = euler_substraction(theta2, theta1)
    return rotation_angle


def fromPointToEulerAngle(vec):
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
                               smoothing_window=DEFAULT_SMOOTHING_WINDOW_SIZE):
    """implement a fast frame alignment based on orientation and offset of root
       of last frame of first motion and first frame of second motion 
    """

    angle, offset = fast_quat_frames_transformation(
        quaternion_frames_a, quaternion_frames_b)
    transformed_frames = transform_quaternion_frames(quaternion_frames_b,
                                                     [0, angle, 0],
                                                     offset)
    # concatenate the quaternion_frames_a and transformed_framess
    if smooth:
        quaternion_frames = smoothly_concatenate_quaternion_frames(quaternion_frames_a,
                                                                   transformed_frames,
                                                                   window_size=smoothing_window)
    else:
        quaternion_frames = np.concatenate((quaternion_frames_a,
                                            transformed_frames))
    return quaternion_frames


def calculate_point_cloud_distance(a, b):
    """
    calculates the distance between two point clouds with equal length and
    corresponding distances
    """
    assert len(a) == len(b)
    distance = 0
    n_points = len(a)
    for i in xrange(n_points):
        d = [a[i][0] - b[i][0], a[i][1] - b[i][1], a[i][2] - b[i][2]]
        distance += sqrt(d[0]**2 + d[1]**2 + d[2]**2)
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


def calculate_pose_distance(skeleton, euler_frames_a, euler_frames_b):
    ''' Converts euler frames to point clouds and finds the aligning transformation
        and calculates the distance after the aligning transformation
    '''

#    theta, offset_x, offset_z = find_aligning_transformation(bvh_reader, euler_frames_a, euler_frames_b, node_name_map)
    point_cloud_a = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_a[-1])
    point_cloud_b = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_b[0])

    weights = skeleton.joint_weights
    theta, offset_x, offset_z = align_point_clouds_2D(
        point_cloud_a, point_cloud_b, weights)
    t_point_cloud_b = transform_point_cloud(
        point_cloud_b, theta, offset_x, offset_z)
    error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    return error


def calculate_frame_distance(skeleton,
                             euler_frame_a,
                             euler_frame_b,
                             return_transform=False):
    point_cloud_a = convert_euler_frame_to_cartesian_frame(skeleton,
                                                           euler_frame_a)
    point_cloud_b = convert_euler_frame_to_cartesian_frame(skeleton,
                                                           euler_frame_b)
    weights = skeleton.joint_weights

    theta, offset_x, offset_z = align_point_clouds_2D(
        point_cloud_a, point_cloud_b, weights)
    t_point_cloud_b = transform_point_cloud(
        point_cloud_b, theta, offset_x, offset_z)
    error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    if return_transform:
        return error, theta, offset_x, offset_z
    else:
        return error


def calculate_pose_distances_from_low_dim(skeleton, mm_models, X, Y):
    """
    Converts low dimensional vectors to euler vectors and calculates the
    pose distance error by calling calculate_pose_distance
    Parameters
    ----------
    * skeleton: BVHReader
    \tContains the skeleton definition needed for the point cloud conversion

    * mm_models: Dict of MotionPrimitives
    \tContains the motion primitives for X and Y data

     * X: List
    \tList of low dimensional vectors

    * Y: List
    \tList of low dimensional vectors
    """
    print "test"
    assert len(X) == len(Y)
    n = len(X)
    errors = []
    for i in xrange(n):
        euler_frames_a = mm_models["X"].back_project(X[i]).get_motion_vector()
        euler_frames_b = mm_models["Y"].back_project(Y[i]).get_motion_vector()
        error = calculate_pose_distance(
            skeleton, euler_frames_a, euler_frames_b)
        errors.append(error)

    return errors


def extract_root_positions(frames):

    roots_2D = []
    for i in xrange(len(frames)):
        position_2D = np.array([frames[i][0], frames[i][2]])
        # print "sample",position2D
        roots_2D.append(position_2D)

    return np.array(roots_2D)


def get_orientation_vec(euler_frames):
    """ get orientation for last few frames of a motion sequence
    """
    # get 3d root position from motion sample for last 5 frames

    points = extract_root_positions(euler_frames)

    slope, intercept, r_value, p_value, std_err = stats.linregress(*points.T)

    dir_vector = points[-1] - points[0]

    if isnan(slope):
        orientation_vec1 = np.array([0, 1])
        orientation_vec2 = np.array([0, -1])
    else:
        orientation_vec1 = np.array([1, slope])
        orientation_vec2 = np.array([-1, -slope])
    if np.dot(orientation_vec1, dir_vector) > np.dot(orientation_vec2, dir_vector):
        orientation_vec = orientation_vec1
    else:
        orientation_vec = orientation_vec2
    orientation_vec = orientation_vec / np.linalg.norm(orientation_vec)
    return orientation_vec


def pose_orientation(quaternion_frame):
    """Estimate pose orientation from root orientation
    """
    ref_offset = np.array([0, 0, 1, 1])
    rotmat = quaternion_matrix(quaternion_frame[3:7])
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0], rotated_point[2]])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
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
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    return dir_vec


def get_trajectory_dir_from_2d_points(points):
    """Estimate the trajectory heading

    Parameters
    *\Points: numpy array
    Step 1: fit the points with a 2d straight line
    Step 2: estimate the direction vector from first and last point
    """
    dir_vector = points[-1] - points[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(*points.T)
    if isnan(slope):
        orientation_vec1 = np.array([0, 1])
        orientation_vec2 = np.array([0, -1])
    else:
        orientation_vec1 = np.array([slope, 1])
        orientation_vec2 = np.array([-slope, -1])
    if np.dot(orientation_vec1, dir_vector) > np.dot(orientation_vec2, dir_vector):
        orientation_vec = orientation_vec1
    else:
        orientation_vec = orientation_vec2
    orientation_vec = orientation_vec / np.linalg.norm(orientation_vec)
    return orientation_vec


def main():
    q = [2.03844784e-01, 6.46012476e-01, 7.41049869e-01, -5.18757119e-03]
    start = time.clock()
    euler_angles = quaternion_to_euler2(q)
    end = time.clock()
    duration = end - start
    print "time duration: " + str(duration)
    print euler_angles
    q = [2.03844784e-01, 6.46012476e-01, 7.41049869e-01, -5.18757119e-03]
    start1 = time.clock()
    euler_angles1 = quaternion_to_euler1(q)
    end1 = time.clock()
    duration1 = end1 - start1
    print "time duration: " + str(duration1)
    print euler_angles1
#    euler_angles = np.rad2deg(euler_angles)
#    print euler_angles


if __name__ == "__main__":
    main()
