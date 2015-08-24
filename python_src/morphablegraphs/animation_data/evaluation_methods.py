# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:39:02 2015

@author: Erik Herrmann
"""

import collections
import numpy as np
from copy import copy

from python_src.morphablegraphs.animation_data.motion_editing import \
    convert_euler_frames_to_cartesian_frames, \
    convert_quaternion_frames_to_euler_frames,\
    euler_substraction


def cartesian_frame_diff(frame_a, frame_b):
    diff = frame_a - frame_b
    return diff


def euler_frame_diff(frame_a, frame_b):
    """
    Calculate the difference between two keyframes where rotation is
    represented by Euler angles
    """
    diff = (frame_a[:3] - frame_b[:3]).tolist()
    diff += [euler_substraction(a, b)
             for a, b in zip(frame_a[3:], frame_b[3:])]
    return diff


def calculate_velocity(euler_frames):
    velocity = []
    for i in xrange(len(euler_frames)):
        if i > 0:
            diff_frame = np.sqrt(
                np.power(
                    (euler_frame_diff(
                        euler_frames[i],
                        euler_frames[
                            i -
                            1])),
                    2))
            velocity.append(diff_frame)
    return velocity


def filter_vectors(frames, node_names):
    """
    Filters out the channels from a list of frames that correspond to joints
    starting with "Bip"
    """
    filtered_vectors = []
    for frame in frames:
        filtered_v = []
        j = 0
        for node_name in node_names.keys():
            # only add non bip frames
            if not node_name.startswith("Bip") and \
                    not node_names[node_name].isEndSite():
                for channel in node_names[node_name].channels:
                    filtered_v.append(frame[j])
                    j += 1
            else:
                  # j must be also increased when a name is skipped
                j += len(node_names[node_name].channels)
            filtered_vectors.append(filtered_v)

    return np.array(filtered_vectors)


def calculate_avg_motion_velocity_from_bvh(bvh_reader,
                                           normalize_over_frames=True):
    """ Extracts euler_frames from a bvh file and calculates  for each channel
        over all keyframes the average velocity or energy as the sum of
        absolute velocity.

        Parameters
        ----------
        * bvh_reader: BVHReader
        \tInitialized BVHReader object containing keyframes
        * normalize_over_frames: Bool
        \tSets whether velocity or energy should be calculated

        Returns
        -------
        * time_function: numpy.ndarray
        \tA data structure containing the energy for each joint and channel
            based on OrderedDicts
    """
    assert len(bvh_reader.keyframes) > 2
    # calculate the average velocity over each frame
    diff_vectors = calculate_velocity(np.array(bvh_reader.keyframes))

    # filter vectors
    diff_vectors = filter_vectors(diff_vectors, bvh_reader.node_names)
    # calculate average/energy, min, max and variance

    if normalize_over_frames:
        avg_vector = np.zeros(diff_vectors[0].shape)
        for vector in diff_vectors:
            avg_vector += vector
        avg_vector /= len(diff_vectors)
    else:
        energy_vector = np.zeros(diff_vectors[0].shape)
        for vector in diff_vectors:
            energy_vector += abs(vector)

    min_vector = np.min(diff_vectors, axis=0)
    max_vector = np.max(diff_vectors, axis=0)
    var_vector = np.var(diff_vectors, axis=0)

    # copy into a OrderedDict data structure representing a pose
    average_velocity = collections.OrderedDict()
    j = 0
    for node_name in bvh_reader.node_names.keys():
        # only add non bip frames
        if not node_name.startswith("Bip") and \
           not bvh_reader.node_names[node_name].isEndSite():

            average_velocity[node_name] = collections.OrderedDict()
            # add one entry for each channel
            for channel in bvh_reader.node_names[node_name].channels:
                average_velocity[node_name][channel] = {"var": var_vector[j],
                                                  "min": min_vector[j],
                                                  "max": max_vector[j]}
                if normalize_over_frames:
                    average_velocity[node_name][channel]["avg"] = avg_vector[j]
                else:
                    average_velocity[node_name][channel]["energy"] = energy_vector[j]

                j += 1

    return average_velocity


def create_filtered_node_name_map(bvh_reader):
    """
    creates dictionary that maps node names to indices in a frame vector
    without "Bip" joints
    """
    node_name_map = collections.OrderedDict()
    j = 0
    for node_name in bvh_reader.node_names:
        if not node_name.startswith("Bip") and \
           not bvh_reader.node_names[node_name].isEndSite():
            node_name_map[node_name] = j
            j += 1

    return node_name_map


def update_bb_value(bounding_box, value):
    """
    Cecks if a value lies inside or outside of a bounding box and then updates
    the bounding box to contain the value if necessary.

    Parameters
    ----------
    * bb: Dict
    \tA dictionary containing the keys min and max
    * value: Float
    \tA float value to check

    Returns
    -------
    True if update was performed and False if not

    """
    update = False
    if bounding_box["min"] > value:
        bounding_box["min"] = copy(value)
        update = True
    if bounding_box["max"] < value:
        bounding_box["max"] = copy(value)
        update = True
    return update


def check_bb_value(bounding_box, value, eps):
    """
    Cecks if a value lies in or outside of a bounding box

    Parameters
    ----------
    * bb: Dict
    \tA dictionary containing the keys min and max
    * value: Float
    \tA float value to check
    * eps: Float
    \tThe allowable threshold for beeing outside of the bounding box

    Returns
    -------
    True if the value lies inside of the bounding box and False if not

    """
    if bounding_box["min"] > value + eps:
        return False
    if bounding_box["max"] < value - eps:
        return False
    return True


def calculate_parameter_bounding_box(bvh_reader):
    """ Extract keyframes from a BVHReader and calculate the pose parameter
        bounding box over a motion
    """

    bounding_box = collections.OrderedDict()  # {}
    for node_name in bvh_reader.node_names.keys():
        # only add non bip frames
        if not node_name.startswith("Bip") and \
           not bvh_reader.node_names[node_name].isEndSite():
            bounding_box[node_name] = collections.OrderedDict()
            # add one entry for each channel
            for c in bvh_reader.node_names[node_name].channels:
                bounding_box[node_name][c] = {"min": np.inf, "max": -np.inf}

    i = 0
    while i < len(bvh_reader.keyframes):

        frame = bvh_reader.keyframes[i]
        j = 0
        for node_name in bvh_reader.node_names.keys():
            if node_name in bounding_box:
                for c in bvh_reader.node_names[node_name].channels:
                    update_bb_value(bounding_box[node_name][c], frame[j])
                    j += 1
            else:
                # j must be also increased when a name is skipped
                j += len(bvh_reader.node_names[node_name].channels)
        i += 1
    return bounding_box


def calculate_cartesian_pose_bounding_box(bvh_reader):
    """ Extract keyframes from a BVHReader and calculate the relative cartesian
        bounding box over a motion
    """

    bounding_box = collections.OrderedDict()  # {}
    for c in ["X", "Y", "Z"]:
        bounding_box[c] = {"min": np.inf, "max": -np.inf}
    # print bb
    # get cartesian frames
    cartesian_frames = convert_euler_frames_to_cartesian_frames(
        bvh_reader,
        bvh_reader.keyframes,
        node_name_map=None)
    # print cartesian_frames.shape
    for frame in cartesian_frames:  # iterate over frames
        j = 0
        for node_name in bvh_reader.node_names.keys():  # iterate over joints
            if not node_name.startswith("Bip") and \
               not bvh_reader.node_names[node_name].isEndSite():
               # print "add",node_name,j,frame.shape
                # iterate over X Y Z for bounding box breach check
                k = 0
                for c in bounding_box.keys():
                    update_bb_value(bounding_box[c], frame[j][k])
                    k += 1
                j += 1
    # print bb
    return bounding_box


def check_parameter_bounding_box(
        bvh_reader,
        euler_frames,
        pose_bb,
        eps=4,
        bip_present=False,
        update_bb=False):
    """
    checks if the parameters in the frames correspond to the pose bounding box
    1) compares each channel in each frame in euler_frames if it goes outside
    of the bounding box
    """
    if bip_present:  # when Bip joints are present in euler frames
        for frame in euler_frames:
            i = 0
            for node_name in bvh_reader.node_names.keys():
                if not node_name.startswith("Bip") and \
                   not bvh_reader.node_names[node_name].isEndSite():
                    # print node_name
                    for c in pose_bb[node_name].keys():
                        if abs(frame[i]) > eps:
                            if not check_bb_value(
                                    pose_bb[node_name][c],
                                    frame[i],
                                    eps):
                                if not update_bb:
                                    print "bounding box", node_name, c, \
                                        pose_bb[node_name], frame[i]
                                    return False
                                else:
                                    update_bb_value(
                                        pose_bb[node_name][c], frame[i])
                        i += 1
                else:
                    i += len(bvh_reader.node_names[node_name].channels)
    else:  # when Bip joints are ignored
        for frame in euler_frames:
            i = 0
            for node_name in pose_bb.keys():

                # print node_name
                for c in pose_bb[node_name].keys():
                    if abs(frame[i]) > eps:
                        if not check_bb_value(
                                pose_bb[node_name][c],
                                frame[i],
                                eps):
                            if not update_bb:
                                print "bounding box", node_name, c, \
                                    pose_bb[node_name], frame[i]
                                return False
                            else:
                                update_bb_value(
                                    pose_bb[node_name][c], frame[i])
                    i += 1

    return True


def check_parameter_bounding_box(
        bvh_reader,
        euler_frames,
        pose_bb,
        eps=4,
        bip_present=False,
        update_bb=False):
    """
    checks if the parameters in the frames correspond to the pose bounding box
    1) compares each channel in each frame in euler_frames if it goes outside
    of the bounding box
    """
    if bip_present:  # when Bip joints are present in euler frames
        for frame in euler_frames:
            i = 0
            for node_name in bvh_reader.node_names.keys():
                if not node_name.startswith("Bip") and \
                   not bvh_reader.node_names[node_name].isEndSite():
                    # print node_name
                    for c in pose_bb[node_name].keys():
                        if abs(frame[i]) > eps:
                            if not check_bb_value(
                                    pose_bb[node_name][c],
                                    frame[i],
                                    eps):
                                if not update_bb:
                                    print "bounding box", node_name, c, \
                                        pose_bb[node_name], frame[i]
                                    return False
                                else:
                                    update_bb_value(
                                        pose_bb[node_name][c], frame[i])
                        i += 1
                else:
                    i += len(bvh_reader.node_names[node_name].channels)
    else:  # when Bip joints are ignored
        for frame in euler_frames:
            i = 0
            for node_name in pose_bb.keys():

                # print node_name
                for c in pose_bb[node_name].keys():
                    if abs(frame[i]) > eps:
                        if not check_bb_value(
                                pose_bb[node_name][c],
                                frame[i],
                                eps):
                            if not update_bb:
                                print "bounding box", node_name, c, \
                                    pose_bb[node_name], frame[i]
                                return False
                            else:
                                update_bb_value(
                                    pose_bb[node_name][c], frame[i])
                    i += 1

    return True


def check_parameter_bounding_box2(
        euler_frames,
        pose_bb,
        eps=4,
        update_bb=False):
    """
    checks if the parameters in the frames correspond to the pose bounding box
    1) compares each channel in each frame in euler_frames if it goes outside
    of the bounding box
    """

    for frame in euler_frames:
        i = 0
        for node_name in pose_bb.keys():

            # print node_name
            for c in pose_bb[node_name].keys():
                if abs(frame[i]) > eps:
                    if not check_bb_value(
                            pose_bb[node_name][c],
                            frame[i],
                            eps):
                        if not update_bb:
                            #                            print "bounding box", node_name, c, \
                            #                                   pose_bb[node_name], frame[i]
                            return False
                        else:
                            update_bb_value(pose_bb[node_name][c], frame[i])
                i += 1

    return True


def check_cartesian_bounding_box(
        bvh_reader,
        euler_frames,
        cartesian_bb,
        eps=4,
        bip_present=False):
    """
    converts euler frames into cartesian frames and checks if the motion
    goes outside the bounding box
    """

    # get cartesian frames
    # print np.array(bvh_reader.keyframes).shape
    if not bip_present:
        node_name_map = create_filtered_node_name_map(bvh_reader)
    else:
        node_name_map = None

    cartesian_frames = convert_euler_frames_to_cartesian_frames(bvh_reader,
                                                                euler_frames,
                                                                node_name_map)

    # print "check",cartesian_frames.shape,bb
    frame_index = 0
    for frame in cartesian_frames:  # iterate over frames
        # print "frame",frame_index
        j = 0
        for node_name in bvh_reader.node_names.keys():  # iterate over joints
            # print "node_name", node_name, bip_present,frame[j][:3]
            if not node_name.startswith("Bip") and \
               not bvh_reader.node_names[node_name].isEndSite():
                # iterate over X Y Z for bounding box breach check
                k = 0
                for c in cartesian_bb.keys():  # X Y Z
                    if not check_bb_value(cartesian_bb[c], frame[j][k], eps):
                        print "cartesian bounding box", node_name, c,\
                            cartesian_bb[c], frame[j][k]
                        return False
                    k += 1
                j += 1
        frame_index += 1
    return True


def check_velocity_value(velocity, value, eps):
    """
    check if value is in a range around the mean
    """
    boundary = abs(velocity["var"]) + eps

    if velocity["avg"] - boundary < value \
            and value < velocity["avg"] + boundary:
        return True
    else:
        return False


def check_average_velocity(
        bvh_reader,
        euler_frames,
        average_velocity,
        eps=4,
        bip_present=False):
    """
    checks if the parameters in the frames correspond to the average velocity
    #1) calculates the average velocity for each joint over all euler_frames
    #2) compares it with the average_velocity vector
    """

    velocity_vectors = np.array(calculate_velocity(euler_frames))

    # filter vectors
    if bip_present:
        velocity_vectors = filter_vectors(
            velocity_vectors, bvh_reader.node_names)

    avg_vector = np.zeros(velocity_vectors[0].shape)
    for frame in velocity_vectors:
        avg_vector += frame

    avg_vector /= len(velocity_vectors)

    # check average velocity
    i = 0
    for node_name in average_velocity.keys():
        for c in average_velocity[node_name].keys():
            success = check_velocity_value(average_velocity[node_name][c],
                                           avg_vector[i], eps)
            if not success:
                print "velocity", node_name, c, \
                    average_velocity[node_name][c], avg_vector[i]
                return False
            i += 1
    return True


def check_sample_validity(
        graph_node,
        sample,
        bvh_reader,
        node_name_map=None,
        eps=4):
    valid = True
    parameter_bb = graph_node.parameter_bb
    if parameter_bb is not None:
        quaternion_frames = graph_node.back_project(
            sample).get_motion_vector().tolist()
        euler_frames = convert_quaternion_frames_to_euler_frames(
            quaternion_frames)
        valid = check_parameter_bounding_box2(
            euler_frames, parameter_bb, eps, update_bb=False)
        # print "valid",valid
    return valid


