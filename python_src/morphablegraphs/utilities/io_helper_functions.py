# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 12:56:39 2015

@author: Han Du, Martin Manns, Erik Herrmann
"""


import os
import glob
import json
import collections
import math
import numpy as np
from datetime import datetime
from ..animation_data.motion_editing import transform_euler_frames, \
    transform_quaternion_frames
from ..animation_data.bvh import BVHWriter
from ..motion_generator.constraint.splines.parameterized_spline import ParameterizedSpline

def write_to_logfile(path, time_string, data):
    """ Appends json data to a text file.
        Creates the file if it does not exist.
        TODO use logging library instead
    """
    data_string = json.dumps(data, indent=4)
    line = time_string + ": \n" + data_string + "\n-------\n\n"
    if not os.path.isfile(path):
        file_handle = open(path, "wb")
        file_handle.write(line)
        file_handle.close()
    else:
        with open(path, "a") as file_handle:
            file_handle.write(line)


def load_json_file(filename, use_ordered_dict=False):
    """ Load a dictionary from a file

    Parameters
    ----------
    * filename: string
    \tThe path to the saved json file.
    * use_ordered_dict: bool
    \tIf set to True dicts are read as OrderedDicts.
    """
    tmp = None
    with open(filename, 'rb') as infile:
        if use_ordered_dict:
            tmp = json.JSONDecoder(
                object_pairs_hook=collections.OrderedDict).decode(
                infile.read())
        else:
            tmp = json.load(infile)
        infile.close()
    return tmp


def write_to_json_file(filename, serializable, indent=4):
    with open(filename, 'wb') as outfile:
        tmp = json.dumps(serializable, indent=4)
        outfile.write(tmp)
        outfile.close()


def gen_file_paths(directory, mask='*mm.json'):
    """Generator of input file paths

    Parameters
    ----------

     * dir: String
    \tPath of input folder, in which the input files reside
     * mask: String, defaults to '*.bvh'
    \tMask pattern for file search

    """

    if not directory.endswith(os.sep):
        directory += os.sep

    for filepath in glob.glob(directory + mask):
        yield filepath


def get_morphable_model_directory(
        root_directory, morphable_model_type="motion_primitives_quaternion_PCA95"):
    """
    Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    mm_dir = os.sep.join([root_directory,
                          data_dir_name,
                          process_step_dir_name,
                          morphable_model_type])

    return mm_dir


def get_motion_primitive_directory(root_directory, elementary_action):
    """Return motion primitive file path
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    morphable_model_type = "motion_primitives_quaternion_PCA95"
    mm_path = os.sep.join([root_directory,
                           data_dir_name,
                           process_step_dir_name,
                           morphable_model_type,
                           'elementary_action_' + elementary_action
                           ])
    return mm_path


def get_motion_primitive_path(root_directory, elementary_action,
                              motion_primitive):
    """Return motion primitive file
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    morphable_model_type = "motion_primitives_quaternion_PCA95"
    mm_path = os.sep.join([root_directory,
                           data_dir_name,
                           process_step_dir_name,
                           morphable_model_type,
                           'elementary_action_' + elementary_action,
                           '_'.join([elementary_action,
                                     motion_primitive,
                                     'quaternion',
                                     'mm.json'])
                           ])
    return mm_path


def get_transition_model_directory(root_directory):
    data_dir_name = "data"
    process_step_dir_name = "4 - Transition model"
    transition_dir = os.sep.join([root_directory,
                                  data_dir_name,
                                  process_step_dir_name, "output"])
    return transition_dir


def clean_path(path):
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [
            directory for directory in os.getcwd().split(
                os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd) - relative_levels] + [
                                       directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path


def export_euler_frames_to_bvh(
        output_dir, skeleton, euler_frames, prefix="", start_pose=None, time_stamp=True):
    """ Exports a list of euler frames to a bvh file after transforming the frames
    to the start pose.

    Parameters
    ---------
    * output_dir : string
        directory without trailing os.sep
    * skeleton : Skeleton
        contains joint hiearchy information
    * euler_frames : np.ndarray
        Represents the motion
    * start_pose : dict
        Contains entry position and orientation each as a list with three components

    """
    if start_pose is not None:
        euler_frames = transform_euler_frames(
            euler_frames, start_pose["orientation"], start_pose["position"])
    if time_stamp:
        filepath = output_dir + os.sep + prefix + "_" + \
            unicode(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
    elif prefix != "":
        filepath = output_dir + os.sep + prefix + ".bvh"
    else:
        filepath = output_dir + os.sep + "output" + ".bvh"
    print filepath
    BVHWriter(
        filepath,
        skeleton,
        euler_frames,
        skeleton.frame_time,
        is_quaternion=False)


def get_bvh_writer(skeleton, quat_frames, start_pose=None):
    """
    Returns
    -------
    * bvh_writer: BVHWriter
        An instance of the BVHWriter class filled with Euler frames.
    """
    if start_pose is not None:
        quat_frames = transform_quaternion_frames(quat_frames,
                                                  start_pose["orientation"],
                                                  start_pose["position"])

    bvh_writer = BVHWriter(None, skeleton, quat_frames, skeleton.frame_time,
                           is_quaternion=True)
    return bvh_writer


def export_quat_frames_to_bvh_file(output_dir, skeleton, quat_frames, prefix="",
                                   start_pose=None, time_stamp=True):
    """ Exports a list of quat frames to a bvh file after transforming the
    frames to the start pose.

    Parameters
    ---------
    * output_dir : string
        directory without trailing os.sep
    * skeleton : Skeleton
        contains joint hiearchy information
    * quat_frames : np.ndarray
        Represents the motion
    * start_pose : dict
        Contains entry position and orientation each as a list with three components

    """
    bvh_writer = get_bvh_writer(skeleton, quat_frames,
                                start_pose=None)
    if time_stamp:
        filepath = output_dir + os.sep + prefix + "_" + \
            unicode(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
    elif prefix != "":
        filepath = output_dir + os.sep + prefix + ".bvh"
    else:
        filepath = output_dir + os.sep + "output" + ".bvh"
    print filepath
    bvh_writer.write(filepath)

def gen_spline_from_control_points(control_points):
    """

    :param control_points: a list of dictionary,
           each dictionary contains the position and orientation of one point
    :return: Parameterized spline
    """
    tmp = []
    for point in control_points:
        if not math.isnan(sum(np.asarray(point['position']))):
             tmp.append(point['position'])
    dim = len(tmp[0])

    spline = ParameterizedSpline(tmp, dim)
    # print tmp
    return  spline

def load_collision_free_constraints(json_file):
    """
    load control points of collision free path for collided joints
    :param json_file:
    :return: a dictionary {'elementaryActionIndex': {'jointName': spline, ...}}
    """
    try:
        with open(json_file, "rb") as infile:
            json_data = json.load(infile)
    except IOError:
        print('cannot read data from ' + json_file)
    collision_free_constraints = {}
    for action in json_data['modification']:
        collision_free_constraints[action["elementaryActionIndex"]] = {}
        for path in action["trajectories"]:
            collision_free_constraints[action["elementaryActionIndex"]]['jointName'] = path['jointName']
            spline = gen_spline_from_control_points(path['controlPoints'])
            collision_free_constraints[action["elementaryActionIndex"]]['spline'] = spline
    return collision_free_constraints
