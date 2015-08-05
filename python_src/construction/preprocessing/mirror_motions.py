# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:24:42 2015
@Brief: mirror the motion data in the source folder to target folder, change 
the file name correspondingly 
@author: hadu01
"""
from mirror_animation import mirror_animation
from mirror_animation_list import get_mirror_map, \
    gen_file_paths, \
    clean_path
import os
from helper_functions import *
from lib.bvh import BVHReader, BVHWriter
ROOT_DIR = os.sep.join([".."] * 3)


def get_input_data_folder(elementary_action, motion_primitive):
    """Returns folder path as string without trailing os.sep

    Parameters
    ----------

     * elementary_action: String
    \tElementary action of the motion primitive
     * motion_primitive: String
    \tMotion primitive for which the folder shall be returned

    """

    data_dir_name = "data"
    mocap_dir_name = "1 - MoCap"
#    alignment_dir_name = "4 - Alignment"
    cutted_dir_name = "3 - Cutting"

    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             mocap_dir_name,
                             #                             alignment_dir_name,
                             cutted_dir_name,
                             'elementary_action_' + elementary_action,
                             motion_primitive])

    return input_dir


def get_output_folder(elementary_action, motion_primitive):
    """
    Return folder path to store result without trailing os.sep


    Parameters
    ----------

     * elementary_action: String
    \tElementary action of the motion primitive
     * motion_primitive: String
    \tMotion primitive for which the folder shall be returned
    """
    data_dir_name = "data"
    mocap_dir_name = "1 - MoCap"
#    alignment_dir_name = "4 - Alignment"
    cutted_dir_name = "3 - Cutting"

    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              mocap_dir_name,
                              #                             alignment_dir_name,
                              cutted_dir_name,
                              'elementary_action_' + elementary_action,
                              motion_primitive])
    return output_dir


def main():
    elementary_action = 'walk'
    input_primitive = 'sidestepLeft'
    output_primitive = 'sidestepRight'
    input_dir = get_input_data_folder(elementary_action, input_primitive)
    output_dir = get_output_folder(elementary_action, output_primitive)
    mirror_map = get_mirror_map()
    for path in gen_file_paths(input_dir):
        # generate output filename
        filename = os.path.split(path)[-1][:-4]
        segments = filename.split('_')
        if 'mirrored' not in segments:
            segments[3] = output_primitive
            output_filename = '_'.join(segments) + '_mirrored.bvh'
            if len(path) > 116:
                path = clean_path(path)
            bvh_reader = BVHReader(path)
            frames = get_quaternion_frames(path)
            frames = get_frame_vectors_from_quat_animations(frames)
            new_frames = mirror_animation(bvh_reader.node_names,
                                          frames,
                                          mirror_map)
            output_path = output_dir + os.sep + output_filename
            BVHWriter(output_path, bvh_reader, new_frames,
                      frame_time=bvh_reader.frame_time, is_quaternion=True)
if __name__ == '__main__':
    main()
