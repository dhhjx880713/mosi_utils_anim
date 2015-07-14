# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:56:45 2015

@author: herrmann
"""

import os
import glob
from lib.quaternion_frame import *
from lib.bvh import *
from helper_functions import *
from mirror_animation import *
from mirror_animation_list import *

ROOT_DIR = os.sep.join([".."] * 3)


def gen_file_paths(dir, mask='*.bvh'):
    """Generator of input file paths

    Parameters
    ----------

     * dir: String
    \tPath of input folder, in which the input files reside
     * mask: String, defaults to '*.bvh'
    \tMask pattern for file search

    """

    if not dir.endswith(os.sep):
        dir += os.sep

    for filepath in glob.glob(dir + mask):
        yield filepath


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
    alignment_dir_name = "3 - Cutting"

    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             mocap_dir_name,
                             alignment_dir_name,
                             elementary_action,
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
    MoCap_dir_name = "1 - MoCap"
    process_step_dir_name = "5 - Mirrored Data"
    input_type_dir_name = "Cutting"
    #motion_type_dir_name = "elementary_action_walk"
    #motion_primitive_dir_name = "leftStance"
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              MoCap_dir_name,
                              process_step_dir_name,
                              input_type_dir_name,
                              elementary_action,
                              motion_primitive])
    return output_dir


def clean_path(path):
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [directory for directory in os.getcwd().split(os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] + [directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path

def get_mirror_map():
    mirror_map ={ "LeftShoulder":"RightShoulder",#
                    "LeftArm":"RightArm",
                   "LeftForeArm":"RightForeArm",
                   "LeftHand": "RightHand",
                   "LeftUpLeg":"RightUpLeg",
                   "LeftLeg": "RightLeg",
                   "LeftFoot":"RightFoot"
    }    
    for k in mirror_map.keys():
        mirror_map[mirror_map[k]] = k
    return mirror_map

if __name__ =="__main__":
    
    elementary_action = 'elementary_action_walk'
    motion_primitive = 'leftStance'
    input_dir = get_input_data_folder(elementary_action, motion_primitive)
    output_dir = get_output_folder(elementary_action,motion_primitive)
    
    if len(output_dir) > 116:  # avoid a too long path
        output_dir = clean_path(output_dir)
    #output_dir = "mirrored"
    mirror_map = get_mirror_map()
    paths = gen_file_paths(input_dir)
    print "start",input_dir
    for item in gen_file_paths(input_dir):
        filename = os.path.split(item)[-1][:-4]+"_mirrored.bvh"
        print filename
        bvh_reader = BVHReader(item)
        frames = get_quaternion_frames(item)
        frames = get_frame_vectors_from_quat_animations(frames)
        #print frames
        new_frames = mirror_animation(bvh_reader.node_names,frames,mirror_map)
        out_path = output_dir + os.sep +filename
        BVHWriter(out_path,bvh_reader,new_frames,frame_time= bvh_reader.frame_time,\
                                            is_quaternion = True)