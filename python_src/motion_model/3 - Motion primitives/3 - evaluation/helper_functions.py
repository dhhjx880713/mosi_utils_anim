# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 12:56:39 2015

@author: Han Du, Martin Manns, Erik Herrmann
"""


import os
import glob


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


def get_input_data_folder(elementary_action, motion_primitive,sub_dir_name =  "3 - Cutting"):
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


    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             mocap_dir_name,
                             sub_dir_name,
                             elementary_action,
                             motion_primitive])

    return input_dir

def get_output_folder(elementary_action, motion_primitive,morphable_model_type):
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
    process_step_dir_name = "3 - Motion primitives"
    sub_process_step_dir_name = "evaluation"
    output_type_dir_name = "bad_samples"
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              process_step_dir_name,
                              sub_process_step_dir_name,
                              morphable_model_type,
                              output_type_dir_name,
                              elementary_action,
                              motion_primitive])

    if len(output_dir) > 116:  # avoid a too long path
        output_dir = clean_path(output_dir)

    return output_dir

def get_morphable_model_directory(morphable_model_type = "motion_primitives_quaternion_PCA95"):
    """
    Return folder path without trailing os.sep
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    mm_dir = os.sep.join([ROOT_DIR,
                          data_dir_name,
                          process_step_dir_name,
                          morphable_model_type])

    return mm_dir
    
def clean_path(path):
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [directory for directory in os.getcwd().split(os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] + [directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path

