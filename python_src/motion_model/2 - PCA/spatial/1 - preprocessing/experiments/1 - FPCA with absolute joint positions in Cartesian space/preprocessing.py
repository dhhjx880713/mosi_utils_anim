# -*- coding: utf-8 -*-
'''
Created on Nov 18, 2014

@author: hadu01, Martin Manns, Erik Herrmann
'''

import os
import glob
import json
from lib.bvh import BVHReader
from lib.cartesian_frame import CartesianFrame
ROOT_DIR = os.sep.join([".."] * 6)


def get_cartesian_frames(bvhFile):
    """Returns an animations read from a BVH file as a list of cartesian frames.

    Parameters
    ----------

     * bvhFile: String
    \tPath to a bvh file

    """

    bvh_reader = BVHReader(bvhFile)
    frames = []
    number_of_frames = len(bvh_reader.keyframes)
    for frame_number in xrange(number_of_frames):
        frame = CartesianFrame(bvh_reader, frame_number)
        frames.append(frame.values())
    return frames


def load_animations_as_cartesian_frames(input_dir):
    """
    Returns a list of animations read from BVH files as a dictionary of lists
    of CartesianFrames.
    Each CartesianFrames instance contains a pose in form of joint positions in
    cartesian space.

    Parameters
    ----------

     * dir: String
    \tPath of input folder, in which the input files reside

   """

    data = {}
    for item in gen_file_paths(input_dir):

        filename = os.path.split(item)[-1]
        data[filename] = get_cartesian_frames(item)
#     data = np.array(data)
#    print data
    return data


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
    alignment_dir_name = "4 - Alignment"

    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             mocap_dir_name,
                             alignment_dir_name,
                             elementary_action,
                             motion_primitive])

    return input_dir


def get_output_folder():
    """
    Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "1 - preprocessing"
    action = 'experiments'
    test_feature = "1 - FPCA with absolute joint positions in Cartesian space"
    output_dir = os.sep.join([ROOT_DIR,
                              data_dir_name,
                              PCA_dir_name,
                              type_parameter,
                              step,
                              action,
                              test_feature])
    return output_dir


def clean_path(path):
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])

        if ':' not in path:
            cwd = [directory for directory in os.getcwd().split(os.sep)]
        else:
            cwd = []

        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] +
                [directory for directory in path.split(os.sep)
                 if directory != ''][relative_levels:])
    return path


def main():
    input_dir = get_input_data_folder('elementary_action_walk', 'leftStance')
    output_dir = get_output_folder()

    if len(output_dir) > 116:  # avoid a too long path
        output_dir = clean_path(output_dir)

    data = load_animations_as_cartesian_frames(input_dir)
    filename = output_dir + os.sep + 'walk_leftStance_featureVector.json'
    with open(filename, 'wb') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    main()
