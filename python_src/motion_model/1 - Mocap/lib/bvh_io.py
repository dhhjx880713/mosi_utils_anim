# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:34:37 2015

@author: mamauer
"""

#############################################################
# Load functions
#############################################################
from lib.bvh import BVHReader, BVHWriter
from lib.cartesian_frame import CartesianFrame
import os
import glob
import numpy as np

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
        frames.append(frame)
    return frames


def get_euler_frames(bvhFile):
    """Returns an animations read from a BVH file as list of frames.

    Parameters
    ----------

     * bvhFile: String
    \tPath to a bvh file

    """

    bvh_reader = BVHReader(bvhFile)
    return bvh_reader.keyframes


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


def load_animations_as_cartesian_frames(input_dir, mask='*.bvh'):
    """
    Returns a list of animations read from BVH files as a dictionary of lists
    of CartesianFrames.
    Each CartesianFrames instance contains a pose in form of joint positions in
    cartesian space.

    Parameters
    ----------

     * dir: String
    \tPath of input folder, in which the input files reside

    * mask: string
    \tMask pattern for file search, passed to gen_file_path(...)

    """
    data = {}
    for item in gen_file_paths(input_dir, mask=mask):

        filename = os.path.split(item)[-1]
        data[filename] = get_cartesian_frames(item)
    return data


def load_animations_as_euler_frames(input_dir, mask='*.bvh'):
    """
    Returns a list of animations read from BVH files as a dictionary of lists
    of EulerFrames.

    Parameters
    ----------

     * dir: String
    \tPath of input folder, in which the input files reside

    * mask: string
    \tMask pattern for file search, passed to gen_file_path(...)

    """
    data = {}
    for item in gen_file_paths(input_dir, mask=mask):

        filename = os.path.split(item)[-1]
        data[filename] = get_euler_frames(item)
    return data


def euler_frames_to_np_array(frames):
    """ Convert a list of euler_frames into an numpy array

    Parameters
    ----------
    * frames: list of euler_frame
    \tThe data

    Returns
    -------
    * np_frames: numpy.ndarray
    \tThe frames as an numpy.ndarray
    """
    ret_list = []
    for f in frames:
        ret_list.append(np.concatenate(f.values()))
    return np.array(ret_list)


def save_animations_from_euler_frames(data, filename):
    """ Save an animation based on the euler frames

    Parameters
    ----------
    * data: list of euler_frame
    \tThe data

    * filename: string
    \tThe filename to be saved.
    """
    BVHWriter(data, filename, is_quaternion=False)