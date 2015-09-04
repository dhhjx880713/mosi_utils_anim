__author__ = 'hadu01'

import glob
from ..animation_data.bvh import BVHReader
from ..animation_data.skeleton import Skeleton
from ..animation_data.motion_editing import get_step_length
from ..animation_data.motion_editing import get_cartesian_coordinates_from_euler_full_skeleton
import numpy as np
import os

class MocapDataStats(object):
    """
    get statistical information e.g. maximum step length, minimum step length, average step lenght... from motion
    capture data
    """

    def __init__(self):
        self.data_folder = None
        self.bvhreaders = []
        self.skeleton = None

    def get_data_folder(self, folder_path):
        """
        Give the folder path of bvh files
        :param folder_path: string, path to motion capture data
        """
        if not folder_path.endswith(os.sep):
            folder_path += os.sep
        self.data_folder = folder_path
        self._load_data()

    def _load_data(self):
        """
        Load bvh files from data folder

        """
        bvh_files = glob.glob(self.data_folder + '*.bvh')
        for item in bvh_files:
            bvh_reader = BVHReader(item)
            self.bvhreaders.append(bvh_reader)
        self.skeleton = Skeleton(bvh_reader)
        # self.calculate_step_len()

    def calculate_step_len(self):
        """
        Calculate step length of files in bvhreaders
        """
        self.mocap_data_stepLen = {}
        for bvhreader in self.bvhreaders:
            step_len = get_step_length(bvhreader.frames)
            self.mocap_data_stepLen[bvhreader.filename] = step_len

    def get_max_step_length(self):
        """
        Calculate the maximum step length in mocap data
        :return max_len: maximum step length in mocap data
        """
        return np.max(self.mocap_data_stepLen.values())

    def get_min_step_length(self):
        """
        Calculate the minimum step length in mocap data
        :return min_len: minimum step length in mocap data
        """
        return np.min(self.mocap_data_stepLen.values())

    def get_average_step_length(self):
        """
        Calculate average step length in mocap data
        :return aver_len: average step length in mocap data
        """
        return np.mean(self.mocap_data_stepLen.values())

    def get_joint_position_for_all_motion(self, joint_name, frame_idx):
        joint_positions = []
        for bvhreader in self.bvhreaders:
            frame = bvhreader.frames[frame_idx]
            pos = get_cartesian_coordinates_from_euler_full_skeleton(bvhreader,
                                                                     self.skeleton,
                                                                     joint_name,
                                                                     frame)
            joint_positions.append(pos)
        return joint_positions