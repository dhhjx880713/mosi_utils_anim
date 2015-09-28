__author__ = 'erhe01'
import os
import json
from ..animation_data.bvh import BVHReader
from ..animation_data.motion_vector import MotionVector


class HandPose(object):
    def __init__(self):
        self.pose_vector = []
        self.indices = []
        self.joint_names = []

    def set_in_frame(self, pose_vector):
        """
        Overwrites the parameters in the given pose vector with the hand pose
        :param pose_vector:
        :return:
        """
        for i in self.indices:
            pose_vector[i] = self.pose_vector[i]


class HandPoseGenerator(object):
    def __init__(self):
        self.pose_map = dict()
        self.status_change_map = dict()
        self.indices = []
        self.joint_names = []
        return

    def init_generator(self, hand_pose_directory):
        """
        creates an index for all possible status changes
        TODO define in a file
        :param directory_path:
        :return:
        """
        hand_pose_info_file = hand_pose_directory + os.sep + "hand_pose_info.json"
        if os.path.isfile(hand_pose_info_file):
            with open(hand_pose_info_file, "r") as in_file:
                hand_pose_info = json.load(in_file)
                self.status_change_map = hand_pose_info["status_change_map"]
                self.indices = hand_pose_info["indices"]
                self.joint_names = hand_pose_info["joint_names"]

        for root, dirs, files in os.walk(hand_pose_directory):
            for file_name in files:
                if file_name[:4] == ".bvh":
                    bvh_reader = BVHReader(root+os.sep+file_name)
                    motion_vector = MotionVector()
                    motion_vector.from_bvh_reader(bvh_reader)
                    hand_pose = HandPose()
                    hand_pose.indices = self.indices
                    hand_pose.joint_names = self.joint_names
                    hand_pose.pose_vector = motion_vector.quat_frames[0]
                    self.pose_map[file_name[:4]] = hand_pose

    def generate_hand_poses(self, motion_vector, action_list):
        status = "standard"
        for i in xrange(motion_vector.n_frames):
            index_str = str(i)
            if index_str in action_list.keys():
                status = self.status_change_map[action_list[index_str][0]["event"]]#assume there is only one event
            self.pose_map[status].set_in_frame(motion_vector[i])
        return motion_vector

