__author__ = 'erhe01'
import os
import json
from ..animation_data.bvh import BVHReader
from ..animation_data.motion_vector import MotionVector
from ..animation_data.skeleton import Skeleton

class HandPose(object):
    def __init__(self):
        self.pose_vector = []
        self.indices = []
        self.joint_names = []


class HandPoseGenerator(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.pose_map = dict()
        self.status_change_map = dict()
        self.indices = []
        self.joint_names = []
        self.initialized = False
        return

    def init_generator(self, motion_primitive_directory):
        """
        creates a dicitionary for all possible hand poses
        TODO define in a file
        :param directory_path:
        :return:
        """
        hand_pose_directory = motion_primitive_directory+os.sep+"hand_poses"
        hand_pose_info_file = hand_pose_directory + os.sep + "hand_pose_info.json"
        if os.path.isfile(hand_pose_info_file):
            with open(hand_pose_info_file, "r") as in_file:
                hand_pose_info = json.load(in_file)
                self.status_change_map = hand_pose_info["status_change_map"]
                self.indices = hand_pose_info["indices"]
                self.joint_names = hand_pose_info["joint_names"]

            for root, dirs, files in os.walk(hand_pose_directory):
                for file_name in files:
                    if file_name[-4:] == ".bvh":
                        print file_name[:-4]
                        bvh_reader = BVHReader(root+os.sep+file_name)
                        skeleton = Skeleton(bvh_reader)
                        #motion_vector = MotionVector()
                        #motion_vector.from_bvh_reader(bvh_reader)
                        hand_pose = HandPose()
                        hand_pose.indices = self.indices
                        hand_pose.joint_names = self.joint_names
                        hand_pose.pose_vector = skeleton.reference_frame
                        self.pose_map[file_name[:-4]] = hand_pose
            self.initialized = True
        else:
            print "Error: Could not load hand poses from",hand_pose_directory

    def generate_hand_poses(self, motion_vector, action_list):
        if self.initialized:
            status = "standard"
            for i in xrange(motion_vector.n_frames):
                if i in action_list.keys():
                    status = self.status_change_map[action_list[i][0]["event"]]#assume there is only one event
                    print "change hand status to", status
                self.set_pose_in_frame(status, motion_vector.quat_frames[i])

    def set_pose_in_frame(self, status, pose_vector):
        """
        Overwrites the parameters in the given pose vector with the hand pose
        :param pose_vector:
        :return:
        """
        for i in self.pose_map[status].indices:
            frame_index = i*4 + 3 #translation is ignored
            if frame_index < len(pose_vector):
                pose_vector[frame_index:frame_index+4] = self.pose_map[status].pose_vector[frame_index:frame_index+4]
