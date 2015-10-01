__author__ = 'erhe01'
import os
import json
from ..animation_data.bvh import BVHReader
from ..animation_data.motion_vector import MotionVector
from ..animation_data.skeleton import Skeleton


class HandPose(object):
    def __init__(self):
        self.pose_vector = []
        self.hand_skeletons = None


class HandPoseGenerator(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.pose_map = dict()
        self.status_change_map = dict()
        self.left_hand_skeleton = dict()
        self.right_hand_skeleton = dict()
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
                self.right_hand_skeleton = hand_pose_info["right_hand_skeleton"]
                self.left_hand_skeleton = hand_pose_info["left_hand_skeleton"]

            for root, dirs, files in os.walk(hand_pose_directory):
                for file_name in files:
                    if file_name[-4:] == ".bvh":
                        print file_name[:-4]
                        bvh_reader = BVHReader(root+os.sep+file_name)
                        skeleton = Skeleton(bvh_reader)
                        #motion_vector = MotionVector()
                        #motion_vector.from_bvh_reader(bvh_reader)
                        hand_pose = HandPose()
                        hand_pose.hand_skeletons = dict()
                        hand_pose.hand_skeletons["RightHand"] = self.right_hand_skeleton
                        hand_pose.hand_skeletons["LeftHand"] = self.left_hand_skeleton
                        hand_pose.pose_vector = skeleton.reference_frame
                        self.pose_map[file_name[:-4]] = hand_pose
            self.initialized = True
        else:
            print "Error: Could not load hand poses from", hand_pose_directory

    def generate_hand_poses(self, motion_vector, action_list):
        if self.initialized:
            right_status = "standard"
            left_status = "standard"

            for i in xrange(motion_vector.n_frames):
                if i in action_list.keys():
                    for event_desc in action_list[i]:
                        joint_name = event_desc["parameters"]["joint"]
                        if joint_name == "RightHand" or joint_name == "RightToolEndSite":
                            right_status = self.status_change_map[event_desc["event"]]
                            print "change right hand status to", right_status
                        elif joint_name == "LeftHand" or joint_name == "LeftToolEndSite":
                            left_status = self.status_change_map[event_desc["event"]]
                            print "change left hand status to", left_status

                self.set_pose_in_frame("RightHand", right_status, motion_vector.quat_frames[i])
                self.set_pose_in_frame("LeftHand", left_status, motion_vector.quat_frames[i])

    def set_pose_in_frame(self, hand, status, pose_vector):
        """
        Overwrites the parameters in the given pose vector with the hand pose
        :param pose_vector:
        :return:
        """
        for i in self.pose_map[status].hand_skeletons[hand]["indices"]:
            frame_index = i*4 + 3 #translation is ignored
            #print self.pose_map[status].pose_vector[frame_index:frame_index+4]
            pose_vector[frame_index:frame_index+4] = self.pose_map[status].pose_vector[frame_index:frame_index+4]
