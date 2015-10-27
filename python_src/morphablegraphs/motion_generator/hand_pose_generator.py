__author__ = 'erhe01'
import os
import json
from ..animation_data.bvh import BVHReader
from ..animation_data.motion_vector import MotionVector
from ..animation_data.skeleton import Skeleton
from ..external.transformations import quaternion_slerp
import numpy as np


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

    def init_from_desc(self, hand_pose_info):
        self.status_change_map = hand_pose_info["status_change_map"]
        self.right_hand_skeleton = hand_pose_info["right_hand_skeleton"]
        self.left_hand_skeleton = hand_pose_info["left_hand_skeleton"]
        for key in hand_pose_info["skeletonStrings"]:
            print key
            bvh_reader = BVHReader("").init_from_string(hand_pose_info["skeletonStrings"][key])
            skeleton = Skeleton(bvh_reader)
            self._add_hand_pose(key, skeleton)
        self.initialized = True

    def init_generator_from_directory(self, motion_primitive_directory):
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
                        self._add_hand_pose(file_name[:-4], skeleton)
                        #motion_vector = MotionVector()
                        #motion_vector.from_bvh_reader(bvh_reader)

            self.initialized = True
        else:
            print "Error: Could not load hand poses from", hand_pose_directory

    def _add_hand_pose(self, name, skeleton):

         hand_pose = HandPose()
         hand_pose.hand_skeletons = dict()
         hand_pose.hand_skeletons["RightHand"] = self.right_hand_skeleton
         hand_pose.hand_skeletons["LeftHand"] = self.left_hand_skeleton
         hand_pose.pose_vector = skeleton.reference_frame
         self.pose_map[name] = hand_pose

    def _is_affecting_hand(self, hand, event_desc):
        if hand == "RightHand":
            return "RightToolEndSite" in event_desc["parameters"]["joint"] or\
                    "RightHand" in event_desc["parameters"]["joint"] or\
                   "RightToolEndSite" == event_desc["parameters"]["joint"] or\
                   "RightHand" == event_desc["parameters"]["joint"]

        elif hand == "LeftHand":
            return "LeftToolEndSite" in event_desc["parameters"]["joint"] or\
                    "LeftHand" in event_desc["parameters"]["joint"] or\
                   "LeftToolEndSite" == event_desc["parameters"]["joint"] or\
                   "LeftHand" == event_desc["parameters"]["joint"]

    def generate_hand_poses(self, motion_vector, action_list):
        if self.initialized:
            right_status = "standard"
            left_status = "standard"
            left_hand_events = []
            right_hand_events = []
            for i in xrange(motion_vector.n_frames):
                if i in action_list.keys():
                    for event_desc in action_list[i]:
                        if self._is_affecting_hand("RightHand", event_desc):
                            right_status = self.status_change_map[event_desc["event"]]
                            print "change right hand status to", right_status
                            right_hand_events.append(i)
                        if self._is_affecting_hand("LeftHand", event_desc):
                            left_status = self.status_change_map[event_desc["event"]]
                            print "change left hand status to", left_status
                            left_hand_events.append(i)

                self.set_pose_in_frame("RightHand", right_status, motion_vector.quat_frames[i])
                self.set_pose_in_frame("LeftHand", left_status, motion_vector.quat_frames[i])


            #print self.right_hand_skeleton["indices"]
            quat_frames = np.array(motion_vector.quat_frames)
            self.smooth_state_transitions(quat_frames, left_hand_events, self.left_hand_skeleton["indices"])
            self.smooth_state_transitions(quat_frames, right_hand_events, self.right_hand_skeleton["indices"])
            motion_vector.quat_frames = quat_frames.tolist()

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

    def smooth_state_transitions(self, quat_frames, events, indices, window=30):
        #event_frame2 = events[0]
        #joint_index = indices[0]*4+3
        #print "before", quat_frames[event_frame2-window:event_frame2+window, joint_index]

        for event_frame in events:
            for i in indices:
                index = i*4+3
                self.smooth_quaternion_frames_using_slerp(quat_frames, range(index, index+4), event_frame, window)
            #print "handle event", event_frame, quat_frames[event_frame][indices[0]*4+3]
            #smooth_quaternion_frames_partially(quat_frames, indices, event_frame, window)
            #quat_frames = smooth_quaternion_frames(quat_frames, event_frame, window)
            #print "after smoothing", event_frame, quat_frames[event_frame][indices[0]*4+3]
        #print "after", quat_frames[event_frame2-window:event_frame2+window, joint_index]

    def smooth_quaternion_frames_using_slerp(self, quat_frames, joint_parameter_indices, event_frame, window):
        start_frame = event_frame-window/2
        end_frame = event_frame+window/2
        start_q = quat_frames[start_frame, joint_parameter_indices]
        end_q = quat_frames[end_frame, joint_parameter_indices]
        for i in xrange(window):
            t = float(i)/window
            #nlerp_q = self.nlerp(start_q, end_q, t)
            slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
            #print "slerp",start_q,  end_q, t, nlerp_q, slerp_q
            quat_frames[start_frame+i, joint_parameter_indices] = slerp_q

    def nlerp(self, start, end, t):
        """http://physicsforgames.blogspot.de/2010/02/quaternions.html
        """
        dot = start[0]*end[0] + start[1]*end[1] + start[2]*end[2] + start[3]*end[3]
        result = np.array([0.0, 0.0, 0.0, 0.0])
        inv_t = 1.0 - t
        if dot < 0.0:
            temp = []
            temp[0] = -end[0]
            temp[1] = -end[1]
            temp[2] = -end[2]
            temp[3] = -end[3]
            result[0] = inv_t*start[0] + t*temp[0]
            result[1] = inv_t*start[1] + t*temp[1]
            result[2] = inv_t*start[2] + t*temp[2]
            result[3] = inv_t*start[3] + t*temp[3]

        else:
            result[0] = inv_t*start[0] + t*end[0]
            result[1] = inv_t*start[1] + t*end[1]
            result[2] = inv_t*start[2] + t*end[2]
            result[3] = inv_t*start[3] + t*end[3]

        return result/np.linalg.norm(result)
