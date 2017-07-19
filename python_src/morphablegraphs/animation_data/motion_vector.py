__author__ = 'erhe01'

from datetime import datetime
import os
import numpy as np
from utils import align_frames,transform_euler_frames, convert_euler_frames_to_quaternion_frames
from motion_concatenation import align_and_concatenate_frames#, align_frames_and_fix_feet
from constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER
from bvh import BVHWriter


class MotionVector(object):
    """
    Contains quaternion frames,
    """
    def __init__(self, skeleton=None, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        self.n_frames = 0
        self.frames = None
        self.start_pose = None
        self.rotation_type = rotation_type
        self.apply_spatial_smoothing = False
        self.smoothing_window = 0
        self.spatial_smoothing_method = "smoothing"
        self.frame_time = 1.0/30.0
        self.skeleton = skeleton

        if algorithm_config is not None:
            settings = algorithm_config["smoothing_settings"]
            self.apply_spatial_smoothing = settings["spatial_smoothing"]
            self.smoothing_window = settings["spatial_smoothing_window"]
            if "spatial_smoothing_method" in settings:
                self.spatial_smoothing_method = settings["spatial_smoothing_method"]

    def from_bvh_reader(self, bvh_reader, filter_joints=True):
        if self.rotation_type == ROTATION_TYPE_QUATERNION:
            self.frames = np.array(convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames, filter_joints))
        elif self.rotation_type == ROTATION_TYPE_EULER:
            self.frames = bvh_reader.frames
        self.n_frames = len(self.frames)
        self.frame_time = bvh_reader.frame_time

    def append_frames_generic(self, new_frames):
        """Align quaternion frames to previous frames

        Parameters
        ----------
        * new_frames: list
            A list of frames with the same rotation format type as the motion vector
        """
        if self.apply_spatial_smoothing:
            smoothing_window = self.smoothing_window
        else:
            smoothing_window = 0
        self.frames = align_and_concatenate_frames(self.skeleton, self.skeleton.aligning_root_node, new_frames, self.frames, self.start_pose,
                                                   smoothing_window=smoothing_window, blending_method=self.spatial_smoothing_method)

        self.n_frames = len(self.frames)

    def append_frames_with_foot_ik(self, new_frames, plant_foot):

        ik_chains = self.skeleton.skeleton_model["ik_chains"]
        if self.apply_spatial_smoothing:
            smoothing_window = self.smoothing_window
        else:
            smoothing_window = 0
        if plant_foot == self.skeleton.skeleton_model["left_foot"]:
            swing_foot = "right"
            plant_foot = "left"

        else:
            swing_foot = "left"
            plant_foot = "right"
        import motion_concatenation
        reload(motion_concatenation)
        self.frames = motion_concatenation.align_frames_and_fix_feet(self.skeleton, self.skeleton.aligning_root_node, new_frames,
                                                self.frames, self.start_pose, plant_foot, swing_foot, ik_chains, 8,
                                                smoothing_window)
        self.n_frames = len(self.frames)

    def append_frames(self, new_frames, plant_foot=None):
        if self.skeleton.skeleton_model is not None:
            ik_chains = self.skeleton.skeleton_model["ik_chains"]
            if plant_foot in ik_chains:
                self.append_frames_with_foot_ik(new_frames, plant_foot)
                return
        self.append_frames_generic(new_frames)

    def export(self, skeleton, output_dir, output_filename, add_time_stamp=True):
        bvh_writer = BVHWriter(None, skeleton, self.frames, skeleton.frame_time, True)
        if add_time_stamp:
            filepath = output_dir + os.sep + output_filename + "_" + \
                       unicode(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
        elif output_filename != "":
            filepath = output_dir + os.sep + output_filename + ".bvh"
        else:
            filepath = output_dir + os.sep + "output" + ".bvh"
        bvh_writer.write(filepath)

    def reduce_frames(self, n_frames):
        if n_frames == 0:
            self.frames = None
            self.n_frames = 0
        else:
            self.frames = self.frames[:n_frames]
            self.n_frames = len(self.frames)

    def has_frames(self):
        return self.frames is not None

    def clear(self, end_frame=0):
        if end_frame == 0:
            self.frames = None
        else:
            self.frames = self.frames[:end_frame]

    def translate_root(self, offset):
        for idx in xrange(self.n_frames):
            self.frames[idx][:3] += offset

    def from_fbx(self, animation, animated_joints=None):
        if animated_joints is None:
            animated_joints = animation["curves"].keys()
        self.frame_time = animation["frame_time"]
        print "animated joints", animated_joints
        root_joint = animated_joints[0]
        self.n_frames = len(animation["curves"][root_joint])
        self.frames = []
        for idx in xrange(self.n_frames):
            frame = self._create_frame_from_fbx(animation, animated_joints, idx)
            self.frames.append(frame)

    def _create_frame_from_fbx(self, animation, animated_joints, idx):
        n_dims = len(animated_joints) * 4 + 3
        frame = np.zeros(n_dims)
        offset = 3
        root_name = animated_joints[0]
        frame[:3] = animation["curves"][root_name][idx]["local_translation"]
        print "root translation", frame[:3]
        for node_name in animated_joints:
            if node_name in animation["curves"].keys():
                rotation = animation["curves"][node_name][idx]["local_rotation"]
                frame[offset:offset+4] = rotation
            else:
                frame[offset:offset+4] = [1, 0, 0, 0]
            offset += 4

        return frame

