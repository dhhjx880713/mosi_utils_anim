__author__ = 'erhe01'

import numpy as np
from motion_editing import  align_frames,transform_euler_frames, convert_euler_frames_to_quaternion_frames
from motion_concatenation import align_and_concatenate_frames
from ..utilities.io_helper_functions import export_frames_to_bvh_file
from ..external.transformations import euler_from_quaternion, quaternion_multiply
from . import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER



class MotionVector(object):
    """
    Contains quaternion frames,
    """
    def __init__(self, skeleton=None, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        self.n_frames = 0
        self.frames = None
        self.start_pose = None
        self.rotation_type = rotation_type
        if algorithm_config is not None:
            self.apply_spatial_smoothing = algorithm_config["smoothing_settings"]["spatial_smoothing"]
            self.smoothing_window = algorithm_config["smoothing_settings"]["spatial_smoothing_window"]
        else:
            self.apply_spatial_smoothing = False
            self.smoothing_window = 0
        self.frame_time = 1.0/30.0
        self.skeleton = skeleton

    def from_bvh_reader(self, bvh_reader, filter_joints=True):
        if self.rotation_type == ROTATION_TYPE_QUATERNION:
            self.frames = np.array(convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames, filter_joints))
        elif self.rotation_type == ROTATION_TYPE_EULER:
            self.frames = bvh_reader.frames
        self.n_frames = len(self.frames)
        self.frame_time = bvh_reader.frame_time

    def append_frames(self, new_frames):
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
        self.frames = align_and_concatenate_frames(self.skeleton, self.skeleton.aligning_root_node, new_frames, self.frames, self.start_pose, smoothing_window)
        self.n_frames = len(self.frames)


    def export(self, skeleton, output_dir, output_filename, add_time_stamp=True):
        export_frames_to_bvh_file(output_dir, skeleton, self.frames, prefix=output_filename, time_stamp=add_time_stamp,
                                  is_quaternion=self.rotation_type == ROTATION_TYPE_QUATERNION)

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

