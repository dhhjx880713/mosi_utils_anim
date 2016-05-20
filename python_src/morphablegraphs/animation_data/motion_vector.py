__author__ = 'erhe01'

import numpy as np
from motion_editing import fast_quat_frames_alignment, align_frames,transform_euler_frames, \
                                          transform_quaternion_frames,\
                                            convert_euler_frames_to_quaternion_frames
from ..utilities.io_helper_functions import export_frames_to_bvh_file
from . import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER


def concatenate_frames(prev_frames, new_frames,start_pose, rotation_type, apply_spatial_smoothing=True, smoothing_window=20):
    if prev_frames is not None:
        if rotation_type == ROTATION_TYPE_QUATERNION:
            return fast_quat_frames_alignment(prev_frames,
                                             new_frames,
                                            apply_spatial_smoothing,
                                            smoothing_window)
        elif rotation_type == ROTATION_TYPE_EULER:
            return align_frames(prev_frames, new_frames)
    elif start_pose is not None:
        if rotation_type == ROTATION_TYPE_QUATERNION:
            return transform_quaternion_frames(new_frames,
                                                      start_pose["orientation"],
                                                      start_pose["position"])
        elif rotation_type == ROTATION_TYPE_EULER:
            return transform_euler_frames(new_frames,
                                          start_pose["orientation"],
                                          start_pose["position"])
    else:
        return new_frames


class MotionVector(object):
    """
    Contains quaternion frames,
    """
    def __init__(self, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
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

    def from_bvh_reader(self, bvh_reader, filter_joints=True):
        if self.rotation_type == ROTATION_TYPE_QUATERNION:
            self.frames = np.array(convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames, filter_joints))
        elif self.rotation_type == ROTATION_TYPE_EULER:
            self.frames = bvh_reader.frames
        self.n_frames = len(self.frames)

    def append_frames(self, new_frames):
        """Align quaternion frames to previous frames

        Parameters
        ----------
        * new_frames: list
            A list of frames with the same rotation format type as the motion vector
        """
        self.frames = concatenate_frames(self.frames, new_frames, self.start_pose, self.rotation_type, self.apply_spatial_smoothing, self.smoothing_window)
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
        #assert n_frames == self.n_frames

    def has_frames(self):
        return self.frames is not None

    def clear(self, end_frame=0):
        if end_frame == 0:
            self.frames = None
        else:
            self.frames = self.frames[:end_frame]
