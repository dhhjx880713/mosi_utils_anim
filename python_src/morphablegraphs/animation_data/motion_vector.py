__author__ = 'erhe01'

from datetime import datetime
import os
import numpy as np
from .utils import align_frames,transform_euler_frames, convert_euler_frames_to_quaternion_frames
from .motion_concatenation import align_and_concatenate_frames, smooth_root_positions#, align_frames_and_fix_feet
from .constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER
from .bvh import BVHWriter
import imp


class MotionVector(object):
    """
    Contains quaternion frames,
    """
    def __init__(self, skeleton=None, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        self.n_frames = 0
        self._prev_n_frames = 0
        self.frames = None
        self.start_pose = None
        self.rotation_type = rotation_type
        self.apply_spatial_smoothing = False
        self.apply_foot_alignment = False
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
            if "apply_foot_alignment" in settings:
                self.apply_foot_alignment = settings["apply_foot_alignment"]

    def from_bvh_reader(self, bvh_reader, filter_joints=True, animated_joints=None):
        if self.rotation_type == ROTATION_TYPE_QUATERNION:
            self.frames = np.array(convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames, filter_joints, animated_joints))
        elif self.rotation_type == ROTATION_TYPE_EULER:
            self.frames = bvh_reader.frames
        self.n_frames = len(self.frames)
        self._prev_n_frames = 0
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

        self._prev_n_frames = self.n_frames
        self.n_frames = len(self.frames)

    def append_frames_using_forward_blending(self, new_frames):

        if self.apply_spatial_smoothing:
            smoothing_window = self.smoothing_window
        else:
            smoothing_window = 0
        from . import motion_concatenation
        imp.reload(motion_concatenation)
        ik_chains = self.skeleton.skeleton_model["ik_chains"]
        self.frames = motion_concatenation.align_frames_using_forward_blending(self.skeleton, self.skeleton.aligning_root_node, new_frames,
                                                                               self.frames, self._prev_n_frames, self.start_pose,
                                                                               ik_chains, smoothing_window)
        self._prev_n_frames = self.n_frames
        self.n_frames = len(self.frames)

    def append_frames(self, new_frames, plant_foot=None):
        if self.apply_foot_alignment and self.skeleton.skeleton_model is not None:
            self.append_frames_using_forward_blending(new_frames)
        else:
            self.append_frames_generic(new_frames)

    def export(self, skeleton, output_filename, add_time_stamp=False):
        bvh_writer = BVHWriter(None, skeleton, self.frames, skeleton.frame_time, True)
        if add_time_stamp:
            output_filename = output_filename + "_" + \
                       str(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
        elif output_filename != "":
            if not output_filename.endswith("bvh"):
                output_filename = output_filename + ".bvh"
        else:
            output_filename = "output.bvh"
        bvh_writer.write(output_filename)

    def reduce_frames(self, n_frames):
        if n_frames == 0:
            self.frames = None
            self.n_frames = 0
            self._prev_n_frames = self.n_frames
        else:
            self.frames = self.frames[:n_frames]
            self.n_frames = len(self.frames)
            self._prev_n_frames = 0

    def has_frames(self):
        return self.frames is not None

    def clear(self, end_frame=0):
        if end_frame == 0:
            self.frames = None
            self.n_frames = 0
            self._prev_n_frames = 0
        else:
            self.frames = self.frames[:end_frame]
            self.n_frames = len(self.frames)
            self._prev_n_frames = 0

    def translate_root(self, offset):
        for idx in range(self.n_frames):
            self.frames[idx][:3] += offset

    def scale_root(self, scale_factor):
        for idx in range(self.n_frames):
            self.frames[idx][:3] *= scale_factor

    def from_fbx(self, animation, animated_joints=None):
        if animated_joints is None:
            animated_joints = list(animation["curves"].keys())
        self.frame_time = animation["frame_time"]
        print("animated joints", animated_joints)
        root_joint = animated_joints[0]
        self.n_frames = len(animation["curves"][root_joint])
        self.frames = []
        for idx in range(self.n_frames):
            frame = self._create_frame_from_fbx(animation, animated_joints, idx)
            self.frames.append(frame)

    def _create_frame_from_fbx(self, animation, animated_joints, idx):
        n_dims = len(animated_joints) * 4 + 3
        frame = np.zeros(n_dims)
        offset = 3
        root_name = animated_joints[0]
        frame[:3] = animation["curves"][root_name][idx]["local_translation"]
        print("root translation", frame[:3])
        for node_name in animated_joints:
            if node_name in list(animation["curves"].keys()):
                rotation = animation["curves"][node_name][idx]["local_rotation"]
                frame[offset:offset+4] = rotation
            else:
                frame[offset:offset+4] = [1, 0, 0, 0]
            offset += 4

        return frame

    def from_custom_unity_format(self, data):
        self.frames = []
        for f in data["frames"]:
            t = f["rootTranslation"]
            new_f = [-t["x"], t["y"], t["z"]]
            for q in f["rotations"]:
                new_f.append(-q["w"])
                new_f.append(-q["x"])
                new_f.append(q["y"])
                new_f.append(q["z"])

            self.frames.append(new_f)
        self.n_frames = len(self.frames)
        self.frame_time = data["frameTime"]

    def apply_low_pass_filter_on_root(self, window):
        self.frames[:, :3] = smooth_root_positions(self.frames[:, :3], window)
