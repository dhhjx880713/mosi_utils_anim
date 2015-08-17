# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 10:34:25 2015
@brief: preprocess cutted motion clips for dynamic time warping
@author: du
"""
import os
from ...animation_data.motion_editing import get_rotation_angle, \
                                             get_cartesian_coordinates_from_euler_full_skeleton, \
                                             transform_euler_frames, \
                                             pose_orientation_euler
from ...animation_data.bvh import BVHReader, BVHWriter
from motion_segmentation import MotionSegmentation
from ...animation_data.skeleton import Skeleton
import glob
import numpy as np


class MotionNormalization(MotionSegmentation):

    def __init__(self):
        super(MotionNormalization, self).__init__()
        self.ref_bvh = None

    def load_data_from_files_for_normalization(self, data_folder):
        if not data_folder.endswith(os.sep):
            data_folder += os.sep
        bvh_files = glob.glob(data_folder + '*.bvh')
        self.ref_bvh = bvh_files[0]
        for bvh_file_path in bvh_files:
            bvhreader = BVHReader(bvh_file_path)
            filename = os.path.split(bvh_file_path)[-1]
            self.cutted_motions[filename] = bvhreader.frames

    def translate_to_original_point(self, frames, origin_point,
                                    touch_ground_joint):
        self.ref_bvhreader.frames = frames
        skeleton = Skeleton(self.ref_bvhreader)
        # shift the motion to ground
        touch_point_pos = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
                                                                             skeleton,
                                                                             touch_ground_joint,
                                                                             self.ref_bvhreader.frames[0])
        root_pos = self.ref_bvhreader.frames[0][:3]
        rotation = [0, 0, 0]
        translation = np.array([origin_point[0] - root_pos[0],
                                -touch_point_pos[1],
                                origin_point[2] - root_pos[2]])
        transformed_frames = transform_euler_frames(self.ref_bvhreader.frames,
                                                    rotation,
                                                    translation)
        return transformed_frames

    def set_ref_bvh(self, ref_bvh):
        self.ref_bvh = ref_bvh

    def normalize_root(self, origin_point, touch_ground_joint):
        """set the offset of root joint to (0, 0, 0), and shift the motions to
           original_point, if original_point is None, the set it as (0, 0, 0)
        """
        origin_point = [origin_point['x'],
                        origin_point['y'],
                        origin_point['z']]
        if self.ref_bvh is not None:
            self.ref_bvhreader = BVHReader(self.ref_bvh)
        elif self.bvhreader is not None:
            self.ref_bvhreader = self.bvhreader
        else:
            raise ValueError('No reference BVH file for skeleton information')
        self.ref_bvhreader.node_names['Hips']['offset'] = [0, 0, 0]
        self.translated_motions = {}
        for filename, frames in self.cutted_motions.iteritems():
            self.translated_motions[filename] = self.translate_to_original_point(frames,
                                                                                 origin_point,
                                                                                 touch_ground_joint)

    def align_motion(self, aligned_frame_idx, ref_orientation):
        """calculate the orientation of selected frame, get the rotation angle
           between current orientation and reference orientation, then 
           transform frames by rotation angle
        """
        ref_orientation = [ref_orientation['x'], ref_orientation['z']]
        self.aligned_motions = {}
        for filename, frames in self.translated_motions.iteritems():
            self.aligned_motions[filename] = self.rotate_one_motion(frames,
                                                                    aligned_frame_idx,
                                                                    ref_orientation)

    def rotate_one_motion(self, euler_frames, frame_idx, ref_orientation):
        test_ori = pose_orientation_euler(euler_frames[frame_idx])
        rot_angle = get_rotation_angle(ref_orientation, test_ori)
        translation = np.array([0, 0, 0])
        rotated_frames = transform_euler_frames(euler_frames,
                                                [0, rot_angle, 0],
                                                translation)
        return rotated_frames

    def save_motion(self, save_path):
        if not save_path.endswith(os.sep):
            save_path += os.sep
        for filename, frames in self.aligned_motions.iteritems():
            BVHWriter(save_path + filename, self.ref_bvhreader, frames,
                      frame_time=self.ref_bvhreader.frame_time,
                      is_quaternion=False)


