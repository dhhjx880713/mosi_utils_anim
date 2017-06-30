# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 10:34:25 2015
@brief: preprocess cutted motion clips for dynamic time warping
@author: du
"""
from ...animation_data.utils import euler_to_quaternion, \
                                     get_cartesian_coordinates_from_euler_full_skeleton, \
                                     transform_euler_frames, \
                                     rotate_euler_frames_about_x_axis, \
                                     rotate_euler_frames, \
                                    point_rotation_by_quaternion, quaternion_to_euler
from ...external.transformations import quaternion_inverse, quaternion_multiply
from ...animation_data.bvh import BVHReader, BVHWriter
from motion_segmentation import MotionSegmentation
from ...animation_data.skeleton import Skeleton
import os
import glob
from copy import deepcopy
import numpy as np


def align_euler_frames(euler_frames,
                       frame_idx,
                       ref_orientation_euler):
    new_euler_frames = deepcopy(euler_frames)
    ref_quat = euler_to_quaternion(ref_orientation_euler)
    root_rot_angles = euler_frames[frame_idx][3:6]
    root_rot_quat = euler_to_quaternion(root_rot_angles)
    quat_diff = quaternion_multiply(ref_quat, quaternion_inverse(root_rot_quat))
    for euler_frame in new_euler_frames:
        root_trans = euler_frame[:3]
        new_root_trans = point_rotation_by_quaternion(root_trans, quat_diff)
        euler_frame[:3] = new_root_trans
        root_rot_angles = euler_frame[3:6]
        root_rot_quat = euler_to_quaternion(root_rot_angles)

        new_root_quat = quaternion_multiply(quat_diff, root_rot_quat)
        new_root_euler = quaternion_to_euler(new_root_quat)
        euler_frame[3:6] = new_root_euler
    return new_euler_frames



class MotionNormalization(MotionSegmentation):

    def __init__(self):
        super(MotionNormalization, self).__init__()
        self.ref_bvh = None
        self.aligned_motions = {}
        self.ref_bvh = None
        self.ref_bvhreader = None

    def load_data_for_normalization(self, data_folder):
        if not data_folder.endswith(os.sep):
            data_folder += os.sep
        bvh_files = glob.glob(data_folder + '*.bvh')
        self.ref_bvh = bvh_files[0]
        self.ref_bvhreader = BVHReader(self.ref_bvh)
        self.skeleton = Skeleton()
        self.skeleton.load_from_bvh(self.ref_bvhreader)
        for bvh_file_path in bvh_files:
            bvhreader = BVHReader(bvh_file_path)
            filename = os.path.split(bvh_file_path)[-1]
            self.aligned_motions[filename] = bvhreader.frames

    def translate_to_original_point(self, frames, origin_point,
                                    height_offset):
        self.ref_bvhreader.frames = frames
        root_pos = self.ref_bvhreader.frames[0][:3]
        rotation = [0, 0, 0]
        translation = np.array([origin_point[0] - root_pos[0],
                                -height_offset,
                                origin_point[2] - root_pos[2]])
        transformed_frames = transform_euler_frames(self.ref_bvhreader.frames,
                                                    rotation,
                                                    translation)
        return transformed_frames

    def set_ref_bvh(self, ref_bvh):
        self.ref_bvh = ref_bvh

    def normalize_root(self, origin_point):
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
        skeleton = Skeleton()
        skeleton.load_from_bvh(self.ref_bvhreader)
        for filename, frames in self.aligned_motions.iteritems():
            height_1 = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
                                                                          skeleton,
                                                                          'Bip01_R_Toe0',
                                                                          frames[0])[1]
            height_2 = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
                                                                          skeleton,
                                                                          'Bip01_L_Toe0',
                                                                          frames[0])[1]
            height_3 = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
                                                                          skeleton,
                                                                          'Bip01_R_Toe0',
                                                                          frames[-1])[1]
            height_4 = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
                                                                          skeleton,
                                                                          'Bip01_L_Toe0',
                                                                          frames[-1])[1]
            height_offset = (height_1 + height_2 + height_3 + height_4)/4.0
            self.aligned_motions[filename] = self.translate_to_original_point(
                frames,
                origin_point,
                height_offset)


    def translate_motion_to_ground(self):
        for filename, frames in self.aligned_motions.iteritems():
            height_1 = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
                                                                          self.skeleton,
                                                                          'Bip01_R_Toe0',
                                                                          frames[0])[1]
            # height_2 = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
            #                                                               self.skeleton,
            #                                                               'Bip01_L_Toe0',
            #                                                               frames[0])[1]
            # height_3 = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
            #                                                               self.skeleton,
            #                                                               'Bip01_R_Toe0',
            #                                                               frames[-1])[1]
            # height_4 = get_cartesian_coordinates_from_euler_full_skeleton(self.ref_bvhreader,
            #                                                               self.skeleton,
            #                                                               'Bip01_L_Toe0',
            #                                                               frames[-1])[1]
            # height_offset = (height_1 + height_2 + height_3 + height_4)/4.0
            self.aligned_motions[filename] = transform_euler_frames(frames,
                                                                    [0, 0, 0],
                                                                    -height_1)

    def align_motion(self, aligned_frame_idx, ref_orientation_euler):
        """calculate the orientation of selected frame, get the rotation angle
           between current orientation and reference orientation, then
           transform frames by rotation angle
        """
        # ref_orientation = [ref_orientation['x'], ref_orientation['z']]
        for filename, frames in self.aligned_motions.iteritems():
            print(filename)
            print(len(frames))
            self.aligned_motions[filename] = align_euler_frames(frames,
                                                                aligned_frame_idx,
                                                                ref_orientation_euler)

    def align_motion_by_vector(self, aligned_frame_idx, ref_orientation):
        """calculate the orientation of selected frame, get the rotation angle
           between current orientation and reference orientation, then
           transform frames by rotation angle
        """
        ref_orientation = [ref_orientation['x'], ref_orientation['z']]
        for filename, frames in self.aligned_motions.iteritems():
            self.aligned_motions[filename] = rotate_euler_frames(frames,
                                                                 aligned_frame_idx,
                                                                 ref_orientation)


    def correct_up_axis(self, frame_idx, ref_up_vector):
        ref_up_vector = [ref_up_vector['y'], ref_up_vector['z']]
        for filename, frames in self.aligned_motions.iteritems():
            self.aligned_motions[filename] = rotate_euler_frames_about_x_axis(frames,
                                                                              frame_idx,
                                                                              ref_up_vector)

    def save_motion(self, save_path):
        if not save_path.endswith(os.sep):
            save_path += os.sep
        for filename, frames in self.aligned_motions.iteritems():
            BVHWriter(save_path + filename, self.skeleton, frames,
                      frame_time=self.ref_bvhreader.frame_time,
                      is_quaternion=False)
