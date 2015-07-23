# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 10:34:25 2015
@brief: preprocess cutted motion clips for dynamic time warping
@author: du
"""
import os
import sys
ROOT_DIR = os.sep.join(['..'] * 2)
sys.path.append(ROOT_DIR)
from utilities.motion_editing import get_rotation_angle, \
                               get_cartesian_coordinates, \
                               transform_euler_frames, \
                               pose_orientation_euler
from utilities.bvh import BVHReader, BVHWriter  
import glob
import numpy as np


class MotionNormalization(object):
    def __init__(self, 
                 motion_data=None, 
                 ref_orientaiton=None,
                 original_point=None,
                 ref_bvh=None,
                 touch_ground_joint=None):
        self.motion_data = motion_data
        self.ref_orientaiton = ref_orientaiton
        self.original_point = original_point
        self.ref_bvh = ref_bvh
        self.touch_ground_joint = touch_ground_joint
    
    def load_data_from_file(self, data_folder):
        self.motion_data = {}
        if not data_folder.endswith(os.sep):
            data_folder += os.sep
        bvh_files = glob.glob(data_folder+'*.bvh')
        self.ref_bvh = bvh_files[0]
        for bvh_file_path in bvh_files:
            bvhreader = BVHReader(bvh_file_path)
            filename = os.path.split(bvh_file_path)[-1]
            self.motion_data[filename] = bvhreader.frames
    
    
    def translate_to_original_point(self, frames):        
        self.ref_bvhreader.frames = frames
        # shift the motion to ground
        touch_point_pos = get_cartesian_coordinates(self.ref_bvhreader,
                                                    self.touch_ground_joint,
                                                    self.ref_bvhreader.frames[0])   
        root_pos = self.ref_bvhreader.frames[0][:3] 
        rotation = [0, 0, 0]
        translation = np.array([self.original_point[0] - root_pos[0],
                                -touch_point_pos[1],
                                self.original_point[2] - root_pos[2]])
        transformed_frames = transform_euler_frames(self.ref_bvhreader.frames,
                                                    rotation,
                                                    translation) 
        return transformed_frames                                            
                                                    
                                                
    def normalize_root(self):
        """set the offset of root joint to (0, 0, 0), and shift the motions to
           original_point, if original_point is None, the set it as (0, 0, 0)
        """

        if self.ref_bvh is not None:
            self.ref_bvhreader = BVHReader(self.ref_bvh)
        else:
            raise ValueError('No reference BVH file for skeleton information')
        self.ref_bvhreader.node_names['Hip']['offset'] = [0, 0, 0]   
        self.translated_motions = {}
        for filename, frames in self.motion_data.iteritems():
            self.translated_motions[filename] = self.translate_to_original_point(frames)
    
    def align_motion(self, aligned_frame_idx):
        """calculate the orientation of selected frame, get the rotation angle
           between current orientation and reference orientation, then 
           transform frames by rotation angle
        """
        self.aligned_motions = {}
        for filename, frames in self.translated_motions.iteritems():
            self.align_motions[filename] = self.rotate_one_motion(frames,
                                                                  aligned_frame_idx,
                                                                  self.ref_orientaiton)   
                                                                  
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
            BVHWriter(save_path+filename, self.ref_bvhreader, frames,
                      frame_time=self.ref_bvhreader.frame_time,
                      is_quaternion=False)                                                         

def test():
    testfile = r'C:\Users\du\MG++\repo\data\1 - MoCap\3 - Cutting\elementary_action_walk\sidestepLeft\walk_001_4_sidestepLeft_425_584.bvh'
    bvhreader = BVHReader(testfile)
    
#    print bvhreader.node_names['Hips']  
#    footposition = get_cartesian_coordinates(bvhreader, 'Bip01_R_Toe0', 
#                                             bvhreader.frames[0])
#    print footposition                                         
#    bvhreader.node_names['Hips']['offset'] = [0, 0, 0]
##    filename = 'normalized_file.bvh'
##    BVHWriter(filename, bvhreader, bvhreader.frames, frame_time=0.013889,
##              is_quaternion = False)
#    footposition = get_cartesian_coordinates(bvhreader, 'Bip01_R_Toe0', 
#                                             bvhreader.frames[0])    
#    print footposition                                               
    
if __name__ == '__main__':
#    input_folder = (r'C:\Users\du\MG++\workspace\MotionGraphs++\mocap data'
#                    r'\walk\sidestepRight\normalized_data')
#    output_folder = (r'C:\Users\du\MG++\workspace\MotionGraphs++\mocap data\\'
#                     r'walk\sidestepRight\alignedOrientation')
#    ref_motion = (r'C:\Users\du\MG++\repo\data\1 - MoCap\4 - Alignment\\'
#                  r'elementary_action_pick\firstTwoHands\\'
#                  r'pick_007_3_firstTwoHands_568_673.bvh')
#    ref_index = 0
#    test_index = 0
#    align_orientation(input_folder,
#                      output_folder,
##                      ref_motion,
##                      ref_index,
#                      test_index)
    test()                     
#    normalization(input_folder, output_folder)                 
