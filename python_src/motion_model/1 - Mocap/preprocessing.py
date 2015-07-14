# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 10:34:25 2015
@brief: preprocess cutted motion clips for dynamic time warping
@author: du
"""
import os
os.chdir(r'../1 - segmentation/')
from lib.motion_editing import pose_orientation, \
                               get_rotation_angle, \
                               transform_quaternion_frames, \
                               get_cartesian_coordinates, \
                               transform_euler_frames
from lib.bvh2 import BVHReader, BVHWriter
from lib.helper_functions import get_quaternion_frames   
import glob
import numpy as np


def normalize_root(filename, touch_point, out_path):
    """set root offset of input file to [0, 0, 0], shift root position
       to origin and touch point is on the ground
    """
    bvhreader = BVHReader(filename)
    bvhreader.node_names['Hip']['offset'] = [0, 0, 0]
    touch_point_pos = get_cartesian_coordinates(bvhreader,
                                                touch_point,
                                                bvhreader.frames[0])
    root_pos = bvhreader.frames[0][:3]
    rotation = [0, 0, 0]
    translation = np.array([-root_pos[0],
                            -touch_point_pos[1],
                            -root_pos[2]])
    transformed_frames = transform_euler_frames(bvhreader.frames,
                                                rotation,
                                                translation)
    if 'bvh' not in out_path:
        outfile_name = out_path + os.path.split(filename)[-1]
    BVHWriter(outfile_name, bvhreader, transformed_frames, frame_time=0.013889,
              is_quaternion=False)


def normalization(input_folder, output_folder):
    """Load bvh files from input folder, shift root position of first frame
       to original point, set root offset to [0, 0, 0], and save the motions
       into output folder
    """
    if not input_folder.endswith(os.sep):
        input_folder += os.sep
    if not output_folder.endswith(os.sep):
        output_folder += os.sep
    touch_point = 'Bip01_R_Toe0'
    for item in input_folder:
        filename = os.path.split(item)[-1]
        out_path = output_folder + filename
        normalize_root(item, touch_point, out_path)
                            
def align_orientation(input_folder,
                      output_folder,
#                      ref_motion,
#                      ref_index,
                      test_index):
    """Align the orientation of bvh files in the input_folder to reference
       motion, and save the results into output folder
       
       Parameters
       ----------
       *input_folder: string
       \tPath of test motions to be aligned
       
       *output_folder: string
       \tSave path for aligned motions
       
       *ref_motion: string
       \tPath of reference motion
       
       *ref_index: int
       \tFrame index of reference frame in reference motion
       
       *test_index: int
       \tFrame index of test motion
    """
    if not input_folder.endswith(os.sep):
        input_folder += os.sep
    if not output_folder.endswith(os.sep):
        output_folder += os.sep
#    reader = BVHReader(ref_motion)
    input_files = glob.glob(input_folder + '*.bvh')
    reader = BVHReader(input_files[0])
#    ref_quat_frames = get_quaternion_frames(ref_motion)
#    ref_ori = pose_orientation(ref_quat_frames[ref_index])
    ref_ori = [0, -1]
    for item in input_files:
        filename = os.path.split(item)[-1]
        quat_frames = get_quaternion_frames(item)
        test_ori = pose_orientation(quat_frames[test_index])
        rot_angle = get_rotation_angle(ref_ori, test_ori)
        translation = np.array([0, 0, 0])
        transformed_frames = transform_quaternion_frames(quat_frames,
                                                         [0, rot_angle, 0],
                                                         translation)
        save_path = output_folder + filename
        BVHWriter(save_path, reader, transformed_frames, frame_time=0.013889,
                  is_quaternion=True)

def test():
    testfile = r'walk_001_4_sidestepLeft_425_584.bvh'
    bvhreader = BVHReader(testfile)
#    print bvhreader.node_names['Hips']  
    footposition = get_cartesian_coordinates(bvhreader, 'Bip01_R_Toe0', 
                                             bvhreader.frames[0])
    print footposition                                         
    bvhreader.node_names['Hips']['offset'] = [0, 0, 0]
#    filename = 'normalized_file.bvh'
#    BVHWriter(filename, bvhreader, bvhreader.frames, frame_time=0.013889,
#              is_quaternion = False)
    footposition = get_cartesian_coordinates(bvhreader, 'Bip01_R_Toe0', 
                                             bvhreader.frames[0])    
    print footposition                                               
    
if __name__ == '__main__':
    input_folder = (r'C:\Users\du\MG++\workspace\MotionGraphs++\mocap data'
                    r'\walk\sidestepRight\normalized_data')
    output_folder = (r'C:\Users\du\MG++\workspace\MotionGraphs++\mocap data\\'
                     r'walk\sidestepRight\alignedOrientation')
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
                     
    normalization(input_folder, output_folder)                 
