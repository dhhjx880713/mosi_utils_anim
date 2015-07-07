# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:49:59 2015

@author: du
"""
from helper_functions import get_motion_primitive_directory, \
                             get_motion_primitive_path, \
                             clean_path
from motion_primitive import MotionPrimitive
import os 
#from bvh import BVHReader, BVHWriter
from motion_editing import align_quaternion_frames, \
                           convert_quaternion_to_euler, \
                           align_frames, \
                           pose_orientation, \
                           get_orientation_vec, \
                           transform_quaternion_frames, \
                           transform_point
import time
import glob
from bvh2 import create_filtered_node_name_map
from bvh import BVHReader, BVHWriter
import numpy as np
from custom_transformations import vector_distance,\
                                       get_aligning_transformation2,\
                                       create_transformation
from catmull_rom_spline import CatmullRomSpline, plot_spline 
from quaternion_frame import QuaternionFrame    

def get_quaternion_frames(bvhFile,filter_values = True):
    """Returns an animation read from a BVH file as a list of pose parameters, 
	   where the rotation is represented using quaternions.
       
       To enable the calculation of a mean of different animation samples later on, a filter 
       can be activated to align the rotation direction of the quaternions
       to be close to a fixed reference quaternion ([1,1,1,1] has been experimentally selected). 
       In order to handle an error that can occur using this approach,
       the frames are additionally smoothed by comparison to the previous frame
       We assume based on experiments that the singularity does not happen in the first 
       frame of an animation.
       
    Parameters
    ----------

     * bvhFile: String
    \tPath to a bvh file
     * filter_values: Bool
    \tActivates or deactivates filtering
    
    """

    bvh_reader = BVHReader(bvhFile)

    frames = []
    number_of_frames = len(bvh_reader.keyframes)
    last_quat_frame = None
    if filter_values:
        for i in xrange(number_of_frames):
            quat_frame = QuaternionFrame(bvh_reader, i,True)
            if last_quat_frame != None:
                for joint_name in bvh_reader.node_names.keys():
                     if joint_name in quat_frame.keys():
                    
                        #http://physicsforgames.blogspot.de/2010/02/quaternions.html
                        #get dot product to see if they are far away from each other
                        dot = quat_frame[joint_name][0]*last_quat_frame[joint_name][0] + \
                            quat_frame[joint_name][1]*last_quat_frame[joint_name][1] + \
                            quat_frame[joint_name][2]*last_quat_frame[joint_name][2]  + \
                            quat_frame[joint_name][3]*last_quat_frame[joint_name][3]
                    
                        #if they are far away then flip the sign
                        if dot < 0:
                            quat_frame[joint_name] = [-v for v in quat_frame[joint_name]]   
#            for joint_name in quat_frame.keys():
#                tmp = [abs(quat_frame[joint_name][0]), 
#                       abs(quat_frame[joint_name][1]),
#                       abs(quat_frame[joint_name][2]),
#                       abs(quat_frame[joint_name][3])]
#                max_index = max(xrange(len(tmp)), key = tmp.__getitem__)
#                if quat_frame[joint_name][max_index] < 0:
#                    quat_frame[joint_name] = [-v for v in quat_frame[joint_name]]
            last_quat_frame = quat_frame
            root_translation = bvh_reader.keyframes[i][0:3] 
            # rescale root position by a saclar
#            root_translation = [i/50 for i in root_translation]
            frame_values = [root_translation,]+quat_frame.values()
            frames.append(frame_values)
            
    else:
        for i in xrange(number_of_frames):
            quat_frame = QuaternionFrame(bvh_reader, i,True)
            root_translation = bvh_reader.keyframes[i][0:3]
            # rescale root position by a saclar
#            root_translation = [i*50 for i in root_translation]
            frame_values = [root_translation,]+quat_frame.values()
            frames.append(frame_values)
            
    return frames                  

def test_smooth_quatenrion_frames():
    elementary_action = 'pick'
    motion_dir = get_motion_primitive_directory(elementary_action)
    if not motion_dir.endswith(os.sep):
        motion_dir += os.sep
#    print motion_dir
    first_primitive_name = 'firstTwoHands'
    second_primitive_name = 'secondTwoHands'
    feature = 'quaternion'
    first_primitive_path = motion_dir + '%s_%s_%s_mm.json' % (elementary_action,
                                                              first_primitive_name,
                                                              feature)
    second_primitive_path = motion_dir + '%s_%s_%s_mm.json' % (elementary_action,
                                                               second_primitive_name,
                                                               feature)
    bvh_reader = BVHReader('skeleton.bvh')  
    node_name_map = create_filtered_node_name_map(bvh_reader)                                                         
    first_primitive = MotionPrimitive(first_primitive_path)
    second_primitive = MotionPrimitive(second_primitive_path)
    first_motion = first_primitive.sample()
    second_motion = second_primitive.sample()
    first_motion.save_motion_vector('first_motion.bvh')
    second_motion.save_motion_vector('second_motion.bvh')
    start_time = time.clock()
    euler_frames_a = convert_quaternion_to_euler(first_motion.frames.tolist())
    euler_frames_b = convert_quaternion_to_euler(second_motion.frames.tolist())
    new_euler_frames = align_frames(bvh_reader, 
                                    euler_frames_a,
                                    euler_frames_b,
                                    node_name_map,
                                    smooth=True)
    filename = 'smoothed_motion_euler.bvh'                                
    BVHWriter(filename, bvh_reader, new_euler_frames, frame_time=0.013889, is_quaternion=False)                                
    end_time = time.clock()  
    print 'alignment time (euler) is: ' + str(end_time - start_time)                              
    start_time = time.clock()
    new_frames = align_quaternion_frames(bvh_reader,
                                         first_motion.frames,
                                         second_motion.frames,
                                         node_name_map=node_name_map,
                                         smooth=True) 
    end_time = time.clock()
    print "alignment time is " + str(end_time - start_time)                                     
    filename = 'smoothed_motion_quaternion.bvh'                                     
    BVHWriter(filename, bvh_reader, new_frames, frame_time=0.013889, is_quaternion=True) 

def motion_concatenating():
    first_folder = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\4 - Alignment\elementary_action_pick\firstLeftHand'
    second_folder = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\4 - Alignment\elementary_action_pick\secondLeftHand'
    output_folder = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\4 - Alignment\elementary_action_pick\leftHand'    
    first_files = glob.glob(first_folder + os.sep + '*.bvh')
    second_files = glob.glob(second_folder + os.sep + '*.bvh')
    first_dict = {}
    second_dict = {}
    for item in first_files:
        filename = os.path.split(item)[-1]
        segments = filename.split('_')
        bvh_reader = BVHReader(item)
        first_dict[segments[1]] = {'frames': bvh_reader.keyframes,
                                   'filename': '_'.join([segments[0], 
                                                         segments[1],
                                                         segments[2],
                                                         segments[3]]) + '.bvh'}
    for item in second_files:
        filename = os.path.split(item)[-1]
        segments = filename.split('_')        
        bvh_reader = BVHReader(item)
        second_dict[segments[1]] = {'frames': bvh_reader.keyframes,
                                    'filename': '_'.join([segments[0],
                                                          segments[1],
                                                          segments[2],
                                                          segments[3]]) + '.bvh'}                                              
    if len(first_dict) > len(second_dict):
        for key in second_dict.keys():
            new_frames = align_frames(bvh_reader, 
                                      first_dict[key]['frames'],
                                      second_dict[key]['frames'],
                                      smooth=True)
            filename = output_folder + os.sep + second_dict[key]['filename']
            BVHWriter(filename, bvh_reader, new_frames, frame_time=0.013889, is_quaternion=False)
    else:
        for key in first_dict.keys():
            new_frames = align_frames(bvh_reader, 
                                      first_dict[key]['frames'],
                                      second_dict[key]['frames'],
                                      smooth=True)
            filename = output_folder + os.sep + first_dict[key]['filename']
            BVHWriter(filename, bvh_reader, new_frames, frame_time=0.013889, is_quaternion=False) 

def convert_to_sequence(quaternion_frame):
    n_joints = len(quaternion_frame) - 1
    new_frame = np.array(quaternion_frame[0])
    for i in xrange(n_joints):
        new_frame = np.concatenate((new_frame, np.asarray(quaternion_frame[i+1])))
    return new_frame

def test_pose_orientation():
    test_file = r'C:\Users\hadu01\MG++\repo\src\6 - Motion synthesis\output\session_output_230415_174555.bvh'
    bvh_reader = BVHReader(test_file)
    euler_frames = bvh_reader.frames
#    print euler_frames.shape
    quaternion_frames = get_quaternion_frames(test_file)
    frame_index = -1
    quaternion_frame = convert_to_sequence(quaternion_frames[frame_index])
    dir_vec = pose_orientation(quaternion_frame)
#    print 'first frame orientation: '
    print dir_vec
    
    last_transformation = create_transformation(euler_frames[-1][3:6],[0, 0, 0])
    motion_dir = transform_point(last_transformation,[0,0,1])
    motion_dir = np.array([motion_dir[0], motion_dir[2]])
    motion_dir = motion_dir/np.linalg.norm(motion_dir)
    print 'final direction: '
    print motion_dir
#    frame_index = -1
#    quaternion_frame = convert_to_sequence(quaternion_frames[frame_index])
#    dir_vec = pose_orientation(quaternion_frame)
#    print 'last frame orientation: '
#    print dir_vec
#    dir_vec2 = get_orientation_vec(euler_frames)
#    print 'last frame orientation from old approach: '
#    print dir_vec2
#    quaternion_frames = convert_euler_frame_to_cartesian_frame(bvh_reader, euler_frames)
    
def alignment_test():
    # random generate two motions, compare point cloud alignment and pose alignment
    elementary_action = 'pick'
    motion_dir = get_motion_primitive_directory(elementary_action)
    if not motion_dir.endswith(os.sep):
        motion_dir += os.sep
#    print motion_dir
    first_primitive_name = 'firstTwoHands'
    second_primitive_name = 'secondTwoHands'
    feature = 'quaternion'
    first_primitive_path = motion_dir + '%s_%s_%s_mm.json' % (elementary_action,
                                                              first_primitive_name,
                                                              feature)
    second_primitive_path = motion_dir + '%s_%s_%s_mm.json' % (elementary_action,
                                                               second_primitive_name,
                                                               feature) 
    bvh_reader = BVHReader('skeleton.bvh')  
    node_name_map = create_filtered_node_name_map(bvh_reader)                                                         
    first_primitive = MotionPrimitive(first_primitive_path)
    second_primitive = MotionPrimitive(second_primitive_path)
    first_motion = first_primitive.sample()
    second_motion = second_primitive.sample()
#    first_motion.save_motion_vector('first_motion.bvh')
#    second_motion.save_motion_vector('second_motion.bvh')  
    first_motion.get_motion_vector()
    second_motion.get_motion_vector()
    new_frames = align_quaternion_frames(bvh_reader,
                                         first_motion.frames,
                                         second_motion.frames,
                                         node_name_map=node_name_map,
                                         smooth=False) 
    first_dir = pose_orientation(first_motion.frames[-1])
    print first_dir
    second_dir = pose_orientation(second_motion.frames[0])
    print second_dir   
    theta2 = np.rad2deg(np.arccos(np.dot(first_dir, second_dir)))
    print 'rotation angle from pose estimation: '
    print theta2    

def compute_rotation_angle(vec1, vec2): 
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    ref_vec = np.array([1,0])
    tmp1 = np.dot(vec1, ref_vec)/np.linalg.norm(vec1)
    theta1 = np.rad2deg(np.arccos(tmp1))
    if vec1[1] < 0:
        theta1 = - theta1
    tmp2 = np.dot(vec2, ref_vec)/np.linalg.norm(vec2)
    theta2 = np.rad2deg(np.arccos(tmp2)) 
    if vec2[1] < 0:
        theta2 = - theta2
    theta = theta1 - theta2
    return theta        

def test_halt_for_input():
    print 'starting...'
    raw_input('please press enter to continue...')
    print 'ending'   

def test_Catmull_rom_spline():
    control_points = [[0.0, 0, 0.0], 
                      [5.0, 0, 15.0],
                      [0.0, 0, 155.0]]
                                                 
    spline = CatmullRomSpline(control_points, 3)
    plot_spline(spline)

def orientation_check():
    test_file = r'C:\Users\hadu01\MG++\repo\src\6 - Motion synthesis\output\left_pick_and_left_place_test_straight_output_250415_165830.bvh'
    frame_index = 952
    bvh_reader = BVHReader(test_file)
    euler_frames = bvh_reader.frames
    last_transformation = create_transformation(euler_frames[frame_index][3:6],[0, 0, 0])
#    dir_vec = transform_point(np.dot(last_transformation, transformation),[0,0,1])
#    print last_transformation
    dir_vec = np.dot(last_transformation, [0, 0, 1, 0])
    print dir_vec
    dir_vec = np.array([dir_vec[0], dir_vec[2]])
    dir_vec = dir_vec/np.linalg.norm(dir_vec)    
    print dir_vec

def test_alignment_smoothing():
    first_motion = 'walk'
    first_primitive = 'leftStance'
    second_motion = 'walk'
    second_primitive = 'rightStance'
    first_mm_path = get_motion_primitive_path(first_motion, first_primitive)
    first_mm_path = clean_path(first_mm_path)
    second_mm_path = get_motion_primitive_path(second_motion, second_primitive)
    second_mm_path = clean_path(second_mm_path)
    first_mm = MotionPrimitive(first_mm_path)
    second_mm = MotionPrimitive(second_mm_path)
    first_motion = first_mm.sample()
    second_motion = second_mm.sample()
    bvh_reader = BVHReader('skeleton.bvh')
    first_motion.save_motion_vector('first_motion.bvh')
    second_motion.save_motion_vector('second_motion.bvh')
    node_name_map = create_filtered_node_name_map(bvh_reader)
    new_frames = align_quaternion_frames(bvh_reader,
                                         first_motion.frames,
                                         second_motion.frames,
                                         node_name_map)
    filename = 'concatenated_motion.bvh'
    BVHWriter(filename, bvh_reader, new_frames, frame_time=0.013889, is_quaternion=True)  

def test_transform_quaternion_frames():
    test_file = r'C:\Users\hadu01\MG++\repo\data\1 - MoCap\4 - Alignment\elementary_action_walk\leftStance\walk_001_1_leftStance_43_86.bvh'
#    quat_frames = get_quaternion_frames(test_file)
#    new_quat_frames = []
#    for frame in quat_frames:
#        new_frame = convert_to_sequence(frame)
#        new_quat_frames.append(new_frame)
    mm_file = r'C:\Users\hadu01\MG++\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95\elementary_action_pickRight\pickRight_first_quaternion_mm.json'    
    mm = MotionPrimitive(mm_file)
    l_vec = mm.sample(return_lowdimvector=True)
    new_quat_frames = mm.back_project(l_vec, use_time_parameters=True).get_motion_vector()
#    new_quat_frames = motion.get_motion_vector()
    bvh_reader = BVHReader(test_file)
    rotation_angle = 90
    translation_vec = np.array([0, 0, 0])
    rotation_vec = [0, rotation_angle, 0]
    print type(new_quat_frames)
    print new_quat_frames.shape
    transformed_frames = transform_quaternion_frames(new_quat_frames, 
                                                     rotation_vec,
                                                     translation_vec)                                              
    filename = 'rotated_motion.bvh'                                                 
    BVHWriter(filename, bvh_reader, transformed_frames, frame_time=0.013889, 
              is_quaternion=True)   

def test_transform_point():
    test_point = np.array([1, 2, 3])
    angles = [0, 90, 0]
    offset = np.array([0, 20, 0])
    rotated_point = transform_point(test_point, angles, offset)
    print rotated_point                                                  
    
if __name__ == "__main__":
#    test_smooth_quatenrion_frames()
#    motion_concatenating()
#    test_pose_orientation()
#    alignment_test()
#    vec1 = [0, 1]
#    vec2 = [1, 0]
#    theta = compute_rotation_angle(vec1, vec2)
#    print theta
#    rotation_matrix = np.eye(2)
#    rotation_matrix[0,0] = np.cos(np.deg2rad(theta))
#    rotation_matrix[0,1] = - np.sin(np.deg2rad(theta))
#    rotation_matrix[1,0] = np.sin(np.deg2rad(theta))
#    rotation_matrix[1,1] = np.cos(np.deg2rad(theta))
#    rotated_vec = np.dot(rotation_matrix, np.array(vec2))
#    print rotated_vec
#    theta = compute_rotation_angle(vec2, vec1)
#    print theta
#    test_halt_for_input()
#    test_Catmull_rom_spline()
#    orientation_check()
#    test_alignment_smoothing()
    test_transform_quaternion_frames()
#    test_transform_point()