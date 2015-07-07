# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

@author: erhe01,hadu01
"""

import time
import numpy as np 

from lib.motion_editing import convert_quaternion_to_euler, \
                                find_aligning_transformation, \
                                get_cartesian_coordinates2, \
                                transform_euler_frames,\
                                pose_orientation, \
                                align_frames, \
                                calculate_frame_distance,\
                                transform_quaternion_frames,\
                                get_orientation_vec, \
                                align_quaternion_frames,\
                                convert_quaternion_frame_to_cartesian_frame,\
                                get_cartesian_coordinates_from_quaternion2,\
                                get_joint_weights,\
                                align_point_clouds_2D
#                                compute_heading
from lib.custom_transformations import transform_point,\
                                create_transformation
from lib.constraint import obj_error_sum
from scipy.optimize import minimize
from lib.constraint import constraint_distance, find_aligned_quaternion_frames
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from scipy.optimize.optimize import approx_fprime
from lib.bvh2 import BVHReader, BVHWriter,  create_filtered_node_name_map
import os
from lib.motion_primitive import MotionPrimitive
ROOT_DIR = os.sep.join(['..']*2)

    

                                 
def get_aligning_transformation_matrix(bvh_reader,euler_frames_a,euler_frames_b,node_name_map):        
    """
    performs alignment of the point clouds based on the poses at the end of
    euler_frames_a and the start of euler_frames_b
    Returns
    -------
    * transformatrion : np.ndarray
      A 4x4 homogenous transformation matrix aligning the start of euler_frames_b at the end of euler_frames_a 
    """
    #bring motions to same y-coordinate to make 2d alignment possible
    offset_y = euler_frames_a[-1][1] - euler_frames_b[0][1]
    euler_frames_b = transform_euler_frames(euler_frames_b, [0,0,0],[0,offset_y,0])
    
    theta, offset_x, offset_z = find_aligning_transformation(bvh_reader, euler_frames_a,\
                                                euler_frames_b, node_name_map)
    theta = np.degrees(theta)
    transformation = create_transformation([0,theta,0],[offset_x,offset_y,offset_z])
    return transformation

    
def generate_optimization_settings(method="BFGS",max_iterations=100,quality_scale_factor=1,
                                   error_scale_factor=0.1,tolerance=0.05,optimize_theta=False,kinematic_epsilon=5):
    """ Generates optimization_settings dict that needs to be passed to the run_optimization mbvh_readerethod
    """
    
    settings = {"method":method, 
                 "max_iterations"  : max_iterations,
                "quality_scale_factor":quality_scale_factor,
                "error_scale_factor": error_scale_factor,
                "optimize_theta":optimize_theta,
                "tolerance":tolerance,
                "kinematic_epsilon":kinematic_epsilon}
    return settings

def error_function_jac(x0, data):
    """ jacobian of error function. It is a combination of analytic solution 
        for motion primitive model and numerical solution for kinematic error
    """
    motion_primitive, gmm,  constraints, quality_scale_factor, \
    error_scale_factor,  bvh_reader,prev_frames, node_name_map, bounding_boxes,\
    start_transformation,epsilon = data
    tmp = np.reshape(x0, (1, len(x0)))
    logLikelihoods = _log_multivariate_normal_density_full(tmp,
                                                           gmm.means_, 
                                                           gmm.covars_)
    logLikelihoods = np.ravel(logLikelihoods)

    numerator = 0

    n_models = len(gmm.weights_)
    for i in xrange(n_models):
        numerator += np.exp(logLikelihoods[i]) * gmm.weights_[i] * np.dot(np.linalg.inv(gmm.covars_[i]), (x0 - gmm.means_[i]))
#    numerator = numerator
    denominator = np.exp(gmm.score([x0])[0])
#    denominator = motion_primitive.gmm.score([x0])[0]
    logLikelihood_jac = numerator / denominator
    kinematic_jac = approx_fprime(x0, kinematic_error_func, 1e-7, data)
    jac = logLikelihood_jac * quality_scale_factor + kinematic_jac
    return jac

        

def kinematic_error_func(s, data):
    """Compute kinematic error from the constraints
    """    
    motion_primitive, gmm,  constraints, quality_scale_factor, \
    error_scale_factor,  bvh_reader,prev_frames, node_name_map, bounding_boxes,\
    start_transformation,epsilon = data    # prev_frames are quaternion frames 
    s = np.asarray(s) 
    quaternion_frames = motion_primitive.back_project(s).get_motion_vector()
    #euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())
    smoothness_err = 0.0
    if prev_frames != None:
         #find aligning transformation
    
        point_cloud_a = convert_quaternion_frame_to_cartesian_frame(bvh_reader,
                                                               prev_frames[-1],
                                                               node_name_map)
        point_cloud_b = convert_quaternion_frame_to_cartesian_frame(bvh_reader,
                                                               quaternion_frames[0],
                                                               node_name_map)
        weights = get_joint_weights(bvh_reader, node_name_map)
        theta, offset_x, offset_z = align_point_clouds_2D(point_cloud_a,
                                                          point_cloud_b,
                                                          weights)
##        rotation_angle = [0, theta, 0]
##        translation = [offset_x, 0, offset_z]
        transformation = {"orientation": [0, np.rad2deg(theta), 0],
                          "position": np.array([offset_x, 0, offset_z])} 
      
#        transformation = get_aligning_transformation_matrix(bvh_reader, prev_frames,\
#                                            quaternion_frames, node_name_map)
#        frame_distance = calculate_frame_distance(bvh_reader,
#                                                  prev_frames[-1],
#                                                  quaternion_frames[0],
#                                                  node_name_map)
#        smoothness_err += frame_distance   
    else:
        transformation = start_transformation
    quat_frame = None
    constraint_err = 0.0
    orientation_err = 0.0
    last_transformation = create_transformation(quaternion_frames[-1][3:6],[0, 0, 0])
    dir_vec = transform_point(np.dot(last_transformation, transformation),[0,0,1])
    dir_vec = np.array([dir_vec[0], dir_vec[2]])
    dir_vec = dir_vec/np.linalg.norm(dir_vec)
    for constraint in constraints:

        if "dir_vector" in constraint.keys():

            orientation_constraint = np.array([constraint["dir_vector"][0], 
                                               constraint["dir_vector"][2]])

            orientation_err += abs(orientation_constraint[0] - dir_vec[0]) + \
                              abs(orientation_constraint[1] - dir_vec[1])

        if "position" in constraint.keys() or "orientation" in constraint.keys():
            node_name = constraint["joint"]
    
            if constraint["semanticAnnotation"]["lastFrame"]:
                quat_frame = quaternion_frames[-1]
            elif constraint["semanticAnnotation"]["firstFrame"]:
                quat_frame = quaternion_frames[0]
            else:
                quat_frame = quaternion_frames[-1]
                min_dist = np.inf
                min_index = -1
                #heuristic_range = 0.10 * len(euler_frames)
                filtered_frames = quaternion_frames[-1:]
                for index in xrange(len(filtered_frames)):
                    tmp_target_pos = get_cartesian_coordinates_from_quaternion2(bvh_reader,node_name,filtered_frames[index],node_name_map)#get_cartesian_coordinates_from_quaternion(bvh_reader,node_name,quaternion_frames[index],node_name_map)#
                    tmp_target_pos = transform_point(transformation,tmp_target_pos)
    
    #                parameter_index =node_name_map[node_name]*3 +3
    #                tmp_target_rot = euler_frames[index][parameter_index:parameter_index+3]
    #                tmp_target_rot = transform_euler_angles(transformation,tmp_target_rot)
                    tmp_target_rot = None
                    #distance = vector_distance(constraint["position"], tmp_target)
                    distance = constraint_distance(constraint,target_position = tmp_target_pos,target_orientation=tmp_target_rot)
                    if distance  < min_dist:
                        min_dist = distance 
                        min_index = index 
                if min_index >-1:
                    quat_frame = filtered_frames[min_index]
                    
                    
            if quat_frame != None:
    #            euler_frame = euler_frames[-1]
                target_pos = get_cartesian_coordinates2(bvh_reader,node_name,quat_frame,node_name_map) #get_cartesian_coordinates_from_quaternion(bvh_reader,node_name,quaternion_frames[index],node_name_map)
                #print "before",target
                target_pos = transform_point(transformation,target_pos)
                target_rot = None
                distance = sum(constraint_distance(constraint,target_position = target_pos,target_orientation=target_rot))
                if distance > epsilon:
                    constraint_err += distance
                    
    constraint_err = constraint_err * error_scale_factor   + orientation_err 
    return constraint_err
                
def error_func(s,data):
    """ Calculates the error of a low dimensional motion vector s given 
        constraints and the prior knowledge from the statistical model

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains parameters of get_optimal_parameters_from_optimization
        
    Returns
    -------
    the error

    """
    
#    motion_primitive, gmm,  constraints, quality_scale_factor, \
#    error_scale_factor, bvh_reader,prev_frames, node_name_map, bounding_boxes,\
#    start_transformation,epsilon = data  # the pre_frames are quaternion frames
    #kinematic_error = kinematic_error_func(s,data)
    kinematic_error = obj_error_sum(s,data[:-2])
    #print "s-vector",optimize_theta
    error_scale_factor = data[-1]
    quality_scale_factor = data[-2]
    prior_knowledge = -data[0].gmm.score([s,])[0]
    print "naturalness is: " + str(prior_knowledge)
    error = error_scale_factor * kinematic_error + prior_knowledge * quality_scale_factor
    print "error",error
    return error
  



def run_optimization(motion_primitive,gmm,constraints,initial_guess, bvh_reader, node_name_map , 
                       optimization_settings = {},bounding_boxes=None,prev_frames = None,
                       start_pose=None,verbose=False):
    """ Runs the optimization for a single motion primitive and a list of constraints 
    Parameters
    ----------
    * motion_primitive : MotionPrimitive
        Instance of a motion primitive used for back projection
    * gmm : sklearn.mixture.GMM
        Statistical model of the natural motion parameter space
    * constraints : list of dicts
         Each entry containts "joint", "position"   "orientation" and "semanticAnnotation"
    * bvh_reader: BVHReader
        Used for joint hiearchy information 
    * max_iterations : int
        Maximum number of iterations performed by the optimization algorithm
    * bounding_boxes : tuple
        Bounding box data read from a graph node in the morphable graph
    * prev_frames : np.ndarray
        Motion parameters of the previous motion used for alignment
    * start_pose : dict
        Contains keys position and orientation. "position" contains Cartesian coordinates 
        and orientation contains Euler angles in degrees)
        
    Returns
    -------
    * x : np.ndarray
          optimal low dimensional motion parameter vector
    """

    method = optimization_settings["method"] #"BFGS"
    max_iterations = optimization_settings["max_iterations"]
    quality_scale_factor = optimization_settings["quality_scale_factor"]#0.001 #1/100
    error_scale_factor = optimization_settings["error_scale_factor"] #0.01
    tolerance = optimization_settings["tolerance"]#0.05#0.01
#    optimize_theta = optimization_settings["optimize_theta"]    
#    kinematic_epsilon = optimization_settings["kinematic_epsilon"] #0.0# 0.25
#    
#    if start_pose != None:
#        start_transformation = create_transformation(start_pose["orientation"],start_pose["position"])
#    else:
#        start_transformation = np.eye(4)
    
    if initial_guess == None:
        s0 = np.ravel(gmm.sample())#sample initial guess
    else:
        s0 = initial_guess
        
#    if optimize_theta:
#        s0 = np.concatenate( ([0,],s0) )
    #print len(s0)
    # convert prev_frames to euler frames
#    if prev_frames is not None:
#        prev_frames = convert_quaternion_to_euler(prev_frames)
    data = motion_primitive, constraints, prev_frames,start_pose, bvh_reader, node_name_map,{"pos":1,"rot":1,"smooth":1}, \
           error_scale_factor, quality_scale_factor     #precision
#    data = motion_primitive, gmm, constraints, quality_scale_factor, \
#          error_scale_factor,  bvh_reader,prev_frames, node_name_map,bounding_boxes, \
#          start_transformation,kinematic_epsilon

    options = {'maxiter': max_iterations, 'disp' : verbose}
    
    if verbose: 
        start = time.clock()
        print "Start optimization using", method,max_iterations
#    jac = error_function_jac(s0, data)

    result = minimize(error_func,#
                      s0,
                      args = (data,),
                      method=method, 
                      #jac = error_function_jac, 
                      tol =tolerance,
                      options=options)
    if verbose:
        print "Finished optimization in ",time.clock()-start,"seconds"
    return result.x



def error_func_multiple_primitives(s,data):
    """ Calculates the error of a low dimensional motion vector s given constraints
        data can contain a list of motion primitives but only the last one is constrained
        TODO add threshold
    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains parameters of get_optimal_parameters_from_optimization
        
    Returns
    -------
    the error as a float

    """
    motion_primitives, gmm,  constraints, quality_scale_factor, \
    error_scale_factor,  bvh_reader,prev_frames, node_name_map, bounding_boxes,\
    start_transformation,epsilon = data

    err = 0
    prev_param_lengths = 0
    index = 0
    transformation = None
    for mp in motion_primitives:
        #get parameter length from motion_primitive.9 
        param_length = mp.s_pca["n_components"] 
        if mp.has_time_parameters:
            param_length+=mp.t_pca["n_components"]
        s = np.array(s[prev_param_lengths:prev_param_lengths+param_length])
        
        #add naturalness error
        prior_knowledge = -gmm.score([s,])[0]
        print "naturalness is: " + str(prior_knowledge)
        err += prior_knowledge * quality_scale_factor
        
        quaternion_frames = mp.back_project(s).get_motion_vector()
        euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())
        if prev_frames != None: 
             #find aligning transformation
             # note it is always absolute so it does not 
             # need to be multiplied after each step
             transformation = get_aligning_transformation_matrix(bvh_reader, prev_frames,\
                                                euler_frames, node_name_map)
             #transformation = get_delta_root_transformation(prev_frames,euler_frames)
         
        else:
             transformation = start_transformation
        
        prev_param_lengths += param_length
        index += 1

    euler_frame = None
    constraint_err = 0.0
    orientation_err = 0.0
    # compute direction vector from euler frames
    dir_vec = get_orientation_vec(euler_frames)

    for constraint in constraints:

        if "dir_vector" in constraint.keys():
            orientation_constraint = np.array([constraint["dir_vector"][0], 
                                               constraint["dir_vector"][2]])

            orientation_err = abs(orientation_constraint[0] - dir_vec[0]) + \
                              abs(orientation_constraint[2] - dir_vec[1])
            # orientation_err += np.linalg.norm(np.array(dir_vec) - orientation_constraint) ** 2
            print "orientation error is: " + str(orientation_err)
            
        if "position" in constraint.keys() or "orientation" in constraint.keys():
            node_name = constraint["joint"]
    
            if constraint["semanticAnnotation"]["lastFrame"]:
                euler_frame = euler_frames[-1]
            elif constraint["semanticAnnotation"]["firstFrame"]:
                euler_frame = euler_frames[0]
            else:
                euler_frame = euler_frames[-1]
                min_dist = np.inf
                min_index = -1
                heuristic_range = 0.25 * len(euler_frames)
                filtered_frames = euler_frames[-heuristic_range:]
                for index in xrange(len(filtered_frames)):
                    tmp_target_pos = get_cartesian_coordinates2(bvh_reader,node_name,euler_frames[index],node_name_map)#get_cartesian_coordinates_from_quaternion(bvh_reader,node_name,quaternion_frames[index],node_name_map)#
                    tmp_target_pos = transform_point(transformation,tmp_target_pos)
    
    #                parameter_index =node_name_map[node_name]*3 +3
    #                tmp_target_rot = euler_frames[index][parameter_index:parameter_index+3]
    #                tmp_target_rot = transform_euler_angles(transformation,tmp_target_rot)
                    tmp_target_rot = None
                    #distance = vector_distance(constraint["position"], tmp_target)
                    distance = constraint_distance(constraint,target_position = tmp_target_pos,target_orientation=tmp_target_rot)
                    if distance  < min_dist:
                        min_dist = distance 
                        min_index = index 
                if min_index >-1:
                    euler_frame = euler_frames[min_index]
                    
                    
            if euler_frame != None:
    #            euler_frame = euler_frames[-1]
                target_pos = get_cartesian_coordinates2(bvh_reader,node_name,euler_frame,node_name_map) #get_cartesian_coordinates_from_quaternion(bvh_reader,node_name,quaternion_frames[index],node_name_map)
                #print "before",target
                target_pos = transform_point(transformation,target_pos)
    #            parameter_index =node_name_map[node_name]*3 +3
    #            target_rot = euler_frame[parameter_index:parameter_index+3]
    #            target_rot =  transform_euler_angles(transformation,target_rot)
                #print "after",target
                target_rot = None
                    
                #distance = vector_distance(constraint["position"],target)
                #print target,constraint["position"]
                distance = sum(constraint_distance(constraint,target_position = target_pos,target_orientation=target_rot))
                if distance > epsilon:
                    constraint_err += distance
            
    err += constraint_err * error_scale_factor + orientation_err * 10
    print "error",err
    return err

def run_optimization_multiple_primitives(motion_primitives,gmm,constraints,initial_guess, bvh_reader, node_name_map , 
                       optimization_settings = {},bounding_boxes=None,prev_frames = None,
                       start_pose=None,verbose=False):
    """ Runs the optimization for a single motion primitive and a list of constraints 
    Parameters
    ----------
    * motion_primitive : MotionPrimitive
        Instance of a motion primitive used for back projection
    * gmm : sklearn.mixture.GMM
        Statistical model of the natural motion parameter space
    * constraints : list of dicts
         Each entry containts "joint", "position"   "orientation" and "semanticAnnotation"
    * bvh_reader: BVHReader
        Used for joint hiearchy information 
    * max_iterations : int
        Maximum number of iterations performed by the optimization algorithm
    * bounding_boxes : tuple
        Bounding box data read from a graph node in the morphable graph
    * prev_frames : np.ndarray
        Motion parameters of the previous motion used for alignment
    * start_pose : dict
        Contains keys position and orientation. "position" contains Cartesian coordinates 
        and orientation contains Euler angles in degrees)
        
    Returns
    -------
    * x : np.ndarray
          optimal low dimensional motion parameter vector
    """
    
    method = optimization_settings["method"] #"BFGS"
    max_iterations = optimization_settings["max_iterations"]
    quality_scale_factor = optimization_settings["quality_scale_factor"]#0.001 #1/100
    error_scale_factor = optimization_settings["error_scale_factor"] #0.01
    tolerance = optimization_settings["tolerance"]#0.05#0.01
    optimize_theta = optimization_settings["optimize_theta"]    
    kinematic_epsilon = optimization_settings["kinematic_epsilon"] #0.0# 0.25
    
    if start_pose != None:
        start_transformation = create_transformation(start_pose["orientation"],start_pose["position"])
    else:
        start_transformation = np.eye(4)
    
    if initial_guess == None:
        s0 = np.ravel(gmm.sample())#sample initial guess
    else:
        s0 = initial_guess
        
    if optimize_theta:
        s0 = np.concatenate( ([0,],s0) )
    #print len(s0)
   
    data = motion_primitives, gmm, constraints, quality_scale_factor, \
          error_scale_factor,  bvh_reader,prev_frames, node_name_map,bounding_boxes, \
          start_transformation,kinematic_epsilon

    options = {'maxiter': max_iterations, 'disp' : verbose}
    
    if verbose: 
        start = time.clock()
        print "Start optimization using", method
#    jac = error_function_jac(s0, data)

    result = minimize(error_func_multiple_primitives,
                      s0,
                      args = (data,),
                      method=method, 
                      jac = error_function_jac, 
                      tol =tolerance,
                      options=options)
    if verbose:
        print "Finished optimization in ",time.clock()-start,"seconds"
    return result.x
    
    
    
    
    
    
    
    
    
    
def get_motion_primitive_folder(elementary_action):       
    '''
    Return folder path without trailing os.sep
    '''
    data_dir_name = 'data'
    motion_primitives_dir = '3 - Motion primitives'
    model_type = 'motion_primitives_quaternion_PCA95'
    elementary_motion_type = 'elementary_action_' + elementary_action
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             motion_primitives_dir,
                             model_type,
                             elementary_motion_type])
    return input_dir  
    
def test_run_optimization():
    elementary_action = 'walk'
    primitive_type = 'leftStance'
    motion_primitive_folder = get_motion_primitive_folder(elementary_action)
    feature = 'quaternion'
    filename = '%s_%s_%s_mm.json' % (elementary_action,
                                     primitive_type,
                                     feature)
    filepath = motion_primitive_folder + os.sep + filename
    motion_primitive = MotionPrimitive(filepath)
    gmm = motion_primitive.gmm
    bvh_reader = BVHReader(os.sep.join(['lib', 'skeleton.bvh']))
    node_name_map = create_filtered_node_name_map(bvh_reader)
    initial_guess = motion_primitive.sample(return_lowdimvector=True)
#    constraints = [{"joint": "LeftHand", 
#                    "position": [-50, 80, -45],
#                    "semanticAnnotation": {"firstFrame": True, "lastFrame": None}},
#                   {"joint": "RightHand",
#                    "position": [50, 80, -45],
#                    "semanticAnnotation": {"firstFrame": True, "lastFrame": None}}]
    constraints = [{"joint": "Hips",
                    "position": [-100, None, -155],
                    "semanticAnnotation": {"firstFrame": False, "lastFrame": True}}]
                    
    start = time.clock()                
    optimized_parameters = run_optimization(motion_primitive,
                           gmm,
                           constraints,
                           initial_guess, 
                           bvh_reader, 
                           node_name_map , 
                           optimization_settings = generate_optimization_settings(error_scale_factor=1),
                           bounding_boxes=None,
                           prev_frames = None,
                           start_pose=None,
                           verbose=False)
    end = time.clock()
    print "optimization is finished in: " + str(end-start)                      
    synthesized_motion = motion_primitive.back_project(optimized_parameters) 
#    synthesized_motion.get_motion_vector()
#    for frame in synthesized_motion.frames:
#        frame[2] = frame[2] - 155
    synthesized_motion.save_motion_vector('synthesized_motion.bvh')                      
#    filename = 'synthesized_motion.bvh'
#    BVHWriter(filename, bvh_reader, synthesized_motion.frames, frame_time=0.013889,
#              is_quaternion=True)  
             
def test_orientation_constraint():
    elementary_action = 'carryRight'
    primitive_type = 'endRightStance'
    motion_primitive_folder = get_motion_primitive_folder(elementary_action)
    feature = 'quaternion'
    filename = '%s_%s_%s_mm.json' % (elementary_action,
                                     primitive_type,
                                     feature)
    filepath = motion_primitive_folder + os.sep + filename
    motion_primitive = MotionPrimitive(filepath) 
    # get orientation constraint from an example
    constraint_file_folder = r'../5 - Elementary action breakdown/controlled_walk/endLeftStance'   
    constraint_file = constraint_file_folder + os.sep + 'constraint_motion4.bvh'
    bvh_reader = BVHReader(constraint_file) 
#    constraint_dir_vec = get_orientation_vec(bvh_reader.keyframes) 
    constraint_dir_vec = np.asarray([0, -1])
#    print 'orientation constraint is: '
#    print constraint_dir_vec
#    euler_frames = bvh_reader.keyframes
    # print all root position 2d
    
    node_name_map = create_filtered_node_name_map(bvh_reader)
#    motion_sample = motion_primitive.sample()
#    motion_sample.get_motion_vector()
##    quat_frame = motion_sample.frames[-1]
#    euler_frames = convert_quaternion_to_euler(motion_sample.frames.tolist())
#    orientation = get_orientation_vec(euler_frames)
    # generate a random sample motion, and check the initial orientaion of the motion 
    
#    new_sample = motion_primitive.sample()
#    new_sample.get_motion_vector()
#    euler_frames = new_sample.frames
#    orientation = get_orientation_vec(euler_frames)
#    print "orientation: "
#    print orientation
#    new_sample.save_motion_vector('random_sample_walking.bvh')
    
#    motion_sample.save_motion_vector('orientation_constraint_0.bvh')
    
    # get trajectory direction from last 5 root points
#    points_2d = np.array([[euler_frames[-5][0], euler_frames[-5][2]],
#                          [euler_frames[-4][0], euler_frames[-4][2]],
#                          [euler_frames[-3][0], euler_frames[-3][2]],
#                          [euler_frames[-2][0], euler_frames[-2][2]],
#                          [euler_frames[-1][0], euler_frames[-1][2]]])
#    trajectory_dir = get_trajectory_dir_from_2d_points(points_2d)
#    print "trajectory direction: ", trajectory_dir
#    heading_vec = compute_heading(euler_frames[-1],
#                                  trajectory_dir, 
#                                  bvh_reader, 
#                                  node_name_map)
#    print "heading: ", heading_vec
    # rotate orientation constraint vector
    rotation_angle = 0
    rad = np.deg2rad(rotation_angle)
    rot_mat = np.eye(2)
    rot_mat[0,0] = np.cos(rad)
    rot_mat[0,1] = -np.sin(rad)
    rot_mat[1,0] = np.sin(rad)
    rot_mat[1,1] = np.cos(rad)
    rotated_vec = np.dot(rot_mat, constraint_dir_vec)
    print rotated_vec
    constraints = [{"dir_vector": [rotated_vec[0], 0, rotated_vec[1]]}]
    initial_guess = motion_primitive.sample(return_lowdimvector=True)
    gmm = motion_primitive.gmm

    optimized_parameters = run_optimization(motion_primitive,
                                            gmm,
                                            constraints,
                                            initial_guess, 
                                            bvh_reader, 
                                            node_name_map , 
                                            optimization_settings = generate_optimization_settings(max_iterations= 25),
                                            bounding_boxes=None,
                                            prev_frames = None,
                                            start_pose=None,
                                            verbose=True)   
    synthesized_motion = motion_primitive.back_project(optimized_parameters) 
    synthesized_motion.save_motion_vector('orientation_constraint_0.bvh')

def test_frame_constraint():
    first_elementary_action = 'carry'
    first_primitive_type = 'leftStance'
    second_elementary_action = 'carry'
    second_primitive_type = 'rightStance'
    motion_primitive_folder1 = get_motion_primitive_folder(first_elementary_action)
    motion_primitive_folder2 = get_motion_primitive_folder(second_elementary_action)
    feature = 'quaternion'
    filename1 = '%s_%s_%s_mm.json' % (first_elementary_action,
                                      first_primitive_type,
                                      feature)
    filename2 = '%s_%s_%s_mm.json' % (second_elementary_action,
                                      second_primitive_type,
                                      feature)                                  
    filepath1 = motion_primitive_folder1 + os.sep + filename1
    first_motion_primitive = MotionPrimitive(filepath1)     
    filepath2 = motion_primitive_folder2 + os.sep + filename2
    second_motion_primitive = MotionPrimitive(filepath2)    
    bvh_reader = BVHReader(os.sep.join(['lib', 'skeleton.bvh']))
    node_name_map = create_filtered_node_name_map(bvh_reader)
    """
    Step 1: generate a random sample from first motion primitve as input
    Step 2: generate 100 sample from next motion primitive, and find the closest
            one to the input
    Step 3: concatenate two motion samples by alignment and smoothing
    """
#    print first_motion_primitive.name
    current_motion_primitive = first_motion_primitive                                    
    input_sample = current_motion_primitive.sample()
    input_sample.get_motion_vector()
    input_euler_frames = convert_quaternion_to_euler(input_sample.frames.tolist())    
    N = 100
    Steps = 20
    current_step = 'left'
    threshold = 1.5
    for i in xrange(Steps):
        if current_step == 'left':
            next_motion_primitive = second_motion_primitive
            current_step = 'right'
        else:
            next_motion_primitive = first_motion_primitive
            current_step = 'left'
        dists = []
        output_samples = []
        dist = 5    
        max_bad_samples = 300
        counter = 0
        while dist > threshold and counter < max_bad_samples:
            output_sample = next_motion_primitive.sample()
            output_sample.get_motion_vector()
            output_samples.append(output_sample)
            output_euler_frames = convert_quaternion_to_euler(output_sample.frames.tolist())
            dist = calculate_frame_distance(bvh_reader,
                                                input_euler_frames[-1],
                                                output_euler_frames[0],
                                                node_name_map)
            counter += 1                                    
#            print dist                                    
            dists.append(dist)
        print "step_" + str(i)
        if counter == max_bad_samples:
            min_index = min(xrange(len(dists)), key = dists.__getitem__)
            print 'minimal distance: ', dists[min_index]                         
            output_sample = output_samples[min_index] 
            
        optimal_output_euler_frames = convert_quaternion_to_euler(output_sample.frames.tolist())
        input_euler_frames = align_frames(bvh_reader, 
                                         input_euler_frames,
                                         optimal_output_euler_frames,
                                         node_name_map)
                                   
    filename = 'synthesized_motion.bvh'
    BVHWriter(filename, bvh_reader, input_euler_frames, frame_time=0.013889,
              is_quaternion=False)   
                                          
if __name__ == '__main__':
#    test_run_optimization()        
    test_orientation_constraint()
#    test_frame_constraint()
