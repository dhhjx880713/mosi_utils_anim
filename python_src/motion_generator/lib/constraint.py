# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:32:58 2015

@author: mamauer, erhe01, hadu01, FARUPP

Provides all funktionality to check how good a constraint is met
 by a motion primitive sample s
"""

import numpy as np
from utilities.motion_editing import convert_quaternion_frame_to_cartesian_frame,\
                    get_cartesian_coordinates_from_quaternion2,\
                    align_point_clouds_2D,\
                    transform_quaternion_frames,\
                    pose_orientation,\
                    transform_point_cloud,\
                    calculate_point_cloud_distance,\
                    fast_quat_frames_transformation
from utilities.bvh import get_joint_weights
from utilities.custom_transformations import vector_distance
from external.transformations import rotation_matrix

POSITION_ERROR_FACTOR = 1  # importance of reaching position constraints
ROTATION_ERROR_FACTOR = 10  # importance of reaching rotation constraints
RELATIVE_HUERISTIC_RANGE = 0.10  # used for setting the search range relative to the number of frames of motion primitive

global_counter_dict = {}
global_counter_dict["evaluations"] = 0# counter for calls of the objective function
global_counter_dict["motionPrimitveErrors"] = []# holds errors of individual motion primitives



def find_aligned_quaternion_frames(mm, s, prev_frames, start_pose, bvh_read,
                                   node_name_map):
    """Align quaternion frames from low dimensional vector s based on
    previous frames
       
    Parameters
    ----------
    * mm: motion primitive
    * s: numpy array
    \tLow dimensional vector for motion sample from motion primitive
    * prev_frames: list
    \tA list of quaternion frames
    * start_pose: dict
    \tA dictionary contains staring position and orientation
    
    Returns:
    quaternion_frames
    """
    # get quaternion frames of input motion s
    use_time_parameters = False # Note: time parameters are not necessary for alignment
    quat_frames = mm.back_project(s, use_time_parameters=use_time_parameters).get_motion_vector()
    # find alignment transformation: rotation and translation
    # covert the first frame of quat_frames and the last frame of pre_frames 
    # into euler frames, then compute the transformation based on point cloud
    # alignment
    
    if prev_frames is not None:
        angle, offset = fast_quat_frames_transformation(prev_frames,quat_frames)
        transformation = {"orientation":[0,angle,0],"position":offset}
        #print "transformation from fast quat",transformation
#        point_cloud_a = convert_quaternion_frame_to_cartesian_frame(bvh_read,
#                                                               prev_frames[-1],
#                                                               node_name_map)
#        point_cloud_b = convert_quaternion_frame_to_cartesian_frame(bvh_read,
#                                                               quat_frames[0],
#                                                               node_name_map)
#        weights = get_joint_weights(bvh_read, node_name_map)
#        theta, offset_x, offset_z = align_point_clouds_2D(point_cloud_a,
#                                                          point_cloud_b,
#                                                          weights)
#
#        transformation = {"orientation": [0, np.rad2deg(theta), 0],
#                          "position": np.array([offset_x, 0, offset_z])}  
#        print "transformation from point cloud",transformation
                                                 
    elif start_pose is not None:
        transformation = start_pose

    quat_frames = transform_quaternion_frames(quat_frames,
                                              transformation["orientation"],
                                              transformation["position"])   
    return quat_frames   




def constraint_distance(constraint, target_position=None,
                        target_orientation=None):
    """returns the euclidean distance of the target_position and the position
    of the constraint and the frobenius distance of the rotations

    Parameters
    ----------
    * constraint : dict
    \tThe constraint, contains position and orientation
    * target_position : list
    \tThe position of the target
    * target_orientation : list
    \tThe orientation of the target

    Returns
    -------
    * position_distance : float
    \tThe euclidean distance
    * rotation_distance : float
    \tThe frobenius difference of the rotation matrices
    """
    position_distance = 0.0
    rotation_distance = 0.0

    if "position" in constraint.keys() and target_position is not None:
        position_distance = vector_distance(constraint["position"],
                                            target_position)
    if "orientation" in constraint.keys() and target_orientation is not None:
        # get orientation distance based on unique representation:
        # rotation matrix, orientation order is X, Y, Z

        rotation_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        assert len(constraint["orientation"]) == 3,\
            ('the length of rotation vector should be 3')
        rotmat_constraint = np.eye(4)
        rotmat_target = np.eye(4)
        for i in xrange(len(constraint["orientation"])):
            if constraint["orientation"][i] is not None:
                tmp_constraint = rotation_matrix(np.deg2rad(constraint
                                                 ["orientation"][i]),
                                                 rotation_axes[i])
                rotmat_constraint = np.dot(tmp_constraint, rotmat_constraint)
                tmp_target = rotation_matrix(np.deg2rad(target_orientation[i]),
                                             rotation_axes[i])
                rotmat_target = np.dot(tmp_target, rotmat_target)
        rotation_distance = vector_distance(np.ravel(rotmat_constraint),
                                            np.ravel(rotmat_target))

    return position_distance, rotation_distance


def check_pos_and_rot_constraint_one_frame(frame, constraint, bvh_reader,
                               node_name_map=None,
                               precision={"pos":1,"rot":1},
                               verbose=False):
    """ checks whether the constraint is fullfiled for a given frame

    Parameters
    ----------
    * frame : numpy.ndarry
    \tThe frame to check already transformed to global coordinate system
    * constraint : tuple
    \tThe constraint as (joint, [pos_x, pos_y, pos_z],
    \t[rot_x, rot_y, rot_z]) where unconstrained variables
    \tare set to None
    * bvh_reader: BVHReader
    \tUsed for hierarchy information
    * node_name_map : dict
    \t Maps node name to index in frame vector (ignoring "Bip" joints)
    * pos_precision : float
    \tThe precision of the position in the sample to be rated as
    \t"constraint fulfilled"
    * rot_precision : float
    \tThe precision of the rotation
    Returns
    -------
    * success : bool
    \tWhether the constraint is fulfilled or not
    * distance : float
    \t distance to constraint
    """
    if "joint" in constraint.keys():
        node_name = constraint["joint"]
        if node_name_map is not None:
            target_position = get_cartesian_coordinates_from_quaternion2(bvh_reader,
                                                         node_name, frame,
                                                         node_name_map)

        else:
            target_position = get_cartesian_coordinates_from_quaternion2(bvh_reader, node_name,
                                                        frame)          

        pos_distance, rot_distance = constraint_distance(constraint,
                                                         target_position=
                                                         target_position,
                                                         target_orientation=
                                                         None)
        if pos_distance <= precision["pos"]  and rot_distance <= precision["rot"]:
            success = True
        else:
            success = False

        #return sum of distances for the selection of the best sample
        return success, pos_distance + rot_distance
    else:
        print "missing joint name in constraint definition"
        return False,np.inf


def check_dir_constraint(quat_frames, constraint, precision):
    
          

        # get motion orientation
#        motion_dir = get_orientation_vec(frames)
        motion_dir = pose_orientation(quat_frames[-1])
#        last_transformation = create_transformation(frames[-1][3:6],[0, 0, 0])
#        motion_dir = transform_point(last_transformation,[0,0,1])
#        motion_dir = np.array([motion_dir[0], motion_dir[2]])
#        motion_dir = motion_dir/np.linalg.norm(motion_dir)
        target_dir = np.array([constraint[0], constraint[2]])
        target_dir = target_dir/np.linalg.norm(target_dir)
        
        r_distance = abs(target_dir[0] - motion_dir[0]) + \
                     abs(target_dir[1] - motion_dir[1])
        if r_distance < precision:
            success = True
        else:
            success = False

        r_distance = r_distance * ROTATION_ERROR_FACTOR
        # to check the last frame pass rotation and trajectory constraint or not
        # put higher weights for orientation constraint
        n_frames = len(quat_frames) 
        if success:
            return [(n_frames-1, r_distance)], []
        else:
            return [], [(n_frames - 1,r_distance)]



def check_frame_constraint(quat_frames, constraint, precision, bvh_reader, node_name_map):

        # get point cloud of first frame
        point_cloud = convert_quaternion_frame_to_cartesian_frame(bvh_reader,quat_frames[0],
                                                             node_name_map)

        constraint_point_cloud = []
        for joint in node_name_map.keys():
            constraint_point_cloud.append(constraint[joint])
        weights = get_joint_weights(bvh_reader,node_name_map)

        theta, offset_x, offset_z = align_point_clouds_2D(constraint_point_cloud,
                                                          point_cloud,
                                                          weights)
        t_point_cloud = transform_point_cloud(point_cloud, theta, offset_x, offset_z)

        error = calculate_point_cloud_distance(constraint_point_cloud,t_point_cloud)
        if error < precision:
            success = True
        else:
            success = False
        if success:
            return [(0, error)], []
        else:
            return [], [(0,error)]


def convert_annotation_to_indices(constrain_first_frame, constrain_last_frame):
        start_stop_dict = {
            (None, None): (None, None),
            (True, None): (None, 1),
            (False, None): (1, None),
            (None, True): (-1, None),
            (None, False): (None, -1),
            (True, False): (None, 1),
            (False, True): (-1, None),
            (False, False): (1, -1)
        }
        start, stop = start_stop_dict[(constrain_first_frame, constrain_last_frame)]
        return start, stop





def check_pos_and_rot_constraint(quat_frames, constraint, precision, bvh_reader, node_name_map, annotation, verbose=False):
    #        print "position constraint is called"
        constrain_first_frame, constrain_last_frame = annotation
        n_frames = len(quat_frames) 
        #check specific frames
        last_frame_quat =quat_frames[-1]
        first_frame_quat = quat_frames[0]
        #last_frame_euler = np.ravel(convert_quaternion_to_euler([quat_frames[-1]]))
        if constrain_first_frame and constrain_last_frame:  # special case because of and

            first,f_distance = check_pos_and_rot_constraint_one_frame(first_frame_quat, 
                                                          constraint, 
                                                          bvh_reader,
                                                          node_name_map,
                                                          precision,
                                                          verbose)

            last,l_distance = check_pos_and_rot_constraint_one_frame(last_frame_quat,
                                                         constraint,  
                                                         bvh_reader,
                                                         node_name_map,
                                                         precision,
                                                         verbose)        
            if first+last:
                return [(0,f_distance), (n_frames - 1,l_distance )],[]
            else:
                return [],[(0,f_distance), (n_frames - 1,l_distance )]


    
        start, stop = convert_annotation_to_indices(constrain_first_frame, constrain_last_frame)

        n_frames = len(quat_frames)

        heuristic_range = RELATIVE_HUERISTIC_RANGE * n_frames

        filtered_frames = quat_frames[-heuristic_range:]
        filtered_frame_nos = range(n_frames)

        ok = []
        failed = []
        for frame_no, frame in zip(filtered_frame_nos, filtered_frames):#_euler
            success ,distance = check_pos_and_rot_constraint_one_frame(frame, constraint,  bvh_reader,node_name_map,
                                          precision,verbose)
            if success:
                ok.append((frame_no,distance))
            else:
                failed.append( (frame_no,distance))

        return ok,failed
  


def check_constraint(quat_frames,constraint,
                     bvh_reader,
                     range_start=None,
                     range_stop=None,
                     node_name_map=None,
                     start_pose=None,
                     precision={"pos":1,"rot":1,"smooth":1},
                     constrain_first_frame=None,
                     constrain_last_frame=None,
                     verbose=False):
    """ Main function of the modul. Check whether a sample fullfiles the
    constraint with the given precision or not. 
    Note only one type of constraint is allowed at once.

    Parameters
    ----------
    * quat_frames: np.ndarry
        contains the animation that is supposed to be checked
    * constraint : tuple
        The constraint as (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
        where unconstrained variables are set to None
    * range_start : int
    \tThe index of the frame where the range where the constraint should be
    \tchecked starts. This range is respected iff firstFrame=lastFrame=None
    * range_stop : int
    \tThe index where the range ends
    * bvh_reader: BVHReader
    \tUsed for hierarchy information
    * node_name_map : dict
    \t Optional: Maps node name to index in frame vector ignoring "Bip" joints
    * transformation : dict
      Contains position as cartesian coordinates and orientation
      as euler angles in degrees

    Returns
    -------
    * ok : list
        list of frames, which have been checked and
        where the constrained is fullfilled
        together with the distance to the constraint calculated using l2 norm ignoring None
    * failed: list
        list of frames, which have been checked and
        where the constrained is not fullfilled
        together with the distance to the constraint calculated using l2 norm ignoring None
    """



#    frames = find_aligned_frames(mm, s, prev_frames, start_pose,
#                                 bvh_reader, node_name_map)
          
    #handle the different types of constraints
    if "dir_vector" in constraint.keys() and constrain_last_frame: # orientation constraint on last frame
        return check_dir_constraint(quat_frames, constraint["dir_vector"], precision["rot"])
    elif "frame_constraint" in  constraint.keys():
        return check_frame_constraint(quat_frames, constraint["frame_constraint"], precision["smooth"], bvh_reader, node_name_map)
    elif "position" or "orientation" in constraint.keys():
        return check_pos_and_rot_constraint(quat_frames, constraint, precision, bvh_reader, node_name_map, (constrain_first_frame, constrain_last_frame), verbose=verbose)
    else:
        print "Error: Constraint type not reconginized"
        return [],[(0,10000)]

        

def evaluate_list_of_constraints(motion_primitive,s,constraints,prev_frames,start_pose,bvh_reader,node_name_map=None,\
                        precision = {"pos":1,"rot":1,"smooth":1},verbose=False):
    
    """
    Calculates the error of a list of constraints given a sample parameter value s.
    
    Returns
    -------
    * sum_error : float
    \tThe sum of the errors for all constraints
    * sucesses : list of bool
    \tSets for each entry in the constraints list wether or not a given precision was reached

    """
    error_sum = 0
    successes = []

    #find aligned frames once for all constraints
    aligned_frames  = find_aligned_quaternion_frames(motion_primitive, s, prev_frames,
                                                     start_pose, bvh_reader,
                                                     node_name_map)

    for c in constraints:
         good_frames, bad_frames = check_constraint(aligned_frames, c,
                                          bvh_reader,
                                          node_name_map=node_name_map,
                                          precision=precision, 
                                          constrain_first_frame=c["semanticAnnotation"]["firstFrame"] ,
                                          constrain_last_frame=c["semanticAnnotation"]["lastFrame"] ,
                                          verbose=verbose)

                                          
         if len(good_frames)>0:               
            c_min_distance = min((zip(*good_frames))[1])
            successes.append(True)
         else:
            c_min_distance =  min((zip(*bad_frames))[1])
            successes.append(False)
             
         error_sum+=c_min_distance
    return error_sum,successes
    

def obj_error_sum(s,data):
    s = np.asarray(s)
    motion_primitive, constraints, prev_frames,start_pose, bvh_reader, node_name_map,precision = data
    error_sum, successes = evaluate_list_of_constraints(motion_primitive,s,constraints,prev_frames,start_pose,bvh_reader,node_name_map,
                                                           precision=precision,verbose=False)
    global_counter_dict["evaluations"] += 1
    return error_sum



