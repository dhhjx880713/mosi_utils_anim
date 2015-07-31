# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 10:32:58 2015

@author: mamauer, erhe01, hadu01, FARUPP

Provides all funktionality to check how good a constraint is met
 by a motion primitive sample s
"""
from math import sqrt
import numpy as np
from animation_data.motion_editing import convert_quaternion_frame_to_cartesian_frame,\
                    get_cartesian_coordinates_from_quaternion,\
                    align_point_clouds_2D,\
                    transform_quaternion_frames,\
                    pose_orientation,\
                    transform_point_cloud,\
                    calculate_point_cloud_distance,\
                    fast_quat_frames_transformation
from external.transformations import rotation_matrix

POSITION_ERROR_FACTOR = 1  # importance of reaching position constraints
ROTATION_ERROR_FACTOR = 10  # importance of reaching rotation constraints
RELATIVE_HUERISTIC_RANGE = 0.10  # used for setting the search range relative to the number of frames of motion primitive
CONSTRAINT_CONFLICT_ERROR = 100000  # returned when conflicting constraints were set



def vector_distance(a,b):
    """Returns the distance ignoring entries with None
    """
    d_sum = 0
    #print a,b
    for i in xrange(len(a)):
        if a[i] is not None and b[i] is not None:
            d_sum += (a[i]-b[i])**2
    return sqrt(d_sum)




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


def check_pos_and_rot_constraint_one_frame(frame, constraint, skeleton,
                               precision={"pos":1,"rot":1},
                               verbose=False):
    """Checks whether the constraint is fullfiled for a given frame

    Parameters
    ----------
    * frame : numpy.ndarry
    \tThe frame to check already transformed to global coordinate system
    * constraint : tuple
    \tThe constraint as (joint, [pos_x, pos_y, pos_z],
    \t[rot_x, rot_y, rot_z]) where unconstrained variables
    \tare set to None
    * skeleton: Skeleton
    \tUsed for hierarchy information
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
    
        target_position = get_cartesian_coordinates_from_quaternion(skeleton,
                                                     node_name, frame)
      

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


def check_dir_constraint(aligned_quat_frames, direction_constraint, precision):
    """ Evaluates the direction of the movement and compares it with the 
    constraint value.
    
    Parameters
    ----------
    * aligned_quat_frames: np.ndarray
        Motion aligned to previous motion in quaternion format.
    * direction_constraint: np.ndarray
        3D constraint direction vector.
    * skeleton: Skeleton
        Used for hierarchy information.
    * node_name_map : dict
       Optional: Maps node name to index in frame vector ignoring "Bip" joints.
       
    Returns
    -------
    * error: float
        Difference to the desired constraint value.
    * in_precision: Boolean
        If error is inside range precision.
    """

    # get motion orientation
#   motion_dir = get_orientation_vec(frames)
    motion_dir = pose_orientation(aligned_quat_frames[-1])
    target_dir = np.array([direction_constraint[0], direction_constraint[2]])
    target_dir = target_dir/np.linalg.norm(target_dir)
    
    error = abs(target_dir[0] - motion_dir[0]) + \
                 abs(target_dir[1] - motion_dir[1])
    if error < precision:
        in_precision = True
    else:
        in_precision = False
    # to check the last frame pass rotation and trajectory constraint or not
    # put higher weights for orientation constraint
    return error * ROTATION_ERROR_FACTOR, in_precision



def check_frame_constraint(quat_frames, frame_constraint, precision, skeleton):
    """ Evaluates the difference between the first frame of the motion 
    and the frame constraint.
    
    Parameters
    ----------
    * aligned_quat_frames: np.ndarray
        Motion aligned to previous motion in quaternion format
    * frame_constraint: dict of np.ndarray
        Dict containing a position for each joint
    * skeleton: Skeleton
        Used for hierarchy information
    * node_name_map : dict
       Optional: Maps node name to index in frame vector ignoring "Bip" joints
    
    Returns
    -------
    * error: float
        Difference to the desired constraint value.
    * in_precision: Boolean
        If error is inside range precision.
    """
    # get point cloud of first frame
    point_cloud = convert_quaternion_frame_to_cartesian_frame(skeleton,quat_frames[0])

    constraint_point_cloud = []
    for joint in skeleton.node_name_map.keys():
        constraint_point_cloud.append(frame_constraint[joint])
    theta, offset_x, offset_z = align_point_clouds_2D(constraint_point_cloud,
                                                      point_cloud,
                                                      skeleton.joint_weights)
    t_point_cloud = transform_point_cloud(point_cloud, theta, offset_x, offset_z)

    error = calculate_point_cloud_distance(constraint_point_cloud,t_point_cloud)
    if error < precision:
        in_precision = True
    else:
        in_precision = False
    return error, in_precision 


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





def check_pos_and_rot_constraint(aligned_quat_frames, constraint, precision, skeleton, annotation, verbose=False):
    """Evaluates position and orientation constraints for the aligned frames
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
#   print "position constraint is called"
    constrain_first_frame, constrain_last_frame = annotation
    n_frames = len(aligned_quat_frames) 
    #check specific frames
    # check for a special case which should not happen in a single constraint
    if not (constrain_first_frame and constrain_last_frame):  
        
        start, stop = convert_annotation_to_indices(constrain_first_frame, constrain_last_frame)
    
    
    
        heuristic_range = RELATIVE_HUERISTIC_RANGE * n_frames
    
        filtered_frames = aligned_quat_frames[-heuristic_range:]
        filtered_frame_nos = range(n_frames)
    
        ok = []
        failed = []
        for frame_no, frame in zip(filtered_frame_nos, filtered_frames):
            success ,distance = check_pos_and_rot_constraint_one_frame(frame, constraint, 
                                                                        skeleton,
                                                                        precision,verbose)
            if success:
                ok.append((frame_no,distance))
            else:
                failed.append( (frame_no,distance))
    
        return ok,failed
    else:
        print "Warning conflicting constraint was set"
        return [],[ (0, CONSTRAINT_CONFLICT_ERROR)]
  


def check_constraint(quat_frames,constraint,
                     skeleton,
                     range_start=None,
                     range_stop=None,
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
    * skeleton: Skeleton
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
          
    #handle the different types of constraints
    if "dir_vector" in constraint.keys() and constrain_last_frame: # orientation constraint on last frame
        error, in_precision = check_dir_constraint(quat_frames, constraint["dir_vector"], precision["rot"])
        
        n_frames = len(quat_frames) 
        if in_precision:
            return [(n_frames-1, error)], []
        else:
            return [], [(n_frames - 1,error)]
            
    elif "frame_constraint" in  constraint.keys():
        error, in_precision= check_frame_constraint(quat_frames, constraint["frame_constraint"], precision["smooth"], skeleton)
        if in_precision:
            return [(0, error)], []
        else:
            return [], [(0,error)]

    elif "position" or "orientation" in constraint.keys():
        return check_pos_and_rot_constraint(quat_frames, constraint, precision, skeleton, (constrain_first_frame, constrain_last_frame), verbose=verbose)
    else:
        print "Error: Constraint type not recognized"
        return [],[(0,10000)]

        

def evaluate_list_of_constraints(motion_primitive,s,constraints,prev_frames,start_pose,skeleton,\
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
    aligned_frames  = find_aligned_quaternion_frames(motion_primitive, s, prev_frames, start_pose)

    for c in constraints:
         good_frames, bad_frames = check_constraint(aligned_frames, c,
                                          skeleton,
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
    return error_sum, successes
    




