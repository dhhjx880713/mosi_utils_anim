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
                    pose_orientation,\
                    transform_point_cloud,\
                    calculate_point_cloud_distance
from external.transformations import rotation_matrix

POSITION_ERROR_FACTOR = 1  # importance of reaching position constraints
ROTATION_ERROR_FACTOR = 10  # importance of reaching rotation constraints
RELATIVE_HUERISTIC_RANGE = 0.10  # used for setting the search range relative to the number of frames of motion primitive
CONSTRAINT_CONFLICT_ERROR = 100000  # returned when conflicting constraints were set


class PoseConstraint(object):
    def __init__(self, skeleton, constraint_desc, precision):
        self.skeleton = skeleton
        self.pose_constraint = constraint_desc["frame_constraint"]
        self.precision = precision
        return
    
    def evaluate_motion_sample(self, aligned_quat_frames):
        
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
        point_cloud = convert_quaternion_frame_to_cartesian_frame(self.skeleton, aligned_quat_frames[0])
    
        constraint_point_cloud = []
        for joint in self.skeleton.node_name_map.keys():
            constraint_point_cloud.append(self.pose_constraint[joint])
        theta, offset_x, offset_z = align_point_clouds_2D(constraint_point_cloud,
                                                          point_cloud,
                                                          self.skeleton.joint_weights)
        t_point_cloud = transform_point_cloud(point_cloud, theta, offset_x, offset_z)
    
        error = calculate_point_cloud_distance(constraint_point_cloud,t_point_cloud)

        return error

class DirectionConstraint(object):
    def __init__(self, skelton, constraint_desc, precision):
        self.direction_constraint = constraint_desc["dir_vector"]
        self.target_dir = np.array([self.direction_constraint[0], self.direction_constraint[2]])
        self.target_dir = self.target_dir/np.linalg.norm(self.target_dir)
        self.rotation_error_factor = ROTATION_ERROR_FACTOR
        self.precision = precision
        return
        
    def evaluate_motion_sample(self, aligned_quat_frames):
        #   motion_dir = get_orientation_vec(frames)
        motion_dir = pose_orientation(aligned_quat_frames[-1])
    
        
        error = abs(self.target_dir[0] - motion_dir[0]) + \
                     abs(self.target_dir[1] - motion_dir[1])
        
        # to check the last frame pass rotation and trajectory constraint or not
        # put higher weights for orientation constraint
        return error * self.rotation_error_factor


class SpatialFrameConstraint(object):
    """
    * skeleton: Skeleton
        Necessary for the evaluation of frames
    * constraint_desc: dict
        Contains joint, position, orientation and semantic Annotation
    """
    def __init__(self, skeleton, constraint_desc, precision):
        self.skeleton = skeleton
        self.joint_name = constraint_desc["joint"]
        self.position = constraint_desc["position"]
        self.orientation = constraint_desc["orientation"]
        self.relative_heuristic_range = RELATIVE_HUERISTIC_RANGE
        self.constrain_first_frame = constraint_desc["semanticAnnotation"]["firstFrame"]
        self.constrain_last_frame = constraint_desc["semanticAnnotation"]["lastFrame"]
        self._convert_annotation_to_indices(self.constrain_first_frame, self.constrain_last_frame)
        self.precision = precision
        
        return

    def evaluate_motion_sample(self, aligned_quat_frames):

        min_error = 10000
        n_frames = len(aligned_quat_frames) 
        #check specific frames
        # check for a special case which should not happen in a single constraint
        if not (self.constrain_first_frame and self.constrain_last_frame):  

            heuristic_range = self.relative_heuristic_range * n_frames
            filtered_frames = aligned_quat_frames[-heuristic_range:]
            filtered_frame_nos = range(n_frames)            
            for frame_no, frame in zip(filtered_frame_nos, filtered_frames):
                error = self._evaluate_frame(frame)
                if min_error > error:
                    min_error = error

        return min_error
        
    def _evaluate_frame(self, frame):
        error = 0
        if self.position is not None:
            error += self._evaluate_joint_position(frame)
        if self.orientation is not None:
            error +=  self._joint_evaluate_orientation(frame)
        return error
        
    def _evaluate_joint_orientation(self, frame):
        joint_index = self.skeleton.node_name_map[self.joint_name]
        joint_orientation = frame[joint_index:joint_index+3]
        return self._rotation_distance(joint_orientation)
        
    def _evaluate_joint_position(self, frame):
        joint_position = get_cartesian_coordinates_from_quaternion(self.skeleton, self.joint_name, frame)
        return self._vector_distance(self.position, joint_position)
                                          
    def _orientation_distance(self, joint_orientation):
        rotation_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        rotmat_constraint = np.eye(4)
        rotmat_target = np.eye(4)
        for i in xrange(3):
            if self.orientation[i] is not None:
                tmp_constraint = rotation_matrix(np.deg2rad(self.orientation[i]),
                                                 rotation_axes[i])
                rotmat_constraint = np.dot(tmp_constraint, rotmat_constraint)
                tmp_target = rotation_matrix(np.deg2rad(joint_orientation[i]),
                                             rotation_axes[i])
                rotmat_target = np.dot(tmp_target, rotmat_target)
        rotation_distance = vector_distance(np.ravel(rotmat_constraint),
                                            np.ravel(rotmat_target))
        return rotation_distance
        
    def vector_distance(self, a,b):
        """Returns the distance ignoring entries with None
        """
        d_sum = 0
        #print a,b
        for i in xrange(len(a)):
            if a[i] is not None and b[i] is not None:
                d_sum += (a[i]-b[i])**2
        return sqrt(d_sum)

            
    def _convert_annotation_to_indices(self):
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
            self.start, self.stop = start_stop_dict[(self.constrain_first_frame, self.constrain_last_frame)]
      

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
  

