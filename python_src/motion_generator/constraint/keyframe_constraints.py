# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 17:36:36 2015

@author: erhe01
"""

from math import sqrt
import numpy as np
from animation_data.motion_editing import convert_quaternion_frame_to_cartesian_frame,\
                    get_cartesian_coordinates_from_quaternion,\
                    align_point_clouds_2D,\
                    pose_orientation,\
                    transform_point_cloud,\
                    calculate_point_cloud_distance,\
                    quaternion_to_euler
from external.transformations import rotation_matrix

POSITION_ERROR_FACTOR = 1  # importance of reaching position constraints
ROTATION_ERROR_FACTOR = 10  # importance of reaching rotation constraints
RELATIVE_HUERISTIC_RANGE = 0.10  # used for setting the search range relative to the number of frames of motion primitive
CONSTRAINT_CONFLICT_ERROR = 100000  # returned when conflicting constraints were set

class KeyframeConstraint(object):
    def __init__(self,constraint_desc, precision):
        self.semantic_annotation = constraint_desc["semanticAnnotation"]
        self.precision = precision

    def evaluate_motion_sample(self, aligned_quat_frames):
        pass
        
    def evaluate_motion_sample_with_precision(self, aligned_quat_frames):
        error = self.evaluate_motion_sample(aligned_quat_frames)
        if error < self.precision:
            success = True
        else:
            success = False
        return error, success

class PoseConstraint(KeyframeConstraint):
    def __init__(self, skeleton, constraint_desc, precision):
        super(PoseConstraint, self).__init__(constraint_desc, precision)
        self.skeleton = skeleton
        self.pose_constraint = constraint_desc["frame_constraint"]
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


class DirectionConstraint(KeyframeConstraint):
    def __init__(self, skelton, constraint_desc, precision):
        super(DirectionConstraint, self).__init__(constraint_desc, precision)
        self.direction_constraint = constraint_desc["dir_vector"]
        self.target_dir = np.array([self.direction_constraint[0], self.direction_constraint[2]])
        self.target_dir = self.target_dir/np.linalg.norm(self.target_dir)
        self.rotation_error_factor = ROTATION_ERROR_FACTOR
        return
        
    def evaluate_motion_sample(self, aligned_quat_frames):
        #   motion_dir = get_orientation_vec(frames)
        motion_dir = pose_orientation(aligned_quat_frames[-1])
    
        
        error = abs(self.target_dir[0] - motion_dir[0]) + \
                     abs(self.target_dir[1] - motion_dir[1])
        
        # to check the last frame pass rotation and trajectory constraint or not
        # put higher weights for orientation constraint
        return error * self.rotation_error_factor


class PositionAndRotationConstraint(KeyframeConstraint):
    """
    * skeleton: Skeleton
        Necessary for the evaluation of frames
    * constraint_desc: dict
        Contains joint, position, orientation and semantic Annotation
    """
    def __init__(self, skeleton, constraint_desc, precision):
        super(PositionAndRotationConstraint, self).__init__(constraint_desc, precision)
        self.skeleton = skeleton
        self.joint_name = constraint_desc["joint"]
        if "position" in constraint_desc.keys():
            self.position = constraint_desc["position"]
        else:
            self.position = None
        if "orientation" in constraint_desc.keys():
            self.orientation = constraint_desc["orientation"]
        else:
            self.orientation = None
        self.relative_heuristic_range = RELATIVE_HUERISTIC_RANGE
        self.constrain_first_frame = constraint_desc["semanticAnnotation"]["firstFrame"]
        self.constrain_last_frame = constraint_desc["semanticAnnotation"]["lastFrame"]
        self._convert_annotation_to_indices()
        self.rotation_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


    def evaluate_motion_sample(self, aligned_quat_frames):

        min_error = CONSTRAINT_CONFLICT_ERROR
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
            error +=  self._evaluate_joint_orientation(frame)
        return error
        
    def _evaluate_joint_orientation(self, frame):
        joint_index = self.skeleton.node_name_map[self.joint_name]
        joint_orientation = frame[joint_index:joint_index+4]
        return self._orientation_distance(joint_orientation)
        
    def _evaluate_joint_position(self, frame):
        joint_position = get_cartesian_coordinates_from_quaternion(self.skeleton, self.joint_name, frame)
        return self._vector_distance(self.position, joint_position)
                                          
    def _orientation_distance(self, joint_orientation):
        joint_euler_angles = quaternion_to_euler(joint_orientation)
        rotmat_constraint = np.eye(4)
        rotmat_target = np.eye(4)
        for i in xrange(3):
            if self.orientation[i] is not None:
                tmp_constraint = rotation_matrix(np.deg2rad(self.orientation[i]),
                                                 self.rotation_axes[i])
                rotmat_constraint = np.dot(tmp_constraint, rotmat_constraint)
                tmp_target = rotation_matrix(np.deg2rad(joint_euler_angles[i]),
                                             self.rotation_axes[i])
                rotmat_target = np.dot(tmp_target, rotmat_target)
        rotation_distance = self._vector_distance(np.ravel(rotmat_constraint),
                                            np.ravel(rotmat_target))
        return rotation_distance
        
    def _vector_distance(self, a, b):
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
