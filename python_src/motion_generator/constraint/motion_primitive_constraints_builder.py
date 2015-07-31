# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:38:15 2015

@author: erhe01
"""

from copy import copy
import numpy as np
from utilities.exceptions import PathSearchError
from animation_data.motion_editing import convert_quaternion_to_euler, \
                                get_cartesian_coordinates_from_euler                               
from motion_primitive_constraints import MotionPrimitiveConstraints

class MotionPrimitiveConstraintsBuilder(object):
    """ Extracts a list of constraints for a motion primitive from ElementaryActionConstraints 
        based on the variables set by the method set_status. Generates constraints for path following.
    """
    def __init__(self):
        self.action_constraints = None
        self.algorithm_config = None
        self.status = {}
        self.morphable_subgraph = None
        return
    
    def set_action_constraints(self, action_constraints):
        self.action_constraints = action_constraints
        self.morphable_subgraph = self.action_constraints.get_subgraph()
        self.skeleton = self.action_constraints.get_skeleton()
        
    def set_trajectory_following_settings(self, algorithm_config):
        self.algorithm_config = algorithm_config
        self.trajectory_following_settings = algorithm_config["trajectory_following_settings"]#  TODO move trajectory_following_settings to different key of the algorithm_config
        
    def set_status(self, motion_primitive_name, last_arc_length, prev_frames=None, is_last_step=False):
        self.status["motion_primitive_name"] = motion_primitive_name
        self.status["last_arc_length"] = last_arc_length
        if prev_frames is None:
            last_pos = self.action_constraints.start_pose["position"]  
        else:
            last_pos = prev_frames[-1][:3]
	
        last_pos = copy(last_pos)
        last_pos[1] = 0.0
        self.status["last_pos"] = last_pos
        self.status["prev_frames"] = prev_frames
        self.status["is_last_step"] = is_last_step

        
    def build(self):
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.motion_primitive_name =  self.status["motion_primitive_name"]
        mp_constraints.settings = self.trajectory_following_settings
        mp_constraints.constraints = []
        mp_constraints.goal_arc_length = 0
        mp_constraints.step_start = self.status["last_pos"]
        mp_constraints.start_pose = self.action_constraints.start_pose
        mp_constraints.skeleton = self.action_constraints.get_skeleton()
        mp_constraints.precision = self.action_constraints.precision
        mp_constraints.verbose = self.algorithm_config["verbose"]
        
        if self.action_constraints.trajectory is not None:
            self._set_trajectory_constraints(mp_constraints)
   
        if len(self.action_constraints.keyframe_constraints.keys()) > 0:
            self._set_keyframe_constraints(mp_constraints)
            
        mp_constraints.use_optimization = len(self.action_constraints.keyframe_constraints.keys()) > 0\
                                            or self.status["is_last_step"]
        return mp_constraints
        
    def _set_trajectory_constraints(self,mp_constraints):

        print "search for new goal"
        # if it is the last step we need to reach the point exactly otherwise
        # make a guess for a reachable point on the path that we have not visited yet
        if not self.status["is_last_step"]:
            mp_constraints.goal_arc_length = self._make_guess_for_goal_arc_length()
        else:
            mp_constraints.goal_arc_length = self.action_constraints.trajectory.full_arc_length

        mp_constraints.step_goal,orientation,dir_vector = self._get_point_and_orientation_from_arc_length(mp_constraints.goal_arc_length)

        mp_constraints.print_status()

        if mp_constraints.settings["use_frame_constraints"] and self.status["prev_frames"] is not None:
            frame_constraint = self._create_frame_constraint()
            mp_constraints.constraints.append(frame_constraint)
            mp_constraints.pose_constraint_set = True
        
        root_joint_name = self.skeleton.root
        if mp_constraints.settings["use_position_constraints"]:
          pos_semantic_annotation={"firstFrame":None,"lastFrame":True}
          pos_constraint = {"joint":root_joint_name,"position":mp_constraints.step_goal,
                      "semanticAnnotation":pos_semantic_annotation}
          mp_constraints.constraints.append(pos_constraint)
        if mp_constraints.settings["use_dir_vector_constraints"]:
            rot_semantic_annotation={"firstFrame":None,"lastFrame":True}
            rot_constraint = {"joint":root_joint_name, "dir_vector":dir_vector,
                          "semanticAnnotation":rot_semantic_annotation}
            mp_constraints.constraints.append(rot_constraint)
        


    def _set_keyframe_constraints(self, mp_constraints):
        """ Extract keyframe constraints of the motion primitive name.
        """

        if self.status["motion_primitive_name"] in self.action_constraints.keyframe_constraints.keys():
            mp_constraints.constraints+= self.action_constraints.keyframe_constraints[self.status["motion_primitive_name"]]
            
        # generate frame constraints for the last step basd on the previous state
        # if not already done for the trajectory following
        if not mp_constraints.pose_constraint_set and self.status["is_last_step"] and self.status["prev_frames"] is not None:
            frame_constraint= self._create_frame_constraint()
            mp_constraints.constraints.append(frame_constraint)

      

    def _create_frame_constraint(self):
        """ Create frame a constraint from the preceding motion.
        """
    #    last_frame = prev_frames[-1]
        last_euler_frame = np.ravel(convert_quaternion_to_euler([self.status["prev_frames"][-1]]))
        position_dict = {}
        for node_name in self.skeleton.node_name_map.keys():
    #        target_position = get_cartesian_coordinates_from_quaternion(skeleton,
    #                                                             node_name, frame)
            joint_position = get_cartesian_coordinates_from_euler(self.skeleton,
                                                                node_name,
                                                                last_euler_frame)
    #        print "add joint position to constraints",node_name,joint_position
            position_dict[node_name] = joint_position
        frame_constraint = {"frame_constraint":position_dict, "semanticAnnotation":{"firstFrame":True,"lastFrame":None}}
    
        return frame_constraint


    
    
    def _make_guess_for_goal_arc_length(self):
        """ Makes a guess for a reachable arc length based on the current position.
            It searches for the closest point on the trajectory, retrieves the absolute arc length
            and its the arc length of a random sample of the next motion primitive
        Returns
        -------
        * arc_length : float
          The absolute arc length of the new goal on the trajectory.
          The goal should then be extracted using get_point_and_orientation_from_arc_length
        """
        last_arc_length = self.status["last_arc_length"]
        last_pos = self.status["last_pos"]
       
        step_length = self.morphable_subgraph.nodes[self.status["motion_primitive_name"]].average_step_length\
                        * self.trajectory_following_settings["step_length_factor"]
        max_arc_length = last_arc_length + 4.0 * step_length
        #find closest point in the range of the last_arc_length and max_arc_length
        closest_point,distance = self.action_constraints.trajectory.find_closest_point(last_pos,min_arc_length=last_arc_length,max_arc_length=max_arc_length)
        if closest_point is None:
            parameters = {"last":last_arc_length,"max":max_arc_length,"full": self.action_constraints.trajectory.full_arc_length}
            print "did not find closest point",closest_point,str(parameters)
            raise PathSearchError(parameters)
        # approximate arc length of the point closest to the current position
        start_arc_length,eval_point = self.action_constraints.trajectory.get_absolute_arc_length_of_point(closest_point,min_arc_length=last_arc_length)
        #update arc length based on the step length of the next motion primitive
        if start_arc_length == -1 :
            return self.action_constraints.trajectory.full_arc_length
        else:
            return start_arc_length + step_length# max(step_length-distance,0)

        
    def _get_point_and_orientation_from_arc_length(self, arc_length):
        """ Returns a point, an orientation and a direction vector on the trajectory
        """
        point = self.action_constraints.trajectory.query_point_by_absolute_arc_length(arc_length).tolist()
        reference_vector = np.array([0,1])# in z direction
        start,dir_vector,angle = self.action_constraints.trajectory.get_angle_at_arc_length_2d(arc_length,reference_vector)
        orientation = [None,angle,None]
        for i in self.action_constraints.unconstrained_indices:
            point[i] = None
            orientation[i] = None
        return point, orientation, dir_vector


