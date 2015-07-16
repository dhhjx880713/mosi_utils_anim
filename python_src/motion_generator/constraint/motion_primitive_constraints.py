# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:42:13 2015

@author: erhe01
"""
from copy import copy
import numpy as np
from utilities.exceptions import PathSearchError
from utilities.motion_editing import convert_quaternion_to_euler, \
                                get_cartesian_coordinates_from_euler                               

       
class MotionPrimitiveConstraints(object):
    """ Creates a list of constraints for a motion primitive based on the current state and position
     Attributes
     -------
     * constraints : list of dicts
      Each dict contains joint, position,orientation and semanticAnnotation describing a constraint
    """
    def __init__(self, motion_primitive_name, action_constraints,last_arc_length,last_pos, settings, prev_frames=None, is_last_step=False):
        self.motion_primitive_name = motion_primitive_name
        self.settings = settings

        self.constraints = []
        self.goal_arc_length = 0
        
        pose_constraint_set = False
        if action_constraints.trajectory is not None:
            pose_constraint_set = self._set_trajectory_constraints(action_constraints, last_pos, last_arc_length,is_last_step, prev_frames)
   
        if len(action_constraints.keyframe_constraints.keys()) > 0:
            self._set_keyframe_constraints(action_constraints, pose_constraint_set, is_last_step, prev_frames)
            
        self.use_optimization = len(action_constraints.keyframe_constraints.keys()) > 0 or is_last_step

    
    def _set_trajectory_constraints(self, action_constraints, last_pos, last_arc_length, is_last_step, prev_frames):
        pose_constraint_set = False
        morphable_subgraph = action_constraints.get_subgraph()
        skeleton = action_constraints.get_skeleton()
        
        step_length_factor = self.settings["step_length_factor"] #0.8 #used to increase the number of samples that can reach the point
        method =  self.settings["method"]#"arc_length"
        last_pos = copy(last_pos)
        last_pos[1] = 0.0
        print "search for new goal"
        # if it is the last step we need to reach the point exactly otherwise
        # make a guess for a reachable point on the path that we have not visited yet
        if not is_last_step:
            self.goal_arc_length = self._make_guess_for_goal_arc_length(morphable_subgraph,
                                                   self.motion_primitive_name,action_constraints.trajectory,
                                                   last_arc_length,last_pos,
                                                   action_constraints.unconstrained_indices,
                                                   step_length_factor,method)
        else:
            self.goal_arc_length = action_constraints.trajectory.full_arc_length

        goal,orientation,dir_vector = self._get_point_and_orientation_from_arc_length(action_constraints.trajectory,self.goal_arc_length,action_constraints.unconstrained_indices)

#        print  "starting from",last_pos,last_arc_length,"the new goal for", \
#                current_motion_primitive,"is",goal,"at arc length",arc_length
        print "starting from: "
        print last_pos
        print "the new goal for " + self.motion_primitive_name
        print goal
        print "arc length is: " + str(self.goal_arc_length)

        if self.settings["use_frame_constraints"] and  prev_frames is not None and skeleton is not None:
            frame_constraint = self._create_frame_constraint(skeleton, prev_frames)
            self.constraints.append(frame_constraint)
            pose_constraint_set = True

        if self.settings["use_position_constraints"]:
            root_joint_name = skeleton.root
            if not is_last_step:
                pos_semantic_annotation={"firstFrame":None,"lastFrame":True}
            else:
                pos_semantic_annotation={"firstFrame":None,"lastFrame":True}
            pos_constraint = {"joint":root_joint_name,"position":goal,
                      "semanticAnnotation":pos_semantic_annotation}
            self.constraints.append(pos_constraint)

        if self.settings["use_dir_vector_constraints"] :
            rot_semantic_annotation={"firstFrame":None,"lastFrame":True}
            rot_constraint = {"joint":root_joint_name, "dir_vector":dir_vector,
                          "semanticAnnotation":rot_semantic_annotation}
            self.constraints.append(rot_constraint)
        return pose_constraint_set


    def _set_keyframe_constraints(self, action_constraints, pose_constraint_set, is_last_step, prev_frames):
        """ extract keyframe constraints of the motion primitive name
        """
        if self.motion_primitive_name in action_constraints.keyframe_constraints.keys():
            self.constraints+= action_constraints.keyframe_constraints[self.motion_primitive_name]
            
        # generate frame constraints for the last step basd on the previous state
        # if not already done for the trajectory following
        if not pose_constraint_set and is_last_step and prev_frames is not None:
            skeleton = action_constraints.get_skeleton()
            frame_constraint= self._create_frame_constraint(skeleton, prev_frames)
            self.constraints.append(frame_constraint)
      

    def _create_frame_constraint(self, skeleton, prev_frames):
        """
        create frame a constraint from the preceding motion.
    
        """
    
    #    last_frame = prev_frames[-1]
        last_euler_frame = np.ravel(convert_quaternion_to_euler([prev_frames[-1]]))
        position_dict = {}
        for node_name in skeleton.node_name_map.keys():
    #        target_position = get_cartesian_coordinates_from_quaternion(skeleton,
    #                                                             node_name, frame)
            joint_position = get_cartesian_coordinates_from_euler(skeleton,
                                                            node_name,
                                                            last_euler_frame)
    #        print "add joint position to constraints",node_name,joint_position
            position_dict[node_name] = joint_position
        frame_constraint = {"frame_constraint":position_dict, "semanticAnnotation":{"firstFrame":True,"lastFrame":None}}
    
        return frame_constraint


    
    
    def _make_guess_for_goal_arc_length(self, morphable_subgraph, current_motion_primitive, trajectory,
                                       last_arc_length, last_pos, unconstrained_indices=None,
                                        step_length_factor=1.0, method = "arc_length"):
        """ Makes a guess for a reachable arc length based on the current position.
            It searches for the closest point on the trajectory, retrieves the absolute arc length
            and its the arc length of a random sample of the next motion primitive
        Returns
        -------
        * arc_length : float
          The absolute arc length of the new goal on the trajectory.
          The goal should then be extracted using get_point_and_orientation_from_arc_length
        """
        if unconstrained_indices is None:
            unconstrained_indices = []
        step_length = morphable_subgraph.nodes[current_motion_primitive].average_step_length * step_length_factor
        max_arc_length = last_arc_length + 4.0* step_length
        #find closest point in the range of the last_arc_length and max_arc_length
        closest_point,distance = trajectory.find_closest_point(last_pos,min_arc_length = last_arc_length,max_arc_length=max_arc_length)
        if closest_point is None:
            parameters = {"last":last_arc_length,"max":max_arc_length,"full":trajectory.full_arc_length}
            print "did not find closest point",closest_point,str(parameters)
            raise PathSearchError(parameters)
        # approximate arc length of the point closest to the current position
        arc_length,eval_point = trajectory.get_absolute_arc_length_of_point(closest_point,min_arc_length = last_arc_length)
        #update arc length based on the step length of the next motion primitive
        if arc_length == -1 :
            arc_length = trajectory.full_arc_length
        else:
            arc_length +=step_length# max(step_length-distance,0)
        return arc_length
        
    def _get_point_and_orientation_from_arc_length(self, trajectory,arc_length,unconstrained_indices):
        """ Returns a point, an orientation and a directoion vector on the trajectory
        """
        point = trajectory.query_point_by_absolute_arc_length(arc_length).tolist()
    
        reference_vector = np.array([0,1])# in z direction
        start,dir_vector,angle = trajectory.get_angle_at_arc_length_2d(arc_length,reference_vector)
        orientation = [None,angle,None]
        for i in unconstrained_indices:
            point[i] = None
            orientation[i] = None
        return point,orientation,dir_vector


