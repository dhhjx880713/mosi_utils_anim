# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:26:29 2015

@author: erhe01
"""

import copy
import numpy as np
from lib.exceptions import PathSearchError
from lib.input_processing import extract_keyframe_annotations, \
                                transform_point_from_cad_to_opengl_cs, \
                                extract_trajectory_constraint,\
                                create_trajectory_from_constraint,\
                                extract_keyframe_constraint,\
                                extract_all_keyframe_constraints
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
        last_pos = copy.copy(last_pos)
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





        

class ElementaryActionConstraints(object):
    def __init__(self,action_index,motion_constraints):
        self.parent_constraint = motion_constraints
        self.action_name = motion_constraints.elementary_action_list[action_index]["action"]
        self.keyframe_annotations = motion_constraints.keyframe_annotations[action_index]
        self.constraints = motion_constraints.elementary_action_list[action_index]["constraints"]
        self.max_step = motion_constraints.max_step
        self.start_pose = motion_constraints.start_pose
        self._extract_constraints_from_motion_constraint_list()
                                         
    def get_subgraph(self):
        return self.parent_constraint.morphable_graph.subgraphs[self.action_name]
        
    def get_skeleton(self):
        return self.parent_constraint.morphable_graph.skeleton
        
    def _extract_constraints_from_motion_constraint_list(self):
        """ Extracts keyframe and trajectory constraints from constraint_list
        Returns:
        -------
        * trajectory: ParameterizedSpline
            Spline parameterized by arc length.
        * unconstrained_indices: list of indices
            lists of indices of degrees of freedom to ignore in the constraint evaluation.
        * keyframe_constraints: dict of lists
            Lists of constraints for each motion primitive in the subgraph.
        """
        morphable_subgraph = self.parent_constraint.morphable_graph.subgraphs[self.action_name]
        root_joint_name = self.parent_constraint.morphable_graph.skeleton.root# currently only trajectories on the Hips joint are supported
        self.trajectory, self.unconstrained_indices = self._extract_trajectory_from_constraint_list(self.constraints, root_joint_name)
    
        keyframe_constraints = extract_all_keyframe_constraints(self.constraints,
                                                                morphable_subgraph)
        self.keyframe_constraints = self._reorder_keyframe_constraints_for_motion_primitves(morphable_subgraph,
                                                                                 keyframe_constraints)



    def _reorder_keyframe_constraints_for_motion_primitves(self, morphable_subgraph, keyframe_constraints):
         """ Order constraints extracted by extract_all_keyframe_constraints for each state
         """
         constraints = {}#dict of lists
         #iterate over keyframe labels
         for label in keyframe_constraints.keys():
            state = morphable_subgraph.annotation_map[label]
            time_information = morphable_subgraph.mp_annotations[state][label]
            constraints[state] = []
            # iterate over joints constrained at that keyframe
            for joint_name in keyframe_constraints[label].keys():
                # iterate over constraints for that joint
                for c in keyframe_constraints[label][joint_name]:
                    # create constraint definition usable by the algorithm
                    # and add it to the list of constraints for that state
                    constraint_desc = extract_keyframe_constraint(joint_name,c,\
                                                morphable_subgraph,time_information)
                    constraints[state].append(constraint_desc)
         return constraints


    def _extract_trajectory_from_constraint_list(self, constraint_list, joint_name):
        """ Extract the trajectory information from the constraints and constructs
            a trajectory as an ParameterizedSpline instance.
        Returns:
        -------
        * trajectory: ParameterizedSpline
            Spline parameterized by arc length.
        * unconstrained_indices: list of indices
            Lists of indices of degrees of freedom to ignore in the constraint evaluation.
        """
        trajectory_constraint = extract_trajectory_constraint(constraint_list,joint_name)
        if  trajectory_constraint is not None:
            #print "found trajectory constraint"
            return create_trajectory_from_constraint(trajectory_constraint)
        else:
            return None, None


class MotionConstraints(object):
    """
    Parameters
    ----------
    * mg_input : json data read from a file
        Contains elementary action list with constraints, start pose and keyframe annotations.
    * max_step : integer
        Sets the maximum number of graph walk steps to be performed. If less than 0
        then it is unconstrained
    """
    def __init__(self, mg_input, morphable_graph, max_step=-1):
        self.morphable_graph = morphable_graph
        self.max_step = max_step
        self.elementary_action_list = mg_input["elementaryActions"]
        self.keyframe_annotations = extract_keyframe_annotations(self.elementary_action_list)
        self.start_pose = mg_input["startPose"]
        self._transform_from_left_to_right_handed_cs()
        self.action_index = 0
        self.n_actions = len(self.elementary_action_list)

            
    
    def _transform_from_left_to_right_handed_cs(self):
        """ Transform transition and rotation of the start pose from CAD to Opengl 
            coordinate system.
        """
        start_pose_copy = copy.copy(self.start_pose)
        self.start_pose["orientation"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["orientation"])
        self.start_pose["position"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["position"])
        
        
    def get_next_elementary_action_constraints(self):
        """
        Returns:
        --------
        * action_constraints : ElementarActionConstraints
          Constraints for the next elementary action extracted from an input file.
        """
        if self.action_index < self.n_actions:
            action_constraints = ElementaryActionConstraints(self.action_index, self)
            self.action_index+=1
            return action_constraints
        else:
            return None
  