# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:38:15 2015

@author: erhe01
"""

from copy import copy
import numpy as np
from ...utilities.exceptions import PathSearchError
from ...animation_data.motion_editing import convert_quaternion_to_euler, \
                                get_cartesian_coordinates_from_quaternion
from motion_primitive_constraints import MotionPrimitiveConstraints
from keyframe_constraints.pose_constraint import PoseConstraint
from keyframe_constraints.direction_constraint import DirectionConstraint
from keyframe_constraints.pos_and_rot_constraint import PositionAndRotationConstraint


class MotionPrimitiveConstraintsBuilder(object):
    """ Extracts a list of constraints for a motion primitive from ElementaryActionConstraints 
        based on the variables set by the method set_status. Generates constraints for path following.
    """
    def __init__(self):
        self.action_constraints = None
        self.algorithm_config = None
        self.status = {}
        self.motion_primitive_graph = None

    def set_action_constraints(self, action_constraints):
        self.action_constraints = action_constraints
        self.motion_primitive_graph = action_constraints.motion_primitive_graph
        self.node_group = self.action_constraints.get_node_group()
        self.skeleton = self.action_constraints.get_skeleton()

    def set_algorithm_config(self, algorithm_config):
        self.algorithm_config = algorithm_config
        self.precision = algorithm_config["constrained_gmm_settings"]["precision"]
        self.trajectory_following_settings = algorithm_config["trajectory_following_settings"]

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
            self._set_pose_constraint(mp_constraints)
        if len(self.action_constraints.keyframe_constraints.keys()) > 0:
            self._set_keyframe_constraints(mp_constraints)
            # generate frame constraints for the last step based on the previous state
            # if not already done for the trajectory following
            if self.status["is_last_step"] and not mp_constraints.pose_constraint_set:
                self._set_pose_constraint(mp_constraints)

        mp_constraints.use_optimization = len(self.action_constraints.keyframe_constraints.keys()) > 0\
                                            or self.status["is_last_step"]
        return mp_constraints

    def _set_pose_constraint(self, mp_constraints):
       if mp_constraints.settings["transition_pose_constraint_factor"] > 0.0 and self.status["prev_frames"] is not None:
            pose_constraint_desc = self._create_frame_constraint_from_preceding_motion()
            pose_constraint = PoseConstraint(self.skeleton, pose_constraint_desc, self.precision["smooth"], mp_constraints.settings["transition_pose_constraint_factor"])
            mp_constraints.constraints.append(pose_constraint)
            mp_constraints.pose_constraint_set = True

    def _set_trajectory_constraints(self, mp_constraints):
        print "search for new goal"
        # if it is the last step we need to reach the point exactly otherwise
        # make a guess for a reachable point on the path that we have not visited yet
        if not self.status["is_last_step"]:
            mp_constraints.goal_arc_length = self._make_guess_for_goal_arc_length()
        else:
            mp_constraints.goal_arc_length = self.action_constraints.trajectory.full_arc_length
        mp_constraints.step_goal,orientation,dir_vector = self._get_point_and_orientation_from_arc_length(mp_constraints.goal_arc_length)

        mp_constraints.print_status()


        root_joint_name = self.skeleton.root
        if mp_constraints.settings["position_constraint_factor"] > 0.0:
          keyframe_semantic_annotation={"firstFrame": None, "lastFrame": True}
          keyframe_constraint_desc = {"joint": root_joint_name, "position": mp_constraints.step_goal,
                      "semanticAnnotation": keyframe_semantic_annotation}
          keyframe_constraint = PositionAndRotationConstraint(self.skeleton, keyframe_constraint_desc, self.precision["pos"], mp_constraints.settings["position_constraint_factor"])
          mp_constraints.constraints.append(keyframe_constraint)
          
        if mp_constraints.settings["dir_constraint_factor"] > 0.0:
            dir_semantic_annotation={"firstFrame": None, "lastFrame": True}
            dir_constraint_desc = {"joint": root_joint_name, "dir_vector": dir_vector,
                          "semanticAnnotation": dir_semantic_annotation}
            direction_constraint = DirectionConstraint(self.skeleton, dir_constraint_desc, self.precision["rot"], mp_constraints.settings["dir_constraint_factor"])
            mp_constraints.constraints.append(direction_constraint)

    def _set_keyframe_constraints(self, mp_constraints):
        """ Extract keyframe constraints of the motion primitive name.
        """
        if self.status["motion_primitive_name"] in self.action_constraints.keyframe_constraints.keys():
            keyframe_constraint_desc_list = self.action_constraints.keyframe_constraints[self.status["motion_primitive_name"]]
            for i in xrange(len(keyframe_constraint_desc_list)):
                mp_constraints.constraints.append(PositionAndRotationConstraint(self.skeleton, keyframe_constraint_desc_list[i], self.precision["pos"], mp_constraints.settings["position_constraint_factor"]))

    def _create_frame_constraint_from_preceding_motion(self):
        """ Create frame a constraint from the preceding motion.
        """
        #last_euler_frame = np.ravel(convert_quaternion_to_euler([]))
        return MotionPrimitiveConstraintsBuilder.create_frame_constraint(self.skeleton, self.status["prev_frames"][-1])

    @classmethod
    def create_frame_constraint(cls, skeleton, frame):
        position_dict = {}
        for node_name in skeleton.node_name_map.keys():
            joint_position = get_cartesian_coordinates_from_quaternion(skeleton, node_name, frame)
            position_dict[node_name] = joint_position
        frame_constraint = {"frame_constraint": position_dict, "semanticAnnotation": {"firstFrame": True, "lastFrame": None}}
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
        node_key = (self.action_constraints.action_name, self.status["motion_primitive_name"])
        step_length = self.motion_primitive_graph.nodes[node_key].average_step_length\
                        * self.trajectory_following_settings["heuristic_step_length_factor"]
        max_arc_length = last_arc_length + 4.0 * step_length
        #find closest point in the range of the last_arc_length and max_arc_length
        closest_point,distance = self.action_constraints.trajectory.find_closest_point(last_pos, min_arc_length=last_arc_length, max_arc_length=max_arc_length)
        if closest_point is None:
            parameters = {"last":last_arc_length,"max":max_arc_length,"full": self.action_constraints.trajectory.full_arc_length}
            print "did not find closest point",closest_point,str(parameters)
            raise PathSearchError(parameters)
        # approximate arc length of the point closest to the current position
        start_arc_length,eval_point = self.action_constraints.trajectory.get_absolute_arc_length_of_point(closest_point, min_arc_length=last_arc_length)
        #update arc length based on the step length of the next motion primitive
        print start_arc_length + step_length, self.motion_primitive_graph.nodes[node_key].average_step_length
        if start_arc_length == -1:
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


