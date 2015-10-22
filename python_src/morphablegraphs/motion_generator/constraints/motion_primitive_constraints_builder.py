# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:38:15 2015

@author: erhe01
"""

from copy import copy
import numpy as np
from ...utilities.exceptions import PathSearchError
from motion_primitive_constraints import MotionPrimitiveConstraints
from spatial_constraints.keyframe_constraints.pose_constraint import PoseConstraint
from spatial_constraints.keyframe_constraints.direction_constraint import DirectionConstraint
from spatial_constraints.keyframe_constraints.pos_and_rot_constraint import PositionAndRotationConstraint
from spatial_constraints.keyframe_constraints.pose_constraint_quat_frame import PoseConstraintQuatFrame
from spatial_constraints.keyframe_constraints.two_hand_constraint import TwoHandConstraintSet

OPTIMIZATION_MODE_ALL = "all"
OPTIMIZATION_MODE_KEYFRAMES = "keyframes"
OPTIMIZATION_MODE_NONE = "none"


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
        self.local_optimization_mode = algorithm_config["local_optimization_mode"]

    def set_status(self, motion_primitive_name, last_arc_length, prev_frames=None, is_last_step=False):
        self.status["motion_primitive_name"] = motion_primitive_name
        self.status["n_canonical_frames"] = self.motion_primitive_graph.nodes[
            (self.action_constraints.action_name, motion_primitive_name)].n_canonical_frames
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
        mp_constraints.motion_primitive_name = self.status["motion_primitive_name"]
        mp_constraints.settings = self.trajectory_following_settings
        mp_constraints.constraints = []
        mp_constraints.goal_arc_length = 0
        mp_constraints.step_start = self.status["last_pos"]
        mp_constraints.start_pose = self.action_constraints.start_pose
        mp_constraints.skeleton = self.action_constraints.get_skeleton()
        mp_constraints.precision = self.action_constraints.precision
        mp_constraints.verbose = self.algorithm_config["verbose"]
        if self.action_constraints.root_trajectory is not None:
            self._add_path_following_constraints(mp_constraints)
            self._add_pose_constraint(mp_constraints)
        if len(self.action_constraints.keyframe_constraints.keys()) > 0:
            self._add_keyframe_constraints(mp_constraints)
            # generate frame constraints for the last step based on the previous state
            # if not already done for the trajectory following
            if self.status["is_last_step"] and not mp_constraints.pose_constraint_set:
                self._add_pose_constraint(mp_constraints)
        self._add_trajectory_constraints(mp_constraints)
        self._decide_on_optimization(mp_constraints)
        return mp_constraints

    def _add_trajectory_constraints(self, mp_constraints):
        for trajectory_constraint in self.action_constraints.trajectory_constraints:
            # set the previous arc length as new min arc length
            if self.status["prev_frames"] is not None:
                trajectory_constraint.set_min_arc_length_from_previous_frames(self.status["prev_frames"])
                trajectory_constraint.set_number_of_canonical_frames(self.status["n_canonical_frames"])
            mp_constraints.constraints.append(trajectory_constraint)
        return

    def _add_pose_constraint(self, mp_constraints):
        if mp_constraints.settings["transition_pose_constraint_factor"] > 0.0 and self.status["prev_frames"] is not None:
            pose_constraint_desc = self._create_frame_constraint_from_preceding_motion()
            #pose_constraint_desc = self._create_frame_constraint_angular_from_preceding_motion()
            pose_constraint_desc = self._map_label_to_canonical_keyframe(pose_constraint_desc)
            pose_constraint = PoseConstraint(self.skeleton, pose_constraint_desc, self.precision["smooth"],
                                              mp_constraints.settings["transition_pose_constraint_factor"])
            #pose_constraint = PoseConstraintQuatFrame(self.skeleton, pose_constraint_desc, self.precision["smooth"],
            #                                          mp_constraints.settings["transition_pose_constraint_factor"])
            mp_constraints.constraints.append(pose_constraint)
            mp_constraints.pose_constraint_set = True

    def _add_pose_constraint_quat_frame(self, mp_constraints):
        pose_constraint_desc = self._create_frame_constraint_angular_from_preceding_motion()
        pose_constraint_quat_frame = PoseConstraintQuatFrame(self.skeleton, pose_constraint_desc,
                                                             self.precision["smooth"],
                                                             mp_constraints.settings["transition_pose_constraint_factor"])
        mp_constraints.constraints.append(pose_constraint_quat_frame)
        mp_constraints.pose_constraint_set = True

    def _add_path_following_constraints(self, mp_constraints):
        print "search for new goal"
        # if it is the last step we need to reach the point exactly otherwise
        # make a guess for a reachable point on the path that we have not visited yet
        if not self.status["is_last_step"]:
            mp_constraints.goal_arc_length = self._estimate_step_goal_arc_length()
        else:
            mp_constraints.goal_arc_length = self.action_constraints.root_trajectory.full_arc_length
        mp_constraints.step_goal, dir_vector = self._get_point_and_orientation_from_arc_length(mp_constraints.goal_arc_length)
        mp_constraints.print_status()
        self._add_path_following_goal_constraint(self.skeleton.root, mp_constraints, mp_constraints.step_goal)
        self._add_path_following_direction_constraint(self.skeleton.root, mp_constraints, dir_vector)

    def _add_path_following_goal_constraint(self, joint_name, mp_constraints, goal):
        if mp_constraints.settings["position_constraint_factor"] > 0.0:
            keyframe_semantic_annotation = {"keyframeLabel": "end", "generated": True}
            keyframe_constraint_desc = {"joint": joint_name,
                                        "position": goal,
                                        "semanticAnnotation": keyframe_semantic_annotation}
            keyframe_constraint_desc = self._map_label_to_canonical_keyframe(keyframe_constraint_desc)
            keyframe_constraint = PositionAndRotationConstraint(self.skeleton,
                                                                keyframe_constraint_desc,
                                                                self.precision["pos"],
                                                                mp_constraints.settings["position_constraint_factor"])
            mp_constraints.constraints.append(keyframe_constraint)

    def _add_path_following_direction_constraint(self, joint_name, mp_constraints, dir_vector):
        if mp_constraints.settings["dir_constraint_factor"] > 0.0:
            dir_semantic_annotation = {"keyframeLabel": "end", "generated": True}
            dir_constraint_desc = {"joint": joint_name, "dir_vector": dir_vector,
                                   "semanticAnnotation": dir_semantic_annotation}
            dir_constraint_desc = self._map_label_to_canonical_keyframe(dir_constraint_desc)
            direction_constraint = DirectionConstraint(self.skeleton, dir_constraint_desc, self.precision["rot"],
                                                       mp_constraints.settings["dir_constraint_factor"])
            mp_constraints.constraints.append(direction_constraint)

    def _add_keyframe_constraints(self, mp_constraints):
        """ Extract keyframe constraints of the motion primitive name.
        """
        if self.status["motion_primitive_name"] in self.action_constraints.keyframe_constraints.keys():
            keyframe_constraint_desc_list = self.action_constraints.keyframe_constraints[self.status["motion_primitive_name"]]
            for i in xrange(len(keyframe_constraint_desc_list)):
                if "merged" in keyframe_constraint_desc_list[i].keys():
                    keyframe_constraint_desc = self._map_label_to_canonical_keyframe(keyframe_constraint_desc_list[i])
                    keyframe_constraint = TwoHandConstraintSet(self.skeleton,
                                                                keyframe_constraint_desc,
                                                                self.precision["pos"], mp_constraints.settings["position_constraint_factor"])
                    self._add_events_to_event_list(mp_constraints, keyframe_constraint)
                    mp_constraints.constraints.append(keyframe_constraint)
                elif "position" in keyframe_constraint_desc_list[i].keys() \
                        or "orientation" in keyframe_constraint_desc_list[i].keys() \
                        or "time" in keyframe_constraint_desc_list[i].keys():
                    keyframe_constraint_desc = self._map_label_to_canonical_keyframe(keyframe_constraint_desc_list[i])
                    if keyframe_constraint_desc is not None:
                        keyframe_constraint = PositionAndRotationConstraint(self.skeleton,
                                                                            keyframe_constraint_desc,
                                                                            self.precision["pos"], mp_constraints.settings["position_constraint_factor"])
                        self._add_events_to_event_list(mp_constraints, keyframe_constraint)
                        mp_constraints.constraints.append(keyframe_constraint)

    def _decide_on_optimization(self, mp_constraints):
        if self.local_optimization_mode == OPTIMIZATION_MODE_ALL:
            mp_constraints.use_local_optimization = True
        elif self.local_optimization_mode == OPTIMIZATION_MODE_KEYFRAMES:
            mp_constraints.use_local_optimization = len(self.action_constraints.keyframe_constraints.keys()) > 0 \
                                                    or self.status["is_last_step"]
        else:
            mp_constraints.use_local_optimization = False

    def _add_events_to_event_list(self, mp_constraints, keyframe_constraint):
        if keyframe_constraint.keyframe_label in self.action_constraints.keyframe_annotations.keys():
            #simply overwrite it if it exists
            keyframe_event = {"canonical_keyframe": keyframe_constraint.canonical_keyframe,
                           "event_list":  self.action_constraints.keyframe_annotations[keyframe_constraint.keyframe_label]["annotations"]}
            mp_constraints.keyframe_event_list[keyframe_constraint.keyframe_label] = keyframe_event

    def _map_label_to_canonical_keyframe(self, keyframe_constraint_desc):
        """ Enhances the keyframe constraint definition with a canonical keyframe set based on label
            for a keyframe on the canonical timeline
        :param keyframe_constraint:
        :return: Enhanced keyframe description or None if label was not found
        """
        assert "keyframeLabel" in keyframe_constraint_desc["semanticAnnotation"].keys()
        keyframe_constraint_desc = copy(keyframe_constraint_desc)
        keyframe_constraint_desc["n_canonical_frames"] = self.status["n_canonical_frames"]
        keyframe_label = keyframe_constraint_desc["semanticAnnotation"]["keyframeLabel"]
        if keyframe_label == "end":
            keyframe_constraint_desc["canonical_keyframe"] = self.status["n_canonical_frames"]-1
        elif keyframe_label == "start":
            keyframe_constraint_desc["canonical_keyframe"] = 0
        else:
            annotations = self.motion_primitive_graph.node_groups[self.action_constraints.action_name].motion_primitive_annotations
            if self.status["motion_primitive_name"] in annotations.keys() and keyframe_label in annotations[self.status["motion_primitive_name"]].keys():
                keyframe = annotations[self.status["motion_primitive_name"]][keyframe_label]
                if keyframe == "-1" or keyframe == "lastFrame":# TODO set standard for keyframe values
                    keyframe = self.status["n_canonical_frames"]-1
                keyframe_constraint_desc["canonical_keyframe"] = int(keyframe)
            else:
                print "Error could not map keyframe label", keyframe_label, annotations.keys()
                return None
        return keyframe_constraint_desc

    def _create_frame_constraint_from_preceding_motion(self):
        """ Create frame a constraint from the preceding motion.
        """
        return MotionPrimitiveConstraintsBuilder.create_frame_constraint(self.skeleton, self.status["prev_frames"][-1])

    def _create_frame_constraint_angular_from_preceding_motion(self):
        return MotionPrimitiveConstraintsBuilder.create_frame_constraint_angular(self.status["prev_frames"][-1])

    @classmethod
    def create_frame_constraint(cls, skeleton, frame):
        frame_constraint = {"keyframeLabel": "start", "frame_constraint": skeleton.convert_quaternion_frame_to_cartesian_frame(frame),
                            "semanticAnnotation": {"keyframeLabel": "start"}}
        return frame_constraint

    @classmethod
    def create_frame_constraint_angular(cls, frame):
        frame_constraint = {"frame_constraint": frame, "keyframeLabel": "start", "semanticAnnotation": {"keyframeLabel": "start"}}
        return frame_constraint

    def _estimate_step_goal_arc_length(self):
        """ Makes a guess for a reachable arc length based on the current position.
            It searches for the closest point on the trajectory, retrieves the absolute arc length
            and its the arc length of a random sample of the next motion primitive
        Returns
        -------
        * arc_length : float
          The absolute arc length of the new goal on the trajectory.
          The goal should then be extracted using get_point_and_orientation_from_arc_length
        """
        node_key = (self.action_constraints.action_name, self.status["motion_primitive_name"])
        step_length = self.motion_primitive_graph.nodes[node_key].average_step_length \
                     * self.trajectory_following_settings["heuristic_step_length_factor"]
        # find closest point in the range of the last_arc_length and max_arc_length
        closest_point = self.find_closest_point_to_current_position_on_trajectory(step_length)
        # approximate arc length of the point closest to the current position
        start_arc_length, eval_point = self.action_constraints.root_trajectory.get_absolute_arc_length_of_point(closest_point)
        # update arc length based on the step length of the next motion primitive
        if start_arc_length == -1:
            return self.action_constraints.root_trajectory.full_arc_length
        else:
            return start_arc_length + step_length

    def find_closest_point_to_current_position_on_trajectory(self, step_length):
        max_arc_length = self.status["last_arc_length"] + step_length * 4.0
        closest_point, distance = self.action_constraints.root_trajectory.find_closest_point(self.status["last_pos"],
                                                                                             self.status["last_arc_length"],
                                                                                             max_arc_length)

        if closest_point is None:
            self._raise_closest_point_search_exception(max_arc_length)
        return closest_point

    def _get_point_and_orientation_from_arc_length(self, arc_length):
        """ Returns a point, an orientation and a direction vector on the trajectory
        """
        point = self.action_constraints.root_trajectory.query_point_by_absolute_arc_length(arc_length).tolist()
        reference_vector = np.array([0, 1])  # in z direction
        start, dir_vector, angle = self.action_constraints.root_trajectory.get_angle_at_arc_length_2d(arc_length,
                                                                                                      reference_vector)
        #orientation = [None, angle, None]
        for i in self.action_constraints.root_trajectory.unconstrained_indices:
            point[i] = None
            #orientation[i] = None
        return point, dir_vector

    def _raise_closest_point_search_exception(self, max_arc_length):
        parameters = {"last": self.status["last_arc_length"], "max": max_arc_length,
                       "full": self.action_constraints.root_trajectory.full_arc_length}
        print "did not find closest point",  str(parameters)
        raise PathSearchError(parameters)
