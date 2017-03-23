# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:38:15 2015

@author: erhe01
"""

from copy import copy
import numpy as np
from ..utilities.exceptions import PathSearchError
from .motion_primitive_constraints import MotionPrimitiveConstraints
from .spatial_constraints import PoseConstraint, Direction2DConstraint, GlobalTransformConstraint, PoseConstraintQuatFrame, TwoHandConstraintSet, LookAtConstraint, FeetConstraint
from ..animation_data.motion_vector import concatenate_frames
from ..animation_data.motion_editing import get_2d_pose_transform, inverse_pose_transform, fast_quat_frames_transformation, create_transformation_matrix
from . import CA_CONSTRAINTS_MODE_SET, OPTIMIZATION_MODE_ALL, OPTIMIZATION_MODE_KEYFRAMES, OPTIMIZATION_MODE_TWO_HANDS
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from ..motion_model.elementary_action_meta_info import KEYFRAME_LABEL_END, KEYFRAME_LABEL_START, KEYFRAME_LABEL_MIDDLE
from keyframe_event import KeyframeEvent

class MotionPrimitiveConstraintsBuilder(object):
    """ Extracts a list of constraints for a motion primitive from ElementaryActionConstraints 
        based on the variables set by the method set_status. Generates constraints for path following.
    """

    mp_constraint_types = ["position", "orientation", "time"]
    def __init__(self):
        self.action_constraints = None
        self.algorithm_config = None
        self.status = {}
        self.motion_state_graph = None
        self.node_group = None
        self.skeleton = None
        self.precision = 1.0
        self.trajectory_following_settings = None
        self.local_optimization_mode = "None"
        self.ca_constraint_mode = "None"
        self.use_local_coordinates = False
        self.use_transition_constraint = False
    
    def set_action_constraints(self, action_constraints):
        self.action_constraints = action_constraints
        self.motion_state_graph = action_constraints.motion_state_graph
        self.node_group = self.action_constraints.get_node_group()
        self.skeleton = action_constraints.motion_state_graph.skeleton

    def set_algorithm_config(self, algorithm_config):
        self.algorithm_config = algorithm_config
        self.precision = algorithm_config["constrained_gmm_settings"]["precision"]
        self.trajectory_following_settings = algorithm_config["trajectory_following_settings"]
        self.local_optimization_mode = algorithm_config["local_optimization_mode"]
        self.ca_constraint_mode = algorithm_config["collision_avoidance_constraints_mode"]
        self.use_local_coordinates = algorithm_config["use_local_coordinates"]
        self.use_mgrd = algorithm_config["constrained_sampling_mode"] == "random_spline"
        self.use_transition_constraint = self.trajectory_following_settings["use_transition_constraint"]

    def set_status(self, node_key, last_arc_length, graph_walk, is_last_step=False):
        n_prev_frames = graph_walk.get_num_of_frames()
        prev_frames = graph_walk.get_quat_frames()
        n_canonical_frames = self.motion_state_graph.nodes[node_key].get_n_canonical_frames()
        #create a sample to estimate the trajectory arc lengths
        mp_sample_frames = self.motion_state_graph.nodes[node_key].sample(False).get_motion_vector()
        if self.use_local_coordinates:
            aligned_sample_frames = concatenate_frames(prev_frames, mp_sample_frames, graph_walk.motion_vector.start_pose, graph_walk.motion_vector.rotation_type, apply_spatial_smoothing=False)
            self.status["aligned_sample_frames"] = aligned_sample_frames[n_prev_frames:]
        self.status["action_name"] = node_key[0]
        self.status["motion_primitive_name"] = node_key[1]
        self.status["n_canonical_frames"] = n_canonical_frames
        self.status["last_arc_length"] = last_arc_length # defined in actionstate.transition() based on the closest point on the path

        if prev_frames is None:
            last_pos = self.action_constraints.start_pose["position"]
        else:
            last_pos = prev_frames[-1][:3]
        last_pos = copy(last_pos)
        last_pos[1] = 0.0
        self.status["last_pos"] = last_pos
        self.status["prev_frames"] = prev_frames
        self.status["is_last_step"] = is_last_step
        if self.use_mgrd or self.use_local_coordinates:
            self._set_aligning_transform(node_key, prev_frames)
        else:
            self.status["aligning_transform"] = None

    def _set_aligning_transform(self, node_key, prev_frames):
        if prev_frames is None:
            #print "create aligning transform from start pose",self.action_constraints.start_pose
            transform = copy(self.action_constraints.start_pose)
        else:
            aligning_angle, aligning_offset = fast_quat_frames_transformation(prev_frames, self.motion_state_graph.nodes[node_key].sample(False).get_motion_vector()) #TODO return from concatenate_frames
            transform = {"position": aligning_offset,"orientation":[0,aligning_angle,0]}

        self.status["aligning_transform"] = create_transformation_matrix(transform["position"], transform["orientation"])

    def build(self):
        mp_constraints = MotionPrimitiveConstraints()
        mp_constraints.action_name = self.status["action_name"]
        mp_constraints.motion_primitive_name = self.status["motion_primitive_name"]
        mp_constraints.aligning_transform = self.status["aligning_transform"]
        mp_constraints.is_last_step = self.status["is_last_step"]
        mp_constraints.settings = self.trajectory_following_settings
        mp_constraints.constraints = list()
        mp_constraints.goal_arc_length = 0.0
        mp_constraints.step_start = self.status["last_pos"]
        if self.use_local_coordinates:
            mp_constraints.start_pose = None
        else:
            mp_constraints.start_pose = self.action_constraints.start_pose
        mp_constraints.skeleton = self.skeleton
        mp_constraints.precision = self.action_constraints.precision
        mp_constraints.verbose = self.algorithm_config["verbose"]
        if self.action_constraints.root_trajectory is not None:
            self._add_path_following_constraints(mp_constraints)
            if self.use_transition_constraint:
                self._add_pose_constraint(mp_constraints)
        if len(self.action_constraints.keyframe_constraints.keys()) > 0:
            self._add_keyframe_constraints(mp_constraints)
            # generate frame constraints for the last step based on the previous state
            # if not already done for the trajectory following
            if self.status["is_last_step"] and not mp_constraints.pose_constraint_set:
                self._add_pose_constraint(mp_constraints)

        if mp_constraints.action_name in ["pickBoth","placeBoth"] and mp_constraints.motion_primitive_name == "reach":
            self._add_feet_constraint(mp_constraints)

        self._add_trajectory_constraints(mp_constraints)
        self._add_events_to_event_list(mp_constraints)
        self._decide_on_optimization(mp_constraints)
        return mp_constraints

    def _add_trajectory_constraints(self, mp_constraints):
        for trajectory_constraint in self.action_constraints.trajectory_constraints:
            # set the previous arc length as new min arc length
            if self.status["prev_frames"] is not None:
                trajectory_constraint.set_min_arc_length_from_previous_frames(self.status["prev_frames"])
                trajectory_constraint.set_number_of_canonical_frames(self.status["n_canonical_frames"])
            mp_constraints.constraints.append(trajectory_constraint)
        if self.ca_constraint_mode == CA_CONSTRAINTS_MODE_SET and self.action_constraints.ca_trajectory_set_constraint is not None:
            ca_trajectory_set_constraint = copy(self.action_constraints.ca_trajectory_set_constraint)
            ca_trajectory_set_constraint.set_min_arc_length_from_previous_frames(self.status["prev_frames"])
            ca_trajectory_set_constraint.set_number_of_canonical_frames(self.status["n_canonical_frames"])
            mp_constraints.constraints.append(ca_trajectory_set_constraint)
            #TODO generate discrete ca constraint where frames are fixed as inside or outside of the range
        #    for i in xrange(len(ca_trajectory_set_constraint.joint_trajectories)):
        #       discrete_trajectory_constraint = ca_trajectory_set_constraint.joint_trajectories[i].create_discrete_trajectory(self.status["aligned_sample_frames"])
        #       discrete_trajectory_constraint.set_min_arc_length_from_previous_frames(self.status["prev_frames"])
        #       mp_constraints.constraints.append(discrete_trajectory_constraint)
        #       ca_trajectory_set_constraint.joint_trajectories[i].set_min_arc_length_from_previous_frames(self.status["prev_frames"])
        #       mp_constraints.ca_constraints.append(ca_trajectory_set_constraint.joint_trajectories[i])

    def _add_feet_constraint(self, mp_constraints):
        if "LeftFoot" in self.skeleton.nodes.keys() and "RightFoot" in self.skeleton.nodes.keys():
            left_position = self.skeleton.nodes["LeftFoot"].get_global_position(self.status["prev_frames"][-1])
            right_position = self.skeleton.nodes["RightFoot"].get_global_position(self.status["prev_frames"][-1])
            desc = {"left":left_position, "right": right_position}
            desc["semanticAnnotation"] = {}
            desc["semanticAnnotation"]["keyframeLabel"] = "end"
            desc["canonical_keyframe"] = self._get_keyframe_from_annotation("end")
            feet_constraint = FeetConstraint(self.skeleton, desc, 1.0, 2.0)
            mp_constraints.constraints.append(feet_constraint)

    def _add_pose_constraint(self, mp_constraints):
        if mp_constraints.settings["transition_pose_constraint_factor"] > 0.0 and self.status["prev_frames"] is not None:
            pose_constraint_desc = self._create_pose_constraint_from_preceding_motion()
            #pose_constraint_desc = self._create_pose_constraint_angular_from_preceding_motion()
            pose_constraint_desc = self._map_label_to_canonical_keyframe(pose_constraint_desc)
            pose_constraint = PoseConstraint(self.skeleton, pose_constraint_desc, self.precision["smooth"],
                                              mp_constraints.settings["transition_pose_constraint_factor"])
            mp_constraints.constraints.append(pose_constraint)
            mp_constraints.pose_constraint_set = True

    def _add_pose_constraint_quat_frame(self, mp_constraints):
        pose_constraint_desc = self._create_pose_constraint_angular_from_preceding_motion()
        pose_constraint_quat_frame = PoseConstraintQuatFrame(self.skeleton, pose_constraint_desc,
                                                             self.precision["smooth"],
                                                             mp_constraints.settings["transition_pose_constraint_factor"])
        mp_constraints.constraints.append(pose_constraint_quat_frame)
        mp_constraints.pose_constraint_set = True

    def _add_path_following_constraints(self, mp_constraints):
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

    def _get_approximate_step_length(self):
        node_key = (self.action_constraints.action_name, self.status["motion_primitive_name"])
        return self.motion_state_graph.nodes[node_key].average_step_length * self.trajectory_following_settings["heuristic_step_length_factor"]

    def _add_path_following_goal_constraint(self, joint_name, mp_constraints, goal, keyframeLabel="end"):
        if mp_constraints.settings["position_constraint_factor"] > 0.0:
            keyframe_semantic_annotation = {"keyframeLabel": keyframeLabel, "generated": True}
            keyframe_constraint_desc = {"joint": joint_name,
                                        "position": goal,
                                        "semanticAnnotation": keyframe_semantic_annotation}
            keyframe_constraint_desc = self._map_label_to_canonical_keyframe(keyframe_constraint_desc)
            keyframe_constraint = GlobalTransformConstraint(self.skeleton,
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
            direction_constraint = Direction2DConstraint(self.skeleton, dir_constraint_desc, self.precision["rot"],
                                                         mp_constraints.settings["dir_constraint_factor"])
            mp_constraints.constraints.append(direction_constraint)

    def _add_keyframe_constraints(self, mp_constraints):
        """ Extract keyframe constraints of the motion primitive name.
        """
        if self.status["motion_primitive_name"] in self.action_constraints.keyframe_constraints.keys():
            for c_desc in self.action_constraints.keyframe_constraints[self.status["motion_primitive_name"]]:
                keyframe_constraint = self.create_keyframe_constraint(c_desc)
                if keyframe_constraint is not None:
                    mp_constraints.constraints.append(keyframe_constraint)

    def create_keyframe_constraint(self, c_desc):
        if "keyframeLabel" in c_desc["semanticAnnotation"].keys():
            c_desc = self._map_label_to_canonical_keyframe(c_desc)
            constraint_factor = self.trajectory_following_settings["position_constraint_factor"]
            if "merged" in c_desc.keys():
                return TwoHandConstraintSet(self.skeleton, c_desc, self.precision["pos"], constraint_factor)
            elif "look_at" in c_desc.keys():
                return LookAtConstraint(self.skeleton, c_desc, self.precision["pos"], constraint_factor)
            else:
                return GlobalTransformConstraint(self.skeleton, c_desc, self.precision["pos"], constraint_factor)
        else:
            return None

    def _decide_on_optimization(self, mp_constraints):
        if self.local_optimization_mode == OPTIMIZATION_MODE_ALL:
            mp_constraints.use_local_optimization = True
        elif self.local_optimization_mode == OPTIMIZATION_MODE_KEYFRAMES:
            mp_constraints.use_local_optimization = len(self.action_constraints.keyframe_constraints.keys()) > 0 \
                                                    or self.status["is_last_step"]
        elif self.local_optimization_mode == OPTIMIZATION_MODE_TWO_HANDS:
            mp_constraints.use_local_optimization = self.action_constraints.contains_two_hands_constraints and not self.status["is_last_step"]
        else:
            mp_constraints.use_local_optimization = False

    def _add_events_to_event_list(self, mp_constraints):
        for label in self.action_constraints.keyframe_annotations.keys():
            print "try to set annotations for label ", label
            if mp_constraints.motion_primitive_name in self.motion_state_graph.node_groups[self.action_constraints.action_name].motion_primitive_annotations.keys():
                if label in self.motion_state_graph.node_groups[self.action_constraints.action_name].motion_primitive_annotations[mp_constraints.motion_primitive_name]:
                    event_list = self.action_constraints.keyframe_annotations[label]["annotations"]

                    # add keyframe constraint based on joint and label
                    constraint = None
                    if len(event_list) == 1:#only if there is only one constraint on one joint otherwise correspondence is not clear
                        joint_name = event_list[0]["parameters"]["joint"]
                        for c in mp_constraints.constraints:
                            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and c.joint_name == joint_name and c.keyframe_label == label :
                                constraint = c
                                break

                    mp_constraints.keyframe_event_list[label] = KeyframeEvent(label, self._get_keyframe_from_annotation(label),event_list, constraint)

    def _map_label_to_canonical_keyframe(self, keyframe_constraint_desc):
        """ Enhances the keyframe constraint definition with a canonical keyframe set based on label
            for a keyframe on the canonical timeline
        :param keyframe_constraint:
        :return: Enhanced keyframe description or None if label was not found
        """
        #assert "keyframeLabel" in keyframe_constraint_desc["semanticAnnotation"].keys()
        keyframe_constraint_desc = copy(keyframe_constraint_desc)
        keyframe_constraint_desc["n_canonical_frames"] = self.status["n_canonical_frames"]
        keyframe_label = keyframe_constraint_desc["semanticAnnotation"]["keyframeLabel"]
        keyframe = self._get_keyframe_from_annotation(keyframe_label)
        if keyframe is not None:
            keyframe_constraint_desc["canonical_keyframe"] = keyframe
        else:
            return None
        return keyframe_constraint_desc

    def _get_keyframe_from_annotation(self, keyframe_label):
        return self.motion_state_graph.node_groups[self.action_constraints.action_name]. \
            get_keyframe_from_annotation(self.status["motion_primitive_name"], keyframe_label,
                                         self.status["n_canonical_frames"])

    def _create_pose_constraint_from_preceding_motion(self):
        """ Create frame a constraint from the preceding motion.
        """
        return MotionPrimitiveConstraintsBuilder.create_pose_constraint(self.skeleton, self.status["prev_frames"][-1])

    def _create_pose_constraint_angular_from_preceding_motion(self):
        return MotionPrimitiveConstraintsBuilder.create_pose_constraint_angular(self.status["prev_frames"][-1])

    @classmethod
    def create_pose_constraint(cls, skeleton, frame):
        frame_constraint = {"keyframeLabel": "start",
                            "frame_constraint": skeleton.convert_quaternion_frame_to_cartesian_frame(frame),
                            "semanticAnnotation": {"keyframeLabel": "start"}}
        return frame_constraint

    @classmethod
    def create_pose_constraint_angular(cls, frame):
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
        step_length = self._get_approximate_step_length()
        # find closest point in the range of the last_arc_length and max_arc_length
        #closest_point = self.find_closest_point_to_current_position_on_trajectory(step_length)
        # approximate arc length of the point closest to the current position
        #start_arc_length, eval_point = self.action_constraints.root_trajectory.get_absolute_arc_length_of_point(closest_point)

        start_arc_length = self.status["last_arc_length"] #last arc length is already found as closest point on path to current position
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
        reference_vector = np.array([0.0, 1.0])  # is interpreted as x, z
        start, dir_vector, angle = self.action_constraints.root_trajectory.get_angle_at_arc_length_2d(arc_length,
                                                                                                      reference_vector)
        for i in self.action_constraints.root_trajectory.unconstrained_indices:
            point[i] = None
        return point, dir_vector

    def _raise_closest_point_search_exception(self, max_arc_length):
        parameters = {"last": self.status["last_arc_length"], "max": max_arc_length,
                       "full": self.action_constraints.root_trajectory.full_arc_length}
        print "Error: Did not find closest point", str(parameters)
        raise PathSearchError(parameters)