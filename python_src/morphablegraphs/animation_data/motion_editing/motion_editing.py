from copy import copy
import numpy as np
from .numerical_ik_quat import NumericalInverseKinematicsQuat
from .skeleton_pose_model import SkeletonPoseModel
from ..motion_blending import smooth_quaternion_frames_using_slerp, apply_slerp
from ...external.transformations import quaternion_matrix, euler_from_matrix
from ...utilities.log import write_message_to_log, LOG_MODE_DEBUG


class MotionEditing(object):
    def __init__(self, skeleton, algorithm_settings):
        self.skeleton = skeleton
        self._ik_settings = algorithm_settings["inverse_kinematics_settings"]
        self.window = self._ik_settings["interpolation_window"]
        self.transition_window = self._ik_settings["transition_window"]
        self.verbose = False
        self.use_euler = self._ik_settings["use_euler_representation"]
        self.solving_method = self._ik_settings["solving_method"]
        self.success_threshold = self._ik_settings["success_threshold"]
        self.max_retries = self._ik_settings["max_retries"]
        self.activate_look_at = self._ik_settings["activate_look_at"]
        self.optimize_orientation = self._ik_settings["optimize_orientation"]
        self.elementary_action_max_iterations = self._ik_settings["elementary_action_max_iterations"]
        self.elementary_action_epsilon = self._ik_settings["elementary_action_optimization_eps"]
        self.adapt_hands_during_both_hand_carry = self._ik_settings["adapt_hands_during_carry_both"]
        self.pose = SkeletonPoseModel(self.skeleton, self.use_euler)
        self._ik = NumericalInverseKinematicsQuat(self.pose, self._ik_settings)

    def modify_motion_vector(self, motion_vector):
        for idx, action_ik_constraints in enumerate(motion_vector.ik_constraints):
            write_message_to_log("Apply IK to elementary action " + str(idx), LOG_MODE_DEBUG)
            self._optimize_action_ik_constraints(motion_vector, action_ik_constraints)

    def _optimize_action_ik_constraints(self, motion_vector, action_ik_constraints):
        i = 0
        last_error = None
        keep_running = True
        trajectory_weights = 1.0
        # modify individual keyframes based on constraints
        while keep_running:
            error = 0.0
            if "trajectories" in list(action_ik_constraints.keys()):
                constraints = action_ik_constraints["trajectories"]
                c_error = self._modify_motion_vector_using_trajectory_constraint_list(motion_vector, constraints)
                error += c_error * trajectory_weights
            if "keyframes" in list(action_ik_constraints.keys()):
                constraints = action_ik_constraints["keyframes"]
                error += self._modify_motion_vector_using_keyframe_constraint_list(motion_vector, constraints)
            if last_error is not None:
                delta = abs(last_error - error)
            else:
                delta = np.inf
            last_error = error
            i += 1
            keep_running = i < self.elementary_action_max_iterations and delta > self.elementary_action_epsilon
            write_message_to_log("IK iteration " + str(i) + " " + str(error) + " " + str(delta) + " " + str(
                self.elementary_action_epsilon), LOG_MODE_DEBUG)

    def _modify_motion_vector_using_keyframe_constraint_list(self, motion_vector, constraints):
        error = 0.0
        for keyframe, constraints in list(constraints.items()):
            if "single" in list(constraints.keys()):
                for c in constraints["single"]:
                    if c.optimize:
                        if c.frame_range is not None:
                            error += self._modify_motion_vector_using_keyframe_constraint_range(motion_vector, c,
                                                                                                c.frame_range)
                        else:
                            error += self._modify_frame_using_keyframe_constraint(motion_vector, c, keyframe)
                    if self.activate_look_at and c.look_at:
                        start = keyframe
                        end = keyframe + 1
                        self._look_at_in_range(motion_vector, c.position, start, end)
                        if c.orientation is not None and self.optimize_orientation:
                            self._set_hand_orientation(motion_vector, c.orientation, c.joint_name, keyframe, start, end)
        return error

    def _modify_frame_using_keyframe_constraint(self, motion_vector, constraint, keyframe):
        self.set_pose_from_frame(motion_vector.frames[keyframe])
        error = self._ik.modify_pose_general(constraint)
        motion_vector.frames[keyframe] = self.pose.get_vector()
        if self.window > 0:
            self.interpolate_around_keyframe(motion_vector.frames, constraint.get_joint_names(), keyframe, self.window)
        return error

    def _modify_motion_vector_using_keyframe_constraint_range(self, motion_vector, constraint, frame_range):
        error = 0.0
        for frame in range(frame_range[0], frame_range[1] + 1):
            self.set_pose_from_frame(motion_vector.frames[frame])
            error += self._ik.modify_pose_general(constraint)
            motion_vector.frames[frame] = self.pose.get_vector()

        self._create_transition_for_frame_range(motion_vector.frames, frame_range[0], frame_range[1],
                                                self.pose.free_joints_map[constraint.joint_name])
        return error

    def interpolate_around_keyframe(self, frames, joint_names, keyframe, window):
        write_message_to_log("Smooth and interpolate" + str(joint_names), LOG_MODE_DEBUG)
        for target_joint_name in joint_names:
            joint_parameter_indices = self._extract_free_parameter_indices(self.pose.free_joints_map[target_joint_name])
            for joint_name in self.pose.free_joints_map[target_joint_name]:
                smooth_quaternion_frames_using_slerp(frames, joint_parameter_indices[joint_name], keyframe, window)

    def _look_at_in_range(self, motion_vector, position, start, end):
        start = max(0, start)
        end = min(motion_vector.frames.shape[0], end)
        for idx in range(start, end):
            self.set_pose_from_frame(motion_vector.frames[idx])
            self.pose.lookat(position)
            motion_vector.frames[idx] = self.pose.get_vector()
        self._create_transition_for_frame_range(motion_vector.frames, start, end - 1, [self.pose.head_joint])

    def _create_transition_for_frame_range(self, frames, start, end, target_joints):
        for target_joint in target_joints:
            joint_parameter_indices = list(range(*self.pose.extract_parameters_indices(target_joint)))
            transition_start = max(start - self.transition_window, 0)
            transition_end = min(end + self.transition_window, frames.shape[0]) - 1
            apply_slerp(frames, transition_start, start, joint_parameter_indices)
            apply_slerp(frames, end, transition_end, joint_parameter_indices)

    def _set_hand_orientation(self, motion_vector, orientation, joint_name, keyframe, start, end):
        parent_joint_name = self.pose.get_parent_joint(joint_name)
        self.set_pose_from_frame(motion_vector.frames[keyframe])
        self.pose.set_hand_orientation(parent_joint_name, orientation)
        start = max(0, start)
        end = min(motion_vector.frames.shape[0], end)
        self._create_transition_for_frame_range(motion_vector.frames, start, end - 1, [parent_joint_name])

    def set_pose_from_frame(self, reference_frame):
        self.pose.set_pose_parameters(reference_frame)
        self.pose.clear_cache()

    def _extract_free_parameter_indices(self, free_joints):
        """get parameter indices of joints from reference frame
        """
        indices = {}
        for joint_name in free_joints:
            indices[joint_name] = list(range(*self.pose.extract_parameters_indices(joint_name)))
        return indices

    def _modify_motion_vector_using_trajectory_constraint_list(self, motion_vector, constraints):
        error = 0.0
        for c in constraints:
            if c["fixed_range"]:
                error += self._modify_motion_vector_using_trajectory_constraint(motion_vector, c)
            else:
                error += self._modify_motion_vector_using_trajectory_constraint_search_start(motion_vector, c)
        return error

    def _modify_motion_vector_using_trajectory_constraint(self, motion_vector, traj_constraint):
        error_sum = 0.0
        d = traj_constraint["delta"]
        trajectory = traj_constraint["trajectory"]
        start_idx = traj_constraint["start_frame"]
        end_idx = traj_constraint["end_frame"] - 1
        end_idx = min(len(motion_vector.frames) - 1, end_idx)
        n_frames = end_idx - start_idx + 1
        target_direction = None
        if traj_constraint["constrain_orientation"]:
            target_direction = trajectory.get_direction()
            if np.linalg.norm(target_direction) == 0:
                target_direction = None

        full_length = n_frames * d
        for idx in range(n_frames):
            t = (idx * d) / full_length
            target_position = trajectory.query_point_by_parameter(t)
            keyframe = start_idx + idx
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._ik.modify_pose(traj_constraint["joint_name"], target_position, target_direction)
                iter_counter += 1
            error_sum += error
            motion_vector.frames[keyframe] = self.pose.get_vector()
        parent_joint = self.pose.get_parent_joint(traj_constraint["joint_name"])

        if traj_constraint["joint_name"] in list(self.pose.free_joints_map.keys()):
            free_joints = self.pose.free_joints_map[traj_constraint["joint_name"]]
            free_joints = list(set(free_joints + [parent_joint]))
        else:
            free_joints = [parent_joint]
        self._create_transition_for_frame_range(motion_vector.frames, start_idx, end_idx, free_joints)
        return error_sum

    def _modify_motion_vector_using_trajectory_constraint_search_start(self, motion_vector, traj_constraint):
        error_sum = 0.0
        trajectory = traj_constraint["trajectory"]
        start_target = trajectory.query_point_by_parameter(0.0)
        start_idx = self._find_corresponding_frame(motion_vector,
                                                   traj_constraint["start_frame"],
                                                   traj_constraint["end_frame"],
                                                   traj_constraint["joint_name"],
                                                   start_target)
        n_frames = traj_constraint["end_frame"]-start_idx + 1
        arc_length = 0.0
        self.set_pose_from_frame(motion_vector.frames[start_idx])
        prev_position = self.pose.evaluate_position(traj_constraint["joint_name"])
        for idx in range(n_frames):
            keyframe = start_idx+idx
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            current_position = self.pose.evaluate_position(traj_constraint["joint_name"])
            arc_length += np.linalg.norm(prev_position-current_position)
            prev_position = current_position
            if arc_length >= trajectory.full_arc_length:
                break
            target = trajectory.query_point_by_absolute_arc_length(arc_length)

            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._ik.modify_pose(traj_constraint["joint_name"], target)
                iter_counter += 1
            error_sum += error
            motion_vector.frames[keyframe] = self.pose.get_vector()

        self._create_transition_for_frame_range(motion_vector.frames, start_idx, keyframe-1, self.pose.free_joints_map[traj_constraint["joint_name"]])
        return error_sum

    def _find_corresponding_frame(self, motion_vector, start_idx, end_idx, target_joint, target_position):
        closest_start_frame = copy(start_idx)
        min_error = np.inf
        n_frames = end_idx - start_idx
        for idx in range(n_frames):
            keyframe = start_idx + idx
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            position = self.pose.evaluate_position(target_joint)
            error = np.linalg.norm(position - target_position)
            if error <= min_error:
                min_error = error
                closest_start_frame = keyframe
        return closest_start_frame

    def fill_rotate_events(self, motion_vector):
        for keyframe in list(motion_vector.keyframe_event_list.keyframe_events_dict["events"].keys()):
            keyframe = int(keyframe)
            for event in motion_vector.keyframe_event_list.keyframe_events_dict["events"][keyframe]:
                if event["event"] == "rotate":
                    self.fill_rotate_event(motion_vector, event)

    def fill_rotate_event(self, motion_vector, event):
        joint_name = event["parameters"]["joint"]
        orientation = event["parameters"]["globalOrientation"]
        place_keyframe = event["parameters"]["referenceKeyframe"]
        frames = motion_vector.frames[place_keyframe]
        # compare delta with global hand orientation
        joint_orientation = motion_vector.skeleton.nodes[joint_name].get_global_matrix(frames)
        joint_orientation[:3, 3] = [0, 0, 0]
        orientation_constraint = quaternion_matrix(orientation)
        delta_orientation = np.dot(np.linalg.inv(joint_orientation), orientation_constraint)
        euler = np.degrees(euler_from_matrix(delta_orientation))
        # convert to CAD coordinate system
        event["parameters"]["relativeOrientation"] = [euler[0], -euler[2], euler[1]]
