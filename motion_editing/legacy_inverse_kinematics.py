import time
from copy import copy
import numpy as np
from scipy.optimize import minimize
from transformations import quaternion_matrix, euler_from_matrix
from ..animation_data.motion_blending import smooth_joints_around_transition_using_slerp, create_transition_using_slerp
from .skeleton_pose_model import SkeletonPoseModel
from ..animation_data.constants import ROTATION_TYPE_EULER
from ..utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG

LEN_QUATERNION = 4
LEN_TRANSLATION = 3

IK_METHOD_UNCONSTRAINED_OPTIMIZATION = "unconstrained"
IK_METHOD_CYCLIC_COORDINATE_DESCENT = "ccd"


def obj_inverse_kinematics(s, data):
    pose, free_joints, target_joint, target_position, target_direction = data
    pose.set_channel_values(s, free_joints)
    if target_direction is not None:
        parent_joint = pose.get_parent_joint(target_joint)
        pose.point_in_direction(parent_joint, target_direction)
    position = pose.evaluate_position(target_joint)
    d = position - target_position
    return np.dot(d, d)


class LegacyInverseKinematics(object):
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

        if self.use_euler:
            self.skeleton.set_rotation_type(ROTATION_TYPE_EULER)
        self.pose = SkeletonPoseModel(self.skeleton, self.use_euler)

    def _run_optimization(self, objective, initial_guess, data, cons=None):
         return minimize(objective, initial_guess, args=(data,),
                        method=self._ik_settings["optimization_method"],#"SLSQP",#best result using L-BFGS-B
                        constraints=cons, tol=self._ik_settings["tolerance"],
                        options={'maxiter': self._ik_settings["max_iterations"], 'disp': self.verbose})#,'eps':1.0

    def _run_ccd(self, objective, initial_guess, data, cons=None):
        pose, free_joints, target_joint, target_position, target_orientation = data
        reversed_chain = copy(free_joints)
        reversed_chain.reverse()
        delta = np.inf
        epsilon = self._ik_settings["tolerance"]
        max_iter = self._ik_settings["max_iterations"]
        terminate = False
        iteration = 0
        start = time.clock()
        while not terminate:
            for free_joint in reversed_chain:
                self.optimize_joint(objective, target_joint, target_position, target_orientation, free_joint)
            position = pose.evaluate_position(target_joint)
            new_delta = np.linalg.norm(position-target_position)
            if delta < epsilon or abs(delta-new_delta) < epsilon or iteration > max_iter:
                terminate = True
            delta = new_delta
            iteration += 1
        if self.verbose:
            write_message_to_log("Finished optimization after " + str(iteration) + " iterations with error" +
                                 str(delta) + " in " + str(time.clock()-start) + " seconds", LOG_MODE_DEBUG)
        parameters = self._extract_free_parameters(free_joints)
        return parameters, delta

    def set_pose_from_frame(self, reference_frame):
        self.pose.set_pose_parameters(reference_frame)
        self.pose.clear_cache()

    def _modify_pose(self, joint_name, target, direction=None):
        error = 0.0
        if joint_name in list(self.pose.free_joints_map.keys()):
            free_joints = self.pose.free_joints_map[joint_name]
            if self.solving_method == IK_METHOD_CYCLIC_COORDINATE_DESCENT:
                error = self._modify_using_cyclic_coordinate_descent(joint_name, target, free_joints,direction)
            else:
                error = self._modify_using_optimization(joint_name, target, free_joints, direction)
        return error

    def _modify_pose_general(self, constraint):
        free_joints = constraint.free_joints
        initial_guess = self._extract_free_parameters(free_joints)
        data = constraint.data(self, free_joints)
        if self.verbose:
            self.pose.set_channel_values(initial_guess, free_joints)
            p = self.pose.evaluate_position(constraint.joint_name)
            write_message_to_log("Start optimization for joint " + constraint.joint_name +" " + str(len(initial_guess))
                                 + " " + str(len(free_joints)) + " " + str(p), LOG_MODE_DEBUG)
        cons = None#self.pose.generate_constraints(free_joints)
        if self.solving_method == IK_METHOD_UNCONSTRAINED_OPTIMIZATION:
            start = time.clock()
            error = np.inf
            iter_counter = 0
            result = None
            while error > self.success_threshold and iter_counter < self.max_retries:
                result = self._run_optimization(constraint.evaluate, initial_guess, data, cons)
                error = constraint.evaluate(result["x"], data)
                iter_counter += 1
            #write_log("finished optimization in",time.clock()-start,"seconds with error", error)#,result["x"].tolist(), initial_guess.tolist()
            if result is not None:
                self.pose.set_channel_values(result["x"], free_joints)
        else:
            parameters, error = self._run_ccd(constraint.evaluate, initial_guess, data, cons)
            self.pose.set_channel_values(parameters, free_joints)
        return error

    def _modify_using_optimization(self, target_joint, target_position, free_joints, target_direction=None):
        initial_guess = self._extract_free_parameters(free_joints)
        data = self.pose, free_joints, target_joint, target_position, target_direction
        if self.verbose:
            write_message_to_log("Start optimization for joint " + target_joint + " " +  str(len(initial_guess))
                                 + " " + str(len(free_joints)), LOG_MODE_DEBUG)
        start = time.clock()
        cons = None#self.pose.generate_constraints(free_joints)
        result = self._run_optimization(obj_inverse_kinematics, initial_guess, data, cons)
        position = self.pose.evaluate_position(target_joint)
        error = np.linalg.norm(position-target_position)
        if self.verbose:
            write_message_to_log("Finished optimization in " + str(time.clock()-start) + " seconds with error " + str(error), LOG_MODE_DEBUG) #,result["x"].tolist(), initial_guess.tolist()
        self.pose.set_channel_values(result["x"], free_joints)
        return error

    def optimize_joint(self, objective, target_joint, target_position, target_orientation, free_joint):
        initial_guess = self.pose.extract_parameters(free_joint)#self._extract_free_parameters([free_joint])
        data = self.pose, [free_joint], target_joint, target_position, None
        result = self._run_optimization(objective, initial_guess, data)
        #apply constraints here
        self.pose.apply_bounds(free_joint)
        return result["x"]

    def _modify_using_cyclic_coordinate_descent(self, target_joint, target_position, free_joints, target_direction=None):
        reversed_chain = copy(free_joints)
        reversed_chain.reverse()
        delta = np.inf
        epsilon = self._ik_settings["tolerance"]
        max_iter = self._ik_settings["max_iterations"]
        terminate = False
        iteration = 0
        start = time.clock()
        while not terminate:
            for free_joint in reversed_chain:
                self.optimize_joint(obj_inverse_kinematics, target_joint, target_position, None, free_joint)
                #self.pose.set_channel_values(joint_result, [free_joint])
            position = self.pose.evaluate_position(target_joint)
            new_delta = np.linalg.norm(position-target_position)
            if delta < epsilon or abs(delta-new_delta) < epsilon or iteration > max_iter:
                terminate = True
            delta = new_delta
            iteration += 1
        write_message_to_log(" ".join(map(str, ["Finished optimization after", iteration, "iterations with error", delta, "in", time.clock()-start, "seconds"])), LOG_MODE_DEBUG)
        return delta

    def evaluate_delta(self, parameters, target_joint, target_position, free_joints):
        self.pose.set_channel_values(parameters, free_joints)
        position = self.pose.evaluate_position(target_joint)
        d = position - target_position
        return np.dot(d, d)

    def modify_motion_vector(self, motion_vector):
        for idx, elementary_action_ik_constraints in enumerate(motion_vector.ik_constraints):
            write_message_to_log("Apply IK to elementary action " + str(idx), LOG_MODE_DEBUG)
            self._edit_action(motion_vector.frames, elementary_action_ik_constraints)

        if self.adapt_hands_during_both_hand_carry:
            self.adapt_hands_during_carry(motion_vector)

    def _edit_action(self, frames, ik_constraints):
        i = 0
        last_error = None
        keep_running = True
        trajectory_weights = 1.0
        # modify individual keyframes based on constraints
        while keep_running:
            error = 0.0
            if "trajectories" in list(ik_constraints.keys()):
                constraints = ik_constraints["trajectories"]
                c_error = self._modify_motion_vector_using_trajectory_constraint_list(frames,constraints)
                error += c_error * trajectory_weights
            if "keyframes" in list(ik_constraints.keys()):
                constraints = ik_constraints["keyframes"]
                error += self._edit_using_keyframe_constraint_list(frames, constraints)
            if last_error is not None:
                delta = abs(last_error - error)
            else:
                delta = np.inf
            last_error = error
            i += 1
            keep_running = i < self.elementary_action_max_iterations and delta > self.elementary_action_epsilon
            write_message_to_log("IK iteration " + str(i) + " " + str(error) + " " + str(delta) + " " + str(self.elementary_action_epsilon), LOG_MODE_DEBUG)

    def _edit_using_keyframe_constraint_list(self, frames, constraints):
        error = 0.0
        for frame_idx, constraints in list(constraints.items()):
            if "single" in list(constraints.keys()):
                for c in constraints["single"]:
                    if c.optimize:
                        if c.frame_range is not None:
                            error += self._edit_frame_range(frames, c, c.frame_range)
                        else:
                            error += self._edit_frame(frames, c, frame_idx)
                    if self.activate_look_at and c.look_at:
                        start = frame_idx
                        end = frame_idx+1
                        self._look_at_in_range(frames, c.position, start, end)
                        print("orientation constraint", c.orientation)
                        if c.orientation is not None and self.optimize_orientation:
                            self._set_hand_orientation(frames, c.orientation, c.joint_name, frame_idx, start, end)
        return error

    def _edit_frame(self, frames, constraint, keyframe):
        print("edit frame", keyframe)
        self.set_pose_from_frame(frames[keyframe])
        error = self._modify_pose_general(constraint)
        frames[keyframe] = self.pose.get_vector()
        if self.window > 0:
            self.interpolate_around_frame(frames, constraint.get_joint_names(), keyframe, self.window)
        return error

    def _edit_frame_range(self, frames, constraint, frame_range):
        error = 0.0
        for frame in range(frame_range[0],frame_range[1]+1):
            self.set_pose_from_frame(frames[frame])
            error += self._modify_pose_general(constraint)
            frames[frame] = self.pose.get_vector()

        self._create_transition_for_frame_range(frames,frame_range[0],frame_range[1], self.pose.free_joints_map[constraint.joint_name])
        return error

    def interpolate_around_frame(self, frames, joint_names, keyframe, window):
        write_message_to_log("Smooth and interpolate" + str(joint_names), LOG_MODE_DEBUG)
        for target_joint_name in joint_names:
            joint_parameter_indices = self._extract_free_parameter_indices(self.pose.free_joints_map[target_joint_name])
            for joint_name in self.pose.free_joints_map[target_joint_name]:
                smooth_joints_around_transition_using_slerp(frames, joint_parameter_indices[joint_name], keyframe, window)

    def _look_at_in_range(self, frames, position, start, end):
        start = max(0, start)
        end = min(frames.shape[0], end)
        for idx in range(start, end):
            self.set_pose_from_frame(frames[idx])
            self.pose.lookat(position)
            frames[idx] = self.pose.get_vector()
        self._create_transition_for_frame_range(frames, start, end-1, [self.pose.head_joint])

    def _create_transition_for_frame_range(self, frames, start, end, target_joints):
        for target_joint in target_joints:
            joint_parameter_indices = list(range(*self.pose.extract_parameters_indices(target_joint)))
            transition_start = max(start - self.transition_window, 0)
            transition_end = min(end + self.transition_window, frames.shape[0]) - 1
            create_transition_using_slerp(frames, transition_start, start, joint_parameter_indices)
            create_transition_using_slerp(frames, end, transition_end, joint_parameter_indices)

    def _modify_motion_vector_using_trajectory_constraint_list(self, frames, constraints):
        error = 0.0
        for c in constraints:
            if c["fixed_range"]:
                error += self._modify_motion_vector_using_trajectory_constraint(frames, c)
            else:
                error += self._modify_motion_vector_using_trajectory_constraint_search_start(frames, c)
        return error

    def _modify_motion_vector_using_trajectory_constraint(self, frames, traj_constraint):
        error_sum = 0.0
        d = traj_constraint["delta"]
        trajectory = traj_constraint["trajectory"]
        start_idx = traj_constraint["start_frame"]
        end_idx = traj_constraint["end_frame"]-1
        end_idx = min(len(frames)-1,end_idx)
        n_frames = end_idx-start_idx + 1
        target_direction = None
        if traj_constraint["constrain_orientation"]:
            target_direction = trajectory.get_direction()
            if np.linalg.norm(target_direction)==0:
                target_direction = None

        full_length = n_frames*d
        for idx in range(n_frames):
            t = (idx*d)/full_length
            target_position = trajectory.query_point_by_parameter(t)
            keyframe = start_idx+idx
            self.set_pose_from_frame(frames[keyframe])
            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._modify_pose(traj_constraint["joint_name"], target_position, target_direction)
                iter_counter += 1
            error_sum += error
            frames[keyframe] = self.pose.get_vector()
        parent_joint = self.pose.get_parent_joint(traj_constraint["joint_name"])

        if traj_constraint["joint_name"] in list(self.pose.free_joints_map.keys()):
            free_joints = self.pose.free_joints_map[traj_constraint["joint_name"]]
            free_joints = list(set(free_joints+[parent_joint]))
        else:
            free_joints = [parent_joint]
        self._create_transition_for_frame_range(frames, start_idx, end_idx, free_joints)
        return error_sum

    def _modify_motion_vector_using_trajectory_constraint_search_start(self, frames, traj_constraint):
        error_sum = 0.0
        trajectory = traj_constraint["trajectory"]
        start_target = trajectory.query_point_by_parameter(0.0)
        start_idx = self._find_corresponding_frame(frames,
                                                   traj_constraint["start_frame"],
                                                   traj_constraint["end_frame"],
                                                   traj_constraint["joint_name"],
                                                   start_target)
        n_frames = traj_constraint["end_frame"]-start_idx + 1
        arc_length = 0.0
        self.set_pose_from_frame(frames[start_idx])
        prev_position = self.pose.evaluate_position(traj_constraint["joint_name"])
        for idx in range(n_frames):
            keyframe = start_idx+idx
            self.set_pose_from_frame(frames[keyframe])
            current_position = self.pose.evaluate_position(traj_constraint["joint_name"])
            arc_length += np.linalg.norm(prev_position-current_position)
            prev_position = current_position
            if arc_length >= trajectory.full_arc_length:
                break
            target = trajectory.query_point_by_absolute_arc_length(arc_length)

            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._modify_pose(traj_constraint["joint_name"], target)
                iter_counter += 1
            error_sum += error
            frames[keyframe] = self.pose.get_vector()

        self._create_transition_for_frame_range(frames, start_idx, keyframe-1, self.pose.free_joints_map[traj_constraint["joint_name"]])
        return error_sum

    def _find_corresponding_frame_range(self, frames, traj_constraint):
        start_idx = traj_constraint["start_frame"]
        end_idx = traj_constraint["end_frame"]
        start_target = traj_constraint["trajectory"].query_point_by_parameter(0.0)
        end_target = traj_constraint["trajectory"].query_point_by_parameter(1.0)
        start_idx = self._find_corresponding_frame(frames, start_idx, end_idx, traj_constraint["joint_name"], start_target)
        end_idx = self._find_corresponding_frame(frames, start_idx, end_idx, traj_constraint["joint_name"], end_target)
        return start_idx, end_idx

    def _find_corresponding_frame(self, frames, start_idx, end_idx, target_joint, target_position):
        closest_start_frame = copy(start_idx)
        min_error = np.inf
        n_frames = end_idx-start_idx
        for idx in range(n_frames):
            keyframe = start_idx+idx
            self.set_pose_from_frame(frames[keyframe])
            position = self.pose.evaluate_position(target_joint)
            error = np.linalg.norm(position-target_position)
            if error <= min_error:
                min_error = error
                closest_start_frame = keyframe
        return closest_start_frame

    def _modify_motion_vector_using_ca_constraints(self, motion_vector, ca_constraints):
        error = 0.0
        for c in ca_constraints:
            start_frame = motion_vector.graph_walk.steps[c.step_idx].start_frame
            end_frame = motion_vector.graph_walk.steps[c.step_idx].end_frame
            keyframe = self._find_corresponding_frame(motion_vector.frames, start_frame, end_frame, c.joint_name, c.position)
            error += self._edit_frame(motion_vector.frames, c, keyframe)
        return error

    def _extract_free_parameters(self, free_joints):
        """get parameters of joints from reference frame
        """
        parameters = list()
        for joint_name in free_joints:
            parameters += self.pose.extract_parameters(joint_name).tolist()
        return np.asarray(parameters)

    def _extract_free_parameter_indices(self, free_joints):
        """get parameter indices of joints from reference frame
        """
        indices = {}
        for joint_name in free_joints:
            indices[joint_name] = list(range(*self.pose.extract_parameters_indices(joint_name)))
        return indices

    def _set_hand_orientation(self, frames, orientation, joint_name, keyframe, start, end):
        print("set hand orienation", keyframe)
        parent_joint_name = self.pose.get_parent_joint(joint_name)
        self.set_pose_from_frame(frames[keyframe])
        self.pose.set_hand_orientation(parent_joint_name, orientation)
        start = max(0, start)
        end = min(frames.shape[0], end)
        self._create_transition_for_frame_range(frames, start, end-1, [parent_joint_name])

    def _find_carry_frame_ranges(self, motion_vector):
        carrying = False
        frame_ranges = []
        last_frame = motion_vector.n_frames - 1
        for frame_idx in range(motion_vector.n_frames):
            if frame_idx in list(motion_vector.keyframe_event_list.keyframe_events_dict["events"].keys()):
                for event_desc in motion_vector.keyframe_event_list.keyframe_events_dict["events"][frame_idx]:
                    if event_desc["event"] == "attach" and (
                            event_desc["parameters"]["joint"] == ["RightHand", "LeftHand"]
                    or event_desc["parameters"]["joint"] == ["RightToolEndSite",
                                                             "LeftToolEndSite"]):
                        if carrying:
                            frame_ranges[-1][1] = min(frame_idx + 1, last_frame)
                        carrying = True
                        start = max(frame_idx - 1, 0)
                        frame_ranges.append([start, None])
                    elif carrying and event_desc["event"] == "detach" and (
                            event_desc["parameters"]["joint"] == ["RightHand", "LeftHand"]
                    or event_desc["parameters"]["joint"] == ["RightToolEndSite",
                                                             "LeftToolEndSite"]):
                        frame_ranges[-1][1] = min(frame_idx + 1, last_frame)
                        carrying = False
        return frame_ranges

    def adapt_hands_during_carry(self, motion_vector):
        frame_ranges = self._find_carry_frame_ranges(motion_vector)
        for frame_range in frame_ranges:
            if frame_range[1] is not None:
                right_free_joints = self.pose.reduced_free_joints_map["RightHand"]
                left_free_joints = self.pose.reduced_free_joints_map["LeftHand"]
                self._adapt_hand_positions_during_carry(motion_vector, frame_range, left_free_joints, right_free_joints)
                self._adapt_hand_orientations_during_carry(motion_vector, frame_range)

    def _adapt_hand_positions_during_carry(self, motion_vector, frame_region, left_free_joints, right_free_joints):
        action_index = motion_vector.graph_walk.get_action_from_keyframe(frame_region[0])
        end_of_pick_frame = motion_vector.keyframe_event_list.frame_annotation['elementaryActionSequence'][action_index]["endFrame"]
        if action_index + 1 >= len(motion_vector.keyframe_event_list.frame_annotation['elementaryActionSequence']):
            write_log("Warning: Could not adapt hand positions because there is no carry after pick.")
            return
        end_of_carry_frame = motion_vector.keyframe_event_list.frame_annotation['elementaryActionSequence'][action_index + 1]["endFrame"]
        self.pose.set_pose_parameters(motion_vector.frames[end_of_pick_frame])

        left_parameters = self._extract_free_parameters(left_free_joints)
        right_parameters = self._extract_free_parameters(right_free_joints)
        for idx in range(end_of_pick_frame + 1, end_of_carry_frame):
            self.pose.set_pose_parameters(motion_vector.frames[idx])
            self.pose.set_channel_values(left_parameters, left_free_joints)
            self.pose.set_channel_values(right_parameters, right_free_joints)
            motion_vector.frames[idx] = self.pose.get_vector()

        joint_names = list(set(right_free_joints + left_free_joints))
        self._create_transition_for_frame_range(motion_vector.frames, end_of_pick_frame+1, end_of_carry_frame-1, joint_names)

    def _adapt_hand_orientations_during_carry(self, motion_vector, frame_region):
        for frame in range(frame_region[0], frame_region[1]):
            self.pose.set_pose_parameters(motion_vector.frames[frame])
            self.pose.orient_hands_to_each_other()
            motion_vector.frames[frame] = self.pose.get_vector()
        print("create transition for frames", frame_region[0], frame_region[1])
        # joint_parameter_indices = list(range(*self.pose.extract_parameters_indices("RightHand")))
        # before = motion_vector.frames[:,joint_parameter_indices]
        self._create_transition_for_frame_range(motion_vector.frames, frame_region[0]+1, frame_region[1]-1, ["RightHand", "LeftHand"])

    def fill_rotate_events(self, motion_vector):
        for keyframe in list(motion_vector.keyframe_event_list.keyframe_events_dict["events"].keys()):
            keyframe = int(keyframe)
            for event in motion_vector.keyframe_event_list.keyframe_events_dict["events"][keyframe]:
                if event["event"] == "rotate":
                    joint_name = event["parameters"]["joint"]
                    orientation = event["parameters"]["globalOrientation"]
                    place_keyframe = event["parameters"]["referenceKeyframe"]
                    frames = motion_vector.frames[place_keyframe]
                    # compare delta with global hand orientation
                    joint_orientation = motion_vector.skeleton.nodes[joint_name].get_global_matrix(frames)
                    joint_orientation[:3, 3] = [0, 0, 0]
                    # test1 = np.rad2deg(euler_from_matrix(joint_orientation))
                    # test2 = quaternion_to_euler(orientation)
                    # assert np.all(test1 == test2), (test1, test2)
                    orientation_constraint = quaternion_matrix(orientation)
                    delta_orientation = np.dot(np.linalg.inv(joint_orientation),orientation_constraint)
                    euler = np.degrees(euler_from_matrix(delta_orientation))
                    #test_orientation = np.dot(joint_orientation, delta_orientation)
                    #assert np.all(orientation_constraint == test_orientation), (orientation_constraint,test_orientation)
                    # Alternative: bring constraint into local coordinate system of hand parent and then compare delta with local hand orientation
                    event["parameters"]["relativeOrientation"] = [euler[0], -euler[2], euler[1]]  # convert to CAD coordinate system
