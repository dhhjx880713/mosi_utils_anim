import numpy as np
from copy import copy
import time
from scipy.optimize import minimize
from collections import OrderedDict
from ...animation_data import ROTATION_TYPE_EULER,ROTATION_TYPE_QUATERNION
from blending import smooth_quaternion_frames_using_slerp, smooth_quaternion_frames_using_slerp_overwrite_frames, apply_slerp
from skeleton_pose_model import SkeletonPoseModel
from ...utilities import write_log


LEN_QUATERNION = 4
LEN_TRANSLATION = 3

IK_METHOD_UNCONSTRAINED_OPTIMIZATION = "unconstrained"
IK_METHOD_CYCLIC_COORDINATE_DESCENT = "ccd"


def obj_inverse_kinematics(s, data):
    pose, free_joints, target_joint, target_position, target_orientation = data
    #d = ik.evaluate_delta(s, target_joint, target_position, free_joints)
    #return d
    pose.set_channel_values(s, free_joints) #update frame
    position = pose.evaluate_position(target_joint)
    d = position - target_position
    #print target_joint, position, target_position, parameters
    #print parameters.tolist()
    return np.dot(d, d)



class InverseKinematics(object):
    def __init__(self, skeleton, algorithm_settings, reference_frame):
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
        if self.use_euler:
            self.skeleton.set_rotation_type(ROTATION_TYPE_EULER)#change to euler
        self.channels = OrderedDict()
        for node in self.skeleton.nodes.values():
            if node.node_name in skeleton.animated_joints:
                node_channels = copy(node.channels)
                if not self.use_euler:
                    if np.all([ch in node_channels for ch in ["Xrotation", "Yrotation", "Zrotation"]]):
                        node_channels += ["Wrotation"] #TODO fix order
                self.channels[node.node_name] = node_channels
        self.pose = SkeletonPoseModel(self.skeleton, reference_frame, self.channels, self.use_euler)

    def _run_optimization(self, objective, initial_guess, data, cons=None):
         return minimize(objective, initial_guess, args=(data,),
                        method=self._ik_settings["optimization_method"],#"SLSQP",#best result using L-BFGS-B
                        constraints=cons, tol=self._ik_settings["tolerance"],
                        options={'maxiter': self._ik_settings["max_iterations"], 'disp': self.verbose})#,'eps':1.0

    def set_pose_from_frame(self, reference_frame):
        self.pose.set_pose_parameters(reference_frame)
        self.pose.clear_cache()

    def _modify_pose(self, joint_name, target):
        if joint_name in self.pose.free_joints_map.keys():
            free_joints = self.pose.free_joints_map[joint_name]
            if self.solving_method == IK_METHOD_CYCLIC_COORDINATE_DESCENT:
                error = self._modify_using_cyclic_coordinate_descent(joint_name, target, free_joints)
            else:
                error = self._modify_using_optimization(joint_name, target, free_joints)
        return error

    def _modify_pose_general(self, constraint):
        free_joints = constraint.free_joints
        initial_guess = self._extract_free_parameters(free_joints)
        data = constraint.data(self, free_joints)
        write_log("start optimization for joint", len(initial_guess), len(free_joints))
        start = time.clock()
        cons = None#self.pose.generate_constraints(free_joints)
        error = np.inf
        iter_counter = 0
        while error > self.success_threshold and iter_counter < self.max_retries:
            result = self._run_optimization(constraint.evaluate, initial_guess, data, cons)
            error = constraint.evaluate(result["x"], data)
            iter_counter += 1
        write_log("finished optimization in",time.clock()-start,"seconds with error", error)#,result["x"].tolist(), initial_guess.tolist()
        self.pose.set_channel_values(result["x"], free_joints)
        return error

    def _modify_using_optimization(self, target_joint, target_position, free_joints):
        initial_guess = self._extract_free_parameters(free_joints)
        data = self, free_joints, target_joint, target_position, None
        if self.verbose:
            write_log("start optimization for joint", target_joint, len(initial_guess), len(free_joints))
        start = time.clock()
        cons = None#self.pose.generate_constraints(free_joints)
        result = self._run_optimization(obj_inverse_kinematics, initial_guess, data, cons)
        position = self.pose.evaluate_position(target_joint)
        error = np.linalg.norm(position-target_position)
        write_log("finished optimization in",time.clock()-start, "seconds with error", error) #,result["x"].tolist(), initial_guess.tolist()
        self.pose.set_channel_values(result["x"], free_joints)
        return error

    def optimize_joint(self, objective, target_joint, target_position, target_orientation, free_joint):
        initial_guess = self.pose.extract_parameters(free_joint)#self._extract_free_parameters([free_joint])
        data = self.pose, [free_joint], target_joint, target_position, None
        result = self._run_optimization(objective, initial_guess, data)
        #apply constraints here
        self.pose.apply_bounds(free_joint)
        return result["x"]

    def _modify_using_cyclic_coordinate_descent(self, target_joint, target_position, free_joints):
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
        write_log("Finished optimization after", iteration, "iterations with error", delta, "in", time.clock()-start, "seconds")
        return delta

    def evaluate_delta(self, parameters, target_joint, target_position, free_joints):
        self.pose.set_channel_values(parameters, free_joints) #update frame
        position = self.pose.evaluate_position(target_joint)
        d = position - target_position
        #print target_joint, position, target_position, parameters
        #print parameters.tolist()
        return np.dot(d, d)

    def modify_motion_vector(self, motion_vector):
        #modify individual keyframes based on constraints
        if "keyframes" in motion_vector.ik_constraints.keys():
            self._modify_motion_vector_using_keyframe_constraint_list(motion_vector, motion_vector.ik_constraints["keyframes"])
        if "trajectories" in motion_vector.ik_constraints.keys():
            self._modify_motion_vector_using_trajectory_constraint_list(motion_vector, motion_vector.ik_constraints["trajectories"])
        if "collision_avoidance" in motion_vector.ik_constraints.keys():
            self._modify_motion_vector_using_ca_constraints(motion_vector, motion_vector.ik_constraints["collision_avoidance"])

    def _modify_motion_vector_using_keyframe_constraint_list(self, motion_vector, constraints):
        write_log("number of ik keyframe constraints", len(constraints))
        has_multiple_targets = False
        for keyframe, constraints in constraints.items():
            #write_log(keyframe, constraints)
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            if "multiple" in constraints.keys():
                for c in constraints["multiple"]:
                    #self._modify_frame_using_keyframe_constraint(motion_vector, c, keyframe)
                    has_multiple_targets = True
            if "single" in constraints.keys():
                for c in constraints["single"]:
                    self._modify_frame_using_keyframe_constraint(motion_vector, c, keyframe)
                    if self.activate_look_at and not has_multiple_targets:
                        #write_log("look at constraint")
                        start = keyframe
                        end = keyframe+1
                        self._look_at_in_range(motion_vector, c.position, start, end)
                        if c.orientation is not None:
                            self._set_hand_orientation(motion_vector, c.orientation, c.joint_name, keyframe, start, end)

    def _modify_frame_using_keyframe_constraint(self, motion_vector, constraint, keyframe):
        self._modify_pose_general(constraint)
        motion_vector.frames[keyframe] = self.pose.get_vector()
        if self.window > 0:
            self.interpolate_around_keyframe(motion_vector.frames, constraint.get_joint_names(), keyframe, self.window)

    def interpolate_around_keyframe(self, frames, joint_names, keyframe, window):
        write_log("Smooth and interpolate", joint_names)
        for target_joint_name in joint_names:
            joint_parameter_indices = self._extract_free_parameter_indices(self.pose.free_joints_map[target_joint_name])
            for joint_name in self.pose.free_joints_map[target_joint_name]:
                smooth_quaternion_frames_using_slerp(frames, joint_parameter_indices[joint_name], keyframe, window)

    def _look_at_in_range(self, motion_vector, position, start, end):
        start = max(0, start)
        end = min(motion_vector.frames.shape[0], end)
        for idx in xrange(start, end):
            self.set_pose_from_frame(motion_vector.frames[idx])
            self.pose.lookat(position)
            motion_vector.frames[idx] = self.pose.get_vector()
        self._create_transition_for_frame_range(motion_vector.frames, start, end-1, [self.pose.head_joint])

    def _create_transition_for_frame_range(self, frames, start, end, target_joints):
        for target_joint in target_joints:
            joint_parameter_indices = list(range(*self.pose.extract_parameters_indices(target_joint)))
            transition_start = max(start - self.transition_window, 0)
            transition_end = min(end + self.transition_window, frames.shape[0]) - 1
            apply_slerp(frames, transition_start, start, joint_parameter_indices)
            apply_slerp(frames, end, transition_end, joint_parameter_indices)

    def _modify_motion_vector_using_trajectory_constraint_list(self, motion_vector, constraints):
        write_log("Number of ik trajectory constraints", len(constraints))
        for c in constraints:
            self._modify_motion_vector_using_trajectory_constraint2(motion_vector, c)

    def _modify_motion_vector_using_trajectory_constraint(self, motion_vector, traj_constraint):
        write_log("CA constraint for joint", traj_constraint["joint_name"])
        d = traj_constraint["delta"]
        trajectory = traj_constraint["trajectory"]
        start_idx, end_idx = self._find_corresponding_frame_range(motion_vector, traj_constraint)
        n_frames = end_idx-start_idx + 1
        full_length = n_frames*d
        for idx in xrange(n_frames):
            t = (idx*d)/full_length
            target = trajectory.query_point_by_parameter(t)
            keyframe = start_idx+idx
            #write_log("change frame", idx, t, target, constraint["joint_name"])
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._modify_pose(traj_constraint["joint_name"], target)
                iter_counter += 1
            #self._modify_pose(constraint["joint_name"], target)
            motion_vector.frames[keyframe] = self.pose.get_vector()
        self._create_transition_for_frame_range(motion_vector.frames, start_idx, end_idx, self.pose.free_joints_map[traj_constraint["joint_name"]])

    def _modify_motion_vector_using_trajectory_constraint2(self, motion_vector, traj_constraint):
        write_log("CA constraint for joint", traj_constraint["joint_name"])
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
        for idx in xrange(n_frames):
            keyframe = start_idx+idx
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            current_position = self.pose.evaluate_position(traj_constraint["joint_name"])
            arc_length += np.linalg.norm(prev_position-current_position)
            prev_position = current_position
            if arc_length >= trajectory.full_arc_length:
                break
            target = trajectory.query_point_by_absolute_arc_length(arc_length)

            #write_log("change frame", idx, t, target, constraint["joint_name"])
            #print (idx, keyframe, arc_length, n_frames)
            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._modify_pose(traj_constraint["joint_name"], target)
                iter_counter += 1
            #self._modify_pose(constraint["joint_name"], target)
            motion_vector.frames[keyframe] = self.pose.get_vector()

        self._create_transition_for_frame_range(motion_vector.frames, start_idx, keyframe-1, self.pose.free_joints_map[traj_constraint["joint_name"]])

    def _find_corresponding_frame_range(self, motion_vector, traj_constraint):
        start_idx = traj_constraint["start_frame"]
        end_idx = traj_constraint["end_frame"]
        start_target = traj_constraint["trajectory"].query_point_by_parameter(0.0)
        end_target = traj_constraint["trajectory"].query_point_by_parameter(1.0)
        #write_log("looking for corresponding frame range in frame range", start_idx, end_idx, start_target, end_target)
        start_idx = self._find_corresponding_frame(motion_vector, start_idx, end_idx, traj_constraint["joint_name"], start_target)
        end_idx = self._find_corresponding_frame(motion_vector, start_idx, end_idx, traj_constraint["joint_name"], end_target)
        #write_log("found corresponding frame range", start_idx, end_idx)
        return start_idx, end_idx

    def _find_corresponding_frame(self, motion_vector, start_idx, end_idx, target_joint, target_position):
        closest_start_frame = copy(start_idx)
        min_error = np.inf
        n_frames = end_idx-start_idx
        for idx in xrange(n_frames):
            keyframe = start_idx+idx
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            position = self.pose.evaluate_position(target_joint)
            error = np.linalg.norm(position-target_position)
            #print(error, idx)
            if error <= min_error:
                min_error = error
                closest_start_frame = keyframe
        return closest_start_frame

    def _modify_motion_vector_using_ca_constraints(self, motion_vector, ca_constraints):
        print "modify motion vector using ca constraints", len(ca_constraints)
        for c in ca_constraints:
            start_frame = motion_vector.graph_walk.steps[c.step_idx].start_frame
            end_frame = motion_vector.graph_walk.steps[c.step_idx].end_frame
            keyframe = self._find_corresponding_frame(motion_vector, start_frame, end_frame, c.joint_name, c.position)
            print "found keyframe",keyframe,start_frame,end_frame,c.joint_name
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            self._modify_frame_using_keyframe_constraint(motion_vector, c, keyframe)

    def _extract_free_parameters(self, free_joints):
        """get parameters of joints from reference frame
        """
        parameters = list()
        for joint_name in free_joints:
            parameters += self.pose.extract_parameters(joint_name).tolist()
            #print ("array", parameters)
        return np.asarray(parameters)

    def _extract_free_parameter_indices(self, free_joints):
        """get parameter indices of joints from reference frame
        """
        indices = {}
        for joint_name in free_joints:
            indices[joint_name] = list(range(*self.pose.extract_parameters_indices(joint_name)))
            #print ("indices", indices)
        return indices

    def _set_hand_orientation(self, motion_vector, orientation, joint_name, keyframe, start, end):
        #print keyframe
        parent_joint_name = self.pose.get_parent_joint(joint_name)#"RightHand"
        self.set_pose_from_frame(motion_vector.frames[keyframe])
        self.pose.set_hand_orientation(parent_joint_name, orientation)
        start = max(0, start)
        end = min(motion_vector.frames.shape[0], end)
        #print start, end
        self._create_transition_for_frame_range(motion_vector.frames, start, end-1, [parent_joint_name])
