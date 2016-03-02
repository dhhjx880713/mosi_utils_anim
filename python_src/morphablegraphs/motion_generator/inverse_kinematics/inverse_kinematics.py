import numpy as np
from copy import copy
import time
from scipy.optimize import minimize
from collections import OrderedDict
from ...animation_data import ROTATION_TYPE_EULER,ROTATION_TYPE_QUATERNION
from blending import smooth_quaternion_frames_using_slerp, smooth_quaternion_frames_using_slerp_overwrite_frames
from skeleton_pose_model import SkeletonPoseModel
from ...utilities import write_log


LEN_QUATERNION = 4
LEN_TRANSLATION = 3

IK_METHOD_UNCONSTRAINED_OPTIMIZATION = "unconstrained"
IK_METHOD_CYCLIC_COORDINATE_DESCENT = "ccd"

def obj_inverse_kinematics(s, data):
    ik, free_joints, target_joint, target_position = data
    d = ik.evaluate_delta(s, target_joint, target_position, free_joints)
    #print d
    return d


class InverseKinematics(object):
    def __init__(self, skeleton, algorithm_settings):
        self.skeleton = skeleton
        self.pose = None
        self._ik_settings = algorithm_settings["inverse_kinematics_settings"]
        self.window = self._ik_settings["interpolation_window"]
        self.verbose = False
        self.use_euler = self._ik_settings["use_euler_representation"]
        self.solving_method = self._ik_settings["solving_method"]
        self.success_threshold = 5.0
        self.max_retries = 5
        if self.use_euler:
            self.skeleton.set_rotation_type(ROTATION_TYPE_EULER)#change to euler
        self.channels = OrderedDict()
        for node in self.skeleton.nodes.values():
            node_channels = copy(node.channels)
            #change to euler
            if not self.use_euler:
                if np.all([ch in node_channels for ch in ["Xrotation", "Yrotation", "Zrotation"]]):
                    node_channels += ["Wrotation"] #TODO fix order
            self.channels[node.node_name] = node_channels
        #print "channels", self.channels

    def _run_optimization(self, objective, initial_guess, data, cons=None):
         return minimize(objective, initial_guess, args=(data,),
                        method=self._ik_settings["optimization_method"],#"SLSQP",#best result using L-BFGS-B
                        constraints=cons, tol=self._ik_settings["tolerance"],
                        options={'maxiter': self._ik_settings["max_iterations"], 'disp': self.verbose})#,'eps':1.0

    def set_pose_from_frame(self, reference_frame):
        #TODO initialize pose once and just update the frame
        self.pose = SkeletonPoseModel(self.skeleton, reference_frame, self.channels, self.use_euler)
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
        free_joints = constraint.free_joints(self)
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
        data = self, free_joints, target_joint, target_position
        write_log("start optimization for joint", target_joint, len(initial_guess), len(free_joints))
        start = time.clock()
        cons = None#self.pose.generate_constraints(free_joints)
        result = self._run_optimization(obj_inverse_kinematics, initial_guess, data, cons)
        position = self.pose.evaluate_position(target_joint)
        error = np.linalg.norm(position-target_position)
        write_log("finished optimization in",time.clock()-start, "seconds with error", error) #,result["x"].tolist(), initial_guess.tolist()
        self.pose.set_channel_values(result["x"], free_joints)
        return error

    def optimize_joint(self, target_joint, target_position, free_joint):
        #optimize x y z
        initial_guess = self.pose.extract_parameters(free_joint)#self._extract_free_parameters([free_joint])
        data = self, [free_joint], target_joint, target_position
        result = self._run_optimization(obj_inverse_kinematics, initial_guess, data)
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
                self.optimize_joint(target_joint, target_position, free_joint)
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

    def _modify_motion_vector_using_keyframe_constraint_list(self, motion_vector, constraints):
        write_log("number of ik keyframe constraints", len(constraints))
        for keyframe, constraints in constraints.items():
            write_log(keyframe, constraints)
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            if "multiple" in constraints.keys():
                for c in constraints["multiple"]:
                    self._modify_frame_using_keyframe_constraint(motion_vector, c, keyframe)
            if "single" in constraints.keys():
                for c in constraints["single"]:
                    self._modify_frame_using_keyframe_constraint(motion_vector, c, keyframe)

    def _modify_frame_using_keyframe_constraint(self, motion_vector, constraint, keyframe):
        #joint_name = constraint["joint_name"]
        #self._modify_pose(joint_name, constraint["position"])
        self._modify_pose_general(constraint)
        motion_vector.frames[keyframe] = self.pose.get_vector()
        #interpolate
        print "free joints", constraint.free_joints(self)
        #self.window = 0
        if self.window > 0:
            for target_joint_name in constraint.get_joint_names():
                write_log("smooth and interpolate", self.window)
                joint_parameter_indices = self._extract_free_parameter_indices(self.pose.free_joints_map[target_joint_name])
                for joint_name in self.pose.free_joints_map[target_joint_name]:
                    #print joint_name
                    smooth_quaternion_frames_using_slerp(motion_vector.frames, joint_parameter_indices[joint_name], keyframe, self.window)

        start = keyframe - self.window/2
        end = keyframe + self.window/2
        self._look_at_in_range(motion_vector, constraint.position, start, end)

    def _look_at_in_range(self, motion_vector, position, start, end):
        for idx in xrange(start, end):
            self.set_pose_from_frame(motion_vector.frames[idx])
            self.pose.lookat(position)
            motion_vector.frames[idx] = self.pose.get_vector()

    def _modify_motion_vector_using_trajectory_constraint_list(self, motion_vector, constraints):
        write_log("number of ik trajectory constraints", len(constraints))
        for c in constraints:
            self._modify_motion_vector_using_trajectory_constraint(motion_vector, c)

    def _modify_motion_vector_using_trajectory_constraint(self, motion_vector, constraint):
        write_log("ca constraint for joint", constraint["joint_name"])
        d = constraint["delta"]
        trajectory = constraint["trajectory"]
        n_frames = constraint["end_frame"]-constraint["start_frame"]
        full_length = n_frames*d
        for idx in xrange(n_frames):
            t = (idx*d)/full_length
            target = trajectory.query_point_by_parameter(t)
            keyframe = constraint["start_frame"]+idx
            #write_log("change frame", idx, t, target, constraint["joint_name"])
            self.set_pose_from_frame(motion_vector.frames[keyframe])
            error = np.inf
            iter_counter = 0
            while error > self.success_threshold and iter_counter < self.max_retries:
                error = self._modify_pose(constraint["joint_name"], target)
                iter_counter += 1

            #self._modify_pose(constraint["joint_name"], target)
            motion_vector.frames[keyframe] = self.pose.get_vector()

    def _extract_free_parameters(self, free_joints):
        """get parameters of joints from reference frame
        """
        parameters = []
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

