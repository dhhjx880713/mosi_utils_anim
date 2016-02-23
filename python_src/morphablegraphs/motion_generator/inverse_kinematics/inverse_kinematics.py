import numpy as np
from copy import copy
import time
from scipy.optimize import minimize
from collections import OrderedDict
from ...animation_data import ROTATION_TYPE_EULER,ROTATION_TYPE_QUATERNION
from blending import smooth_quaternion_frames_using_slerp, smooth_quaternion_frames_using_slerp_overwrite_frames
from skeleton_pose_model import SkeletonPoseModel


LEN_QUATERNION = 4
LEN_TRANSLATION = 3

IK_METHOD_UNCONSTRAINED_OPTIMIZATION = "unconstrained"
IK_METHOD_CYCLIC_COORDINATTE_DESCENT = "ccd"

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
        print "channels", self.channels

    def set_reference_frame(self, reference_frame):
        self.pose = SkeletonPoseModel(self.skeleton, reference_frame, self.channels, self.use_euler)
        self.pose.clear_cache()


    def run_optimization(self, target_joint, target_position, free_joints):
        initial_guess = self._extract_free_parameters(free_joints)
        data = self, free_joints, target_joint, target_position
        print "start optimization for joint", target_joint, len(initial_guess),len(free_joints)
        start = time.clock()
        cons = None#self.pose.generate_constraints(free_joints)
        result = minimize(obj_inverse_kinematics,
                initial_guess,
                args=(data,),
                method=self._ik_settings["optimization_method"],#"SLSQP",#best result using L-BFGS-B
                constraints=cons,
                tol=self._ik_settings["tolerance"],
                options={'maxiter': self._ik_settings["max_iterations"], 'disp': self.verbose})#,'eps':1.0

        position = self.pose.evaluate_position(target_joint)
        print "finished optimization in",time.clock()-start,"seconds with error",np.linalg.norm(position-target_position)#,result["x"].tolist(), initial_guess.tolist()
        self.pose.set_channel_values(result["x"], free_joints)

    def optimize_joint(self, target_joint, target_position, free_joint):
        #optimize x y z
        initial_guess = self.pose.extract_parameters(free_joint)#self._extract_free_parameters([free_joint])
        data = self, [free_joint], target_joint, target_position
        result = minimize(obj_inverse_kinematics,
                initial_guess,
                args=(data,),
                method=self._ik_settings["optimization_method"],#"SLSQP",#best result using L-BFGS-B
                constraints=None,
                tol=self._ik_settings["tolerance"],
                options={'maxiter': self._ik_settings["max_iterations"], 'disp': self.verbose})#,'eps':1.0
        #apply constraints here
        self.pose.apply_bounds(free_joint)
        return result["x"]

    def run_cyclic_coordinate_descent(self, target_joint, target_position, free_joints):
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
        print "finished optimization after", iteration, "iterations with error",delta, "in",time.clock()-start,"seconds"
        return

    def evaluate_delta(self, parameters, target_joint, target_position, free_joints):

        self.pose.set_channel_values(parameters, free_joints) #update frame
        position = self.pose.evaluate_position(target_joint)
        d = position - target_position
        #print target_joint, position, target_position, parameters
        #print parameters.tolist()
        return np.dot(d, d)

    def modify_motion_vector(self, motion_vector):

        #modify individual keyframes based on constraints
        print "number of ik constraints", len(motion_vector.ik_constraints)
        for keyframe, constraints in motion_vector.ik_constraints.items():
            print(keyframe, constraints)
            self.set_reference_frame(motion_vector.frames[keyframe])
            for c in constraints:
                joint_name = c["joint_name"]
                if joint_name in self.pose.free_joints_map.keys():
                    free_joints = self.pose.free_joints_map[joint_name]
                    target = c["position"]
                    if self.solving_method == IK_METHOD_CYCLIC_COORDINATTE_DESCENT:
                        self.run_cyclic_coordinate_descent(joint_name, target, free_joints)
                    else:
                        self.run_optimization(joint_name, target, free_joints)
                motion_vector.frames[keyframe] = self.pose.get_vector()
                #interpolate
                if self.window > 0:
                    print "smooth and interpolate",self.window
                    joint_parameter_indices = self._extract_free_parameter_indices(self.pose.free_joints_map[joint_name])
                    for joint_name in self.pose.free_joints_map[joint_name]:
                        print joint_name
                        smooth_quaternion_frames_using_slerp(motion_vector.frames, joint_parameter_indices[joint_name], keyframe, self.window)

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
            print ("indices", indices)
        return indices