import numpy as np
from copy import copy
from scipy.optimize import minimize
from collections import OrderedDict


def obj_inverse_kinematics(s, data):
    ik, free_joints, target_joint, target_position = data
    return ik.evaluate_delta(s, target_joint, target_position, free_joints)


class AnimationFrame(object):
    def __init__(self, frame_parameters, channels):
        self.frame_parameters = frame_parameters
        self.channels = channels
        self.n_channels = {}
        self.channels_start = {}
        channel_idx = 0
        for joint, channels in self.channels.items():
            self.channels_start[joint] = channel_idx
            self.n_channels[joint] = len(channels)
            channel_idx += self.n_channels[joint]

    def set_channel_values(self, parameters, free_joints):
        p_idx = 0
        for joint_name in free_joints:
            n_channels = self.n_channels[joint_name]
            p_end = p_idx + n_channels
            f_start = self.channels_start[joint_name]
            f_end = f_start + n_channels
            self.frame_parameters[f_start:f_end] += parameters[p_idx:p_end]
            p_idx += n_channels
        return self.frame_parameters

    def extract_parameters(self, joint_name):
        f_start = self.channels_start[joint_name]
        f_end = f_start + self.n_channels[joint_name]
        return self.frame_parameters[f_start, f_end]

    def get_vector(self):
        return self.frame_parameters


class InverseKinematics(object):
    def __init__(self, skeleton, optimization_settings):
        self.skeleton = skeleton
        self.frame = None
        self.optimization_settings = optimization_settings
        self.verbose = False

    def set_reference_frame(self, reference_frame):
        channels = OrderedDict()
        for node in self.skeleton.nodes:
            channels[node.name] = node.channels
        self.frame = AnimationFrame(reference_frame, channels)

    def run(self, target_joint, target_position, free_joints):
        initial_guess = self._extract_free_parameters(free_joints)
        data = self, free_joints, target_joint, target_position
        result = minimize(obj_inverse_kinematics,
                          initial_guess,
                          args=(data,),
                          method=self.optimization_settings["method"],
                          tol=self.optimization_settings["tolerance"],
                          options={'maxiter': self.optimization_settings["max_iterations"], 'disp': self.verbose})
        return self.frame.get_frame_parameters()

    def _extract_free_parameters(self, free_joints):
        """get parameters of joints from reference frame
        """
        parameters = []
        for joint_name in free_joints:
            parameters += self.frame.extract_parameters(joint_name).tolist()
        return np.asarray(parameters)

    def evaluate(self, target_joint, parameters, free_joints):
        """create frame and run fk
        """
        self.frame.set_channel_values(parameters, free_joints)
        return self.skeleton.nodes[target_joint].get_global_position(self.frame.get_vector())

    def evaluate_delta(self, parameters, target_joint, target_position, free_joints):
        position = self.evaluate(target_joint, parameters, free_joints)
        d = position - target_position
        return np.dot(d, d)

