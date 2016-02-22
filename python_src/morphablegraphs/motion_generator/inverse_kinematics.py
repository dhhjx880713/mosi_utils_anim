import numpy as np
from copy import copy
from scipy.optimize import minimize
from collections import OrderedDict
from ..external.transformations import quaternion_slerp

LEN_QUATERNION = 4
LEN_TRANSLATION = 3


def obj_inverse_kinematics(s, data):
    ik, free_joints, target_joint, target_position = data
    d = ik.evaluate_delta(s, target_joint, target_position, free_joints)
    #print d
    return d


class SkeletonPose(object):
    """
    TODO wrap parameters to allow use with constrained euler, e.g. to rotate an arm using a single parameter
    then have a skeleton model with realistic degrees of freedom and also predefined ik-chains that is initialized using a frame
    """
    def __init__(self, pose_parameters, channels):
        self.pose_parameters = pose_parameters
        self.channels = channels
        self.n_channels = {}
        self.channels_start = {}
        self.types = {}
        channel_idx = 0
        for joint, channels in self.channels.items():
            self.channels_start[joint] = channel_idx
            self.n_channels[joint] = len(channels)
            if len(channels) == LEN_QUATERNION:
                self.types[joint] = "rot"
            elif len(channels) == LEN_TRANSLATION + LEN_QUATERNION:
                self.types[joint] = "trans"
            else:
                self.types[joint] = "rot"
            channel_idx += self.n_channels[joint]
        print "maximum channel", channel_idx
        #TODO read data from file
        self.free_joints_map = {"LeftHand":["LeftShoulder","LeftArm",  "LeftForeArm"],
                           "RightHand":[ "RightShoulder","RightArm","RightForeArm"],
                           "LeftToolEndSite":[ "LeftShoulder","LeftArm","LeftForeArm"],
                           "RightToolEndSite":["RightArm", "RightForeArm"]}#"RightShoulder",
        self.bounds = {}

    def set_channel_values(self, parameters, free_joints):
        p_idx = 0
        for joint_name in free_joints:
            n_channels = self.n_channels[joint_name]
            p_end = p_idx + n_channels
            f_start = self.channels_start[joint_name]
            f_end = f_start + n_channels
            #print f_start, f_end
            self.pose_parameters[f_start:f_end] = parameters[p_idx:p_end]
            p_idx += n_channels
        return self.pose_parameters

    def set_rotation_of_joint(self, joint_name, axis, angle):
        n_channels = self.n_channels[joint_name]
        f_start = self.channels_start[joint_name]
        f_end = f_start + n_channels
        return

    def extract_parameters_indices(self, joint_name):
        f_start = self.channels_start[joint_name]
        f_end = f_start + self.n_channels[joint_name]
        return f_start, f_end

    def extract_parameters(self, joint_name):
        f_start, f_end = self.extract_parameters_indices(joint_name)
        print f_start, f_end, self.pose_parameters[f_start: f_end], len(self.pose_parameters)
        return self.pose_parameters[f_start: f_end]

    def get_vector(self):
        return self.pose_parameters


class InverseKinematics(object):
    def __init__(self, skeleton, algorithm_settings):
        self.skeleton = skeleton
        self.pose = None
        self._ik_settings = algorithm_settings["inverse_kinematics_settings"]
        self.window = self._ik_settings["interpolation_window"]
        self.verbose = False
        self.channels = OrderedDict()
        for node in self.skeleton.nodes.values():
            node_channels = copy(node.channels)
            if np.all([ch in node_channels for ch in ["Xrotation", "Yrotation", "Zrotation"]]):
                node_channels += ["Wrotation"] #TODO fix order
            self.channels[node.node_name] = node_channels
        print "channels", self.channels

    def set_reference_frame(self, reference_frame):
        self.pose = SkeletonPose(reference_frame, self.channels)

    def generate_constraints(self, free_joints):
        """ TODO add bounds on axis components of the quaternion according to
        Inverse Kinematics with Dual-Quaternions, Exponential-Maps, and Joint Limits by Ben Kenwright
        or try out euler based ik
        """
        cons = []
        idx = 0
        for joint_name in free_joints:
            if joint_name in self.pose.bounds.keys():
                start = idx
                for dim, min_b, max_b in self.pose.bounds[joint_name]:
                    if min_b is not None:
                        cons.append(({"type": 'ineq', "fun": lambda x: x[start+dim]-min_b}))
                    if max_b is not None:
                        cons.append(({"type": 'ineq', "fun": lambda x: max_b-x[start+dim]}))
            idx += self.pose.n_channels[joint_name]
        return tuple(cons)

    def run(self, target_joint, target_position, free_joints):
        initial_guess = self._extract_free_parameters(free_joints)
        data = self, free_joints, target_joint, target_position
        print "start optimization for joint", target_joint, len(initial_guess),len(free_joints)
        cons = self.generate_constraints(free_joints)
        result = minimize(obj_inverse_kinematics,
                          initial_guess,
                          args=(data,),
                          method=self._ik_settings["method"],
                          constraints=cons,
                          tol=self._ik_settings["tolerance"],
                          options={'maxiter': self._ik_settings["max_iterations"], 'disp': self.verbose})
        print "finished optimization"
        #return self.pose.get_vector()

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

    def evaluate(self, target_joint, parameters, free_joints):
        """create frame and run fk
        """
        self.pose.set_channel_values(parameters, free_joints)
        return self.skeleton.nodes[target_joint].get_global_position(self.pose.get_vector())

    def evaluate_delta(self, parameters, target_joint, target_position, free_joints):
        position = self.evaluate(target_joint, parameters, free_joints)
        d = position - target_position
        #print target_joint, position, target_position
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
                    self.run(joint_name, target, free_joints)
                motion_vector.frames[keyframe] = self.pose.get_vector()
                #interpolate
                joint_parameter_indices = self._extract_free_parameter_indices(self.pose.free_joints_map[joint_name])
                for joint_name in self.pose.free_joints_map[joint_name]:
                    self.smooth_quaternion_frames_using_slerp(motion_vector.frames,joint_parameter_indices[joint_name], keyframe, self.window)

    def smooth_quaternion_frames_using_slerp(self, quat_frames, joint_parameter_indices, event_frame, window):
        h_window = window/2
        start_frame = event_frame-h_window
        end_frame = event_frame+h_window
        self.apply_slerp(quat_frames, start_frame, event_frame, h_window, joint_parameter_indices)
        self.apply_slerp(quat_frames, event_frame, end_frame, h_window, joint_parameter_indices)

    def apply_slerp(self, quat_frames, start_frame, end_frame, steps, joint_parameter_indices):
        start_q = quat_frames[start_frame, joint_parameter_indices]
        end_q = quat_frames[end_frame, joint_parameter_indices]
        for i in xrange(steps):
            t = float(i)/steps
            #nlerp_q = self.nlerp(start_q, end_q, t)
            slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
            #print "slerp",start_q,  end_q, t, slerp_q
            quat_frames[start_frame+i, joint_parameter_indices] = slerp_q
