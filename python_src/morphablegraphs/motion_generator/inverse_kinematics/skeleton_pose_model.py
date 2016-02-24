import numpy as np
from ...animation_data.motion_editing import convert_quaternion_frames_to_euler_frames as convert_quat_to_euler
from ...animation_data.motion_editing import euler_to_quaternion, quaternion_to_euler
LEN_QUATERNION = 4
LEN_TRANSLATION = 3


def convert_euler_to_quat(euler_frame, joints):
    quat_frame = euler_frame[:3].tolist()
    offset = 3
    step = 3
    for joint in joints:
        e = euler_frame[offset:offset+step]
        #print joint, e
        q = euler_to_quaternion(e)
        quat_frame += list(q)
        offset += step
    return np.array(quat_frame)


class SkeletonPoseModel(object):
    """
    TODO wrap parameters to allow use with constrained euler, e.g. to rotate an arm using a single parameter
    then have a skeleton model with realistic degrees of freedom and also predefined ik-chains that is initialized using a frame
    """
    def __init__(self, skeleton, pose_parameters, channels, use_euler=False):
        self.skeleton = skeleton
        self.use_euler = use_euler
        if self.use_euler:
            self.pose_parameters = convert_quat_to_euler([pose_parameters])[0]#change to euler
        else:
            self.pose_parameters = pose_parameters
        self.channels = channels
        self.n_channels = {}
        self.channels_start = {}
        self.types = {}
        self.modelled_joints = []
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
            if self.n_channels[joint] > 0 :
                self.modelled_joints.append(joint)
        print "modelled joints",self.modelled_joints
        print "maximum channel", channel_idx
        #TODO read data from file
        self.free_joints_map = {"LeftHand":["Spine","LeftArm",  "LeftForeArm"],#"LeftShoulder",
                           "RightHand":["Spine","RightArm","RightForeArm"],# "RightShoulder",
                           "LeftToolEndSite":["Spine", "LeftArm","LeftForeArm"],
                           "RightToolEndSite":["Spine","RightArm", "RightForeArm"]}#"RightShoulder",
        self.bounds = {"LeftArm":[],#{"dim": 1, "min": 0, "max": 90}
                       "RightArm":[]}#{"dim": 1, "min": 0, "max": 90},{"dim": 0, "min": 0, "max": 90}

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

    #def set_rotation_of_joint(self, joint_name, axis, angle):
    #    n_channels = self.n_channels[joint_name]
    #    f_start = self.channels_start[joint_name]
    #    f_end = f_start + n_channels
    #    return

    def extract_parameters_indices(self, joint_name):
        f_start = self.channels_start[joint_name]
        f_end = f_start + self.n_channels[joint_name]
        return f_start, f_end

    def extract_parameters(self, joint_name):
        f_start, f_end = self.extract_parameters_indices(joint_name)
        #print joint_name, f_start, f_end, self.pose_parameters[f_start: f_end].tolist(), len(self.pose_parameters)
        return self.pose_parameters[f_start: f_end]

    def get_vector(self):
        if self.use_euler:
            return convert_euler_to_quat(self.pose_parameters, self.modelled_joints)#convert to quat
        else:
            return self.pose_parameters

    def evaluate_position_with_cache(self, target_joint, free_joints):
        """ run fk
        """
        for joint in free_joints:
            self.skeleton.nodes[joint].get_global_position(self.pose_parameters, use_cache=True)
        return self.skeleton.nodes[target_joint].get_global_position(self.pose_parameters, use_cache=True)#get_vector()

    def evaluate_position(self, target_joint):
        """ run fk
        """
        return self.skeleton.nodes[target_joint].get_global_position(self.pose_parameters)#get_vector()

    def apply_bounds(self, free_joint):
        if free_joint in self.bounds.keys():
            euler_angles = self.get_euler_angles(free_joint)
            for bound in self.bounds[free_joint]:
                self.apply_bound_on_joint(euler_angles, bound)

            if self.use_euler:
                self.set_channel_values(euler_angles, [free_joint])
            else:
                q = euler_to_quaternion(euler_angles)
                #print("apply bound")
                self.set_channel_values(q, [free_joint])
        return

    def apply_bound_on_joint(self, euler_angles, bound):
        #self.pose_parameters[start+bound["dim"]] =
        #print euler_angles
        if "min" in bound.keys():
            euler_angles[bound["dim"]] = max(euler_angles[bound["dim"]],bound["min"])
        if "max" in bound.keys():
            euler_angles[bound["dim"]] = min(euler_angles[bound["dim"]],bound["max"])
        #print "after",euler_angles

    def get_euler_angles(self, joint):
        if self.use_euler:
            euler_angles = self.extract_parameters(joint)
        else:
            q = self.extract_parameters(joint)
            euler_angles = quaternion_to_euler(q)
        return euler_angles


    def generate_constraints(self, free_joints):
        """ TODO add bounds on axis components of the quaternion according to
        Inverse Kinematics with Dual-Quaternions, Exponential-Maps, and Joint Limits by Ben Kenwright
        or try out euler based ik
        """
        cons = []
        idx = 0
        for joint_name in free_joints:
            if joint_name in self.bounds.keys():
                start = idx
                for bound in self.bounds[joint_name]:
                    if "min" in bound.keys():
                        cons.append(({"type": 'ineq', "fun": lambda x: x[start+bound["dim"]]-bound["min"]}))
                    if "max" in bound.keys():
                        cons.append(({"type": 'ineq', "fun": lambda x: bound["max"]-x[start+bound["dim"]]}))
            idx += self.n_channels[joint_name]
        return tuple(cons)

    def clear_cache(self):
        self.skeleton.clear_cached_global_matrices()
