import numpy as np
from ...animation_data.motion_editing import convert_quaternion_frames_to_euler_frames as convert_quat_to_euler
from ...animation_data.motion_editing import euler_to_quaternion
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
        self.free_joints_map = {"LeftHand":["LeftArm",  "LeftForeArm"],#"LeftShoulder",
                           "RightHand":["RightArm","RightForeArm"],# "RightShoulder",
                           "LeftToolEndSite":[ "LeftShoulder","LeftArm","LeftForeArm"],
                           "RightToolEndSite":["RightArm", "RightForeArm"]}#"RightShoulder",
        self.bounds = {"LeftArm":[{"dim": 1, "min": 0, "max": 180}],
                       "RightArm":[{"dim": 1, "min": 0, "max": 180}]}

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
        print joint_name, f_start, f_end, self.pose_parameters[f_start: f_end].tolist(), len(self.pose_parameters)
        return self.pose_parameters[f_start: f_end]

    def get_vector(self):
        if self.use_euler:
            return convert_euler_to_quat(self.pose_parameters, self.modelled_joints)#convert to quat
        else:
            return self.pose_parameters

    def evaluate_position(self, target_joint, parameters, free_joints):
        """create frame and run fk
        """
        self.set_channel_values(parameters, free_joints)
        #self.skeleton.set_rotation_type(ROTATION_TYPE_EULER)#change to euler
        return self.skeleton.nodes[target_joint].get_global_position(self.pose_parameters)#get_vector()
