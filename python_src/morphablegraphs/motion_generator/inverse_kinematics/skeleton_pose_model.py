import numpy as np
from ...animation_data.motion_editing import convert_quaternion_frames_to_euler_frames as convert_quat_to_euler
from ...animation_data.motion_editing import euler_to_quaternion, quaternion_to_euler
from ...external.transformations import quaternion_matrix, quaternion_from_matrix, quaternion_multiply, quaternion_inverse
LEN_QUATERNION = 4
LEN_TRANSLATION = 3


def get_3d_rotation_between_vectors(a ,b):
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s ==0:
        return np.eye(3)
    c = np.dot(a,b)
    v_x = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    v_x_2 = np.dot(v_x,v_x)
    r = np.eye(3) + v_x + (v_x_2* (1-c/s**2))
    return r

def quaternion_from_vector_to_vector(a, b):
    "src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another"
    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    return q/ np.linalg.norm(q)


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


def normalize_quaternion(q):
    return quaternion_inverse(q) / np.dot(q, q)


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
            if self.n_channels[joint] > 0:
                self.modelled_joints.append(joint)
        #print "modelled joints",self.modelled_joints
        #print "maximum channel", channel_idx
        self.free_joints_map = skeleton.free_joints_map
        self.reduced_free_joints_map = skeleton.reduced_free_joints_map
        self.head_joint = skeleton.head_joint
        self.neck_joint = skeleton.neck_joint
        self.relative_hand_dir = np.array([1.0, 0.0, 0.0, 0.0])
        self.relative_hand_cross = np.array([0.0,1.0,0.0, 0.0])
        self.relative_head_dir = np.array([0.0, 0.0, 1.0, 0.0])
        self.bounds = skeleton.bounds

    def set_pose_parameters(self, pose_parameters):
        self.pose_parameters = pose_parameters

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

    def evaluate_orientation(self, target_joint):
        """ run fk
        """
        return self.skeleton.nodes[target_joint].get_global_orientation_quaternion(self.pose_parameters)

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

    def lookat(self, point):
        head_position = self.evaluate_position(self.head_joint)
        target_dir = point - head_position
        target_dir /= np.linalg.norm(target_dir)
        head_direction = self.get_joint_direction(self.head_joint, self.relative_head_dir)
        delta_q = quaternion_from_vector_to_vector(head_direction, target_dir)
        delta_matrix = quaternion_matrix(delta_q)

        #delta*parent*old_local = parent*new_local
        #inv_parent*delta*parent*old_local = new_local
        parent_m = self.skeleton.nodes[self.neck_joint].get_global_matrix(self.pose_parameters, use_cache=False)
        old_local = np.dot(parent_m, self.skeleton.nodes[self.head_joint].get_local_matrix(self.pose_parameters))
        m = np.dot(delta_matrix, old_local)
        new_local = np.dot(np.linalg.inv(parent_m),m)
        new_local_q = quaternion_from_matrix(new_local)
        self.set_channel_values(new_local_q, [self.head_joint])

    def set_hand_orientation(self, joint_name, orientation):
        m = quaternion_matrix(orientation)
        target_dir = np.dot(m, self.relative_hand_dir)
        target_dir /= np.linalg.norm(target_dir)
        cross_dir = np.dot(m, self.relative_hand_cross)
        cross_dir /= np.linalg.norm(cross_dir)
        #print "set hand orientation"
        #print target_dir, cross_dir
        self.point_in_direction(joint_name, target_dir[:3], cross_dir[:3])

    def point_in_direction(self, joint_name, target_dir, target_cross=None):
        parent_joint_name = self.get_parent_joint(joint_name)
        joint_direction = self.get_joint_direction(joint_name, self.relative_hand_dir)
        delta_q = quaternion_from_vector_to_vector(joint_direction, target_dir)
        delta_matrix = quaternion_matrix(delta_q)
        #delta*parent*old_local = parent*new_local
        #inv_parent*delta*parent*old_local = new_local
        parent_m = self.skeleton.nodes[parent_joint_name].get_global_matrix(self.pose_parameters, use_cache=False)
        old_local = np.dot(parent_m, self.skeleton.nodes[joint_name].get_local_matrix(self.pose_parameters))
        m = np.dot(delta_matrix, old_local)
        new_local = np.dot(np.linalg.inv(parent_m),m)
        new_local_q = quaternion_from_matrix(new_local)
        self.set_channel_values(new_local_q, [joint_name])

        #rotate around orientation vector
        joint_cross = self.get_joint_direction(joint_name, self.relative_hand_cross)
        if target_cross is not None:
            delta_q = quaternion_from_vector_to_vector(joint_cross, target_cross)
            delta_matrix = quaternion_matrix(delta_q)
            parent_m = self.skeleton.nodes[parent_joint_name].get_global_matrix(self.pose_parameters, use_cache=False)
            old_local = np.dot(parent_m, self.skeleton.nodes[joint_name].get_local_matrix(self.pose_parameters))
            m = np.dot(delta_matrix, old_local)
            new_local = np.dot(np.linalg.inv(parent_m),m)
            new_local_q = quaternion_from_matrix(new_local)
            self.set_channel_values(new_local_q, [joint_name])
        print joint_direction

    def set_joint_orientation(self, joint_name, target_q):
        global_q = self.skeleton.nodes[joint_name].get_global_orientation_quaternion(self.pose_parameters, use_cache=False)
        global_m = quaternion_matrix(global_q)
        target_m = quaternion_matrix(target_q)
        delta_m = np.linalg.inv(global_m)*target_m
        local_m = self.skeleton.nodes[joint_name].get_local_matrix(self.pose_parameters)
        new_local_m = np.dot(delta_m, local_m)
        new_local_q = quaternion_from_matrix(new_local_m)
        self.set_channel_values(new_local_q, [joint_name])

    def get_joint_direction(self, joint_name, ref_vector):
        q = self.evaluate_orientation(joint_name)
        q /= np.linalg.norm(q)
        rotation_matrix = quaternion_matrix(q)
        vec = np.dot(rotation_matrix, ref_vector)[:3]
        return vec/np.linalg.norm(vec)

    def get_parent_joint(self, joint_name):
        if joint_name not in self.skeleton.parent_dict.keys():
            return None
        return self.skeleton.parent_dict[joint_name]

