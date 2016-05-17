import numpy as np


class IKConstraint(object):
    @staticmethod
    def evaluate(self, params, data):
        pass


class JointIKConstraint(IKConstraint):
    def __init__(self, joint_name, position, orientation, keyframe, free_joints, step_idx=-1):
        self.joint_name = joint_name
        self.position = position
        self.orientation = orientation
        self.keyframe = keyframe
        self.free_joints = free_joints
        self.step_idx = step_idx

    @staticmethod
    def evaluate(parameters, data):
        pose, free_joints, target_joint, target_position = data
        pose.set_channel_values(parameters, free_joints)
        d = pose.evaluate_position(target_joint) - target_position
        return np.dot(d, d)

    def data(self, ik, free_joints=None):
        if free_joints is None:
            free_joints = self.free_joints
        return ik.pose, free_joints, self.joint_name, self.position

    def get_joint_names(self):
        return [self.joint_name]


class TwoJointIKConstraint(IKConstraint):
    def __init__(self, joint_names, target_positions, target_center, target_delta, target_direction, keyframe, free_joints):
        self.joint_names = joint_names
        self.target_positions = target_positions
        self.target_center = target_center
        self.target_delta = target_delta
        self.target_direction = target_direction
        self.keyframe = keyframe
        self.free_joints = free_joints

    @staticmethod
    def evaluate(parameters, data):
        pose, free_joints, joint_names, target_positions, target_center, target_delta, target_direction = data
        pose.set_channel_values(parameters, free_joints)
        left = pose.evaluate_position(joint_names[0])
        right = pose.evaluate_position(joint_names[1])
        delta_vector = right - left
        residual_vector = [0.0, 0.0, 0.0]
        #get distance to center
        residual_vector[0] = np.linalg.norm(target_center - (left + 0.5 * delta_vector))
        #get difference to distance between hands
        delta = np.linalg.norm(delta_vector)
        residual_vector[1] = abs(target_delta - delta)
        #print "difference", residual_vector[1]
        #get difference of global orientation
        direction = delta_vector/delta

        residual_vector[2] = abs(target_direction[0] - direction[0]) + \
                             abs(target_direction[1] - direction[1]) + \
                             abs(target_direction[2] - direction[2])
        residual_vector[2] *= 10.0

        #print (target_center, (left + 0.5 * delta_vector), left, right)
        #error = np.linalg.norm(left-target_positions[0]) + np.linalg.norm(right-target_positions[1])
        return sum(residual_vector)#error#residual_vector[0]#sum(residual_vector)

    def data(self, ik, free_joints=None):
        if free_joints is None:
            free_joints = self.free_joints
        return ik.pose, free_joints, self.joint_names, self.target_positions, self.target_center, self.target_delta, self.target_direction

    def get_joint_names(self):
        return self.joint_names


