""" Analytical IK for arms and legs based on Section 5.3 of [1] and Section 4.4 of [2]

[1] Lee, Jehee, and Sung Yong Shin. "A hierarchical approach to interactive motion editing for human-like figures."
Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.
[2] Kovar, Lucas, John Schreiner, and Michael Gleicher. "Footskate cleanup for motion capture editing." Proceedings of the 2002 ACM SIGGRAPH/Eurographics symposium on Computer animation. ACM, 2002.

"""
import math
import numpy as np
from utils import normalize, to_local_cos, project_vec3
from ..retargeting.utils import find_rotation_between_vectors
from ..utils import quaternion_from_vector_to_vector
from ...external.transformations import quaternion_multiply, quaternion_about_axis, quaternion_matrix, quaternion_from_matrix, quaternion_inverse


def calculate_angle(upper_limb, lower_limb, ru, rl, target_length):
    upper_limb_sq = upper_limb * upper_limb
    lower_limb_sq = lower_limb * lower_limb
    ru_sq = ru * ru
    rl_sq = rl * rl
    lusq_rusq = upper_limb_sq - ru_sq
    lusq_rusq = max(0, lusq_rusq)
    llsq_rlsq = lower_limb_sq - rl_sq
    llsq_rlsq = max(0, llsq_rlsq)
    temp = upper_limb_sq + rl_sq
    temp += 2 * math.sqrt(lusq_rusq) * math.sqrt(llsq_rlsq)
    temp += - (target_length*target_length)
    temp /= 2 * ru * rl
    print "temp",temp
    temp = max(-1, temp)
    return math.acos(temp)


def calculate_angle2(upper_limb,lower_limb,target_length):
    """ get angle between upper and lower limb based on desired length
    https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html"""
    a = upper_limb
    b = lower_limb
    c = target_length
    #print a, b, c
    temp = (a*a + b*b - c*c) / (2 * a * b)
    temp = min(1, temp)
    temp = max(-1, temp)
    #print temp
    angle = math.acos(temp)
    return angle


class AnalyticalLimbIK(object):
    def __init__(self, skeleton, limb_root, limb_joint, end_effector, joint_axis, local_end_effector_dir, min_angle=0.1):
        self.skeleton = skeleton
        self.limb_root = limb_root
        self.limb_joint = limb_joint
        self.end_effector = end_effector
        self.local_joint_axis = joint_axis
        self.local_end_effector_dir = local_end_effector_dir
        joint_idx = self.skeleton.animated_joints.index(self.limb_joint) * 4 + 3
        self.joint_indices = [joint_idx, joint_idx + 1, joint_idx + 2, joint_idx + 3]
        root_idx = self.skeleton.animated_joints.index(self.limb_root) * 4 + 3
        self.root_indices = [root_idx, root_idx + 1, root_idx + 2, root_idx + 3]
        end_effector_idx = self.skeleton.animated_joints.index(self.end_effector) * 4 + 3
        self.end_effector_indices = [end_effector_idx, end_effector_idx + 1, end_effector_idx + 2, end_effector_idx + 3]
        self.min_angle = min_angle

    @classmethod
    def init_from_dict(cls, skeleton, joint_name, data):
        limb_root = data["root"]
        limb_joint = data["joint"]
        joint_axis = data["joint_axis"]
        end_effector_dir = data["end_effector_dir"]
        return AnalyticalLimbIK(skeleton, limb_root, limb_joint, joint_name, joint_axis, end_effector_dir)

    def calculate_limb_joint_rotation(self, frame, target_position):
        """ find angle so the distance from root to end effector is equal to the distance from the root to the target"""
        root_pos = self.skeleton.nodes[self.limb_root].get_global_position(frame)
        joint_pos = self.skeleton.nodes[self.limb_joint].get_global_position(frame)
        end_effector_pos = self.skeleton.nodes[self.end_effector].get_global_position(frame)

        upper_limb_vec = root_pos - joint_pos
        lower_limb_vec = joint_pos - end_effector_pos
        upper_limb = np.linalg.norm(upper_limb_vec)
        lower_limb = np.linalg.norm(lower_limb_vec)

        #initial_length = np.linalg.norm(root_pos - end_effector_pos)

        target_length = np.linalg.norm(root_pos - target_position)
        angle = calculate_angle2(upper_limb, lower_limb, target_length)
        if abs(angle - np.pi) < self.min_angle:
            angle -= self.min_angle
        joint_delta_angle = np.pi - angle
        joint_delta_q = quaternion_about_axis(joint_delta_angle, self.local_joint_axis)
        joint_delta_q = normalize(joint_delta_q)
        frame[self.joint_indices] = joint_delta_q
        #end_effector_pos2 = self.skeleton.nodes[self.end_effector].get_global_position(frame)
        #result_length = np.linalg.norm(root_pos - end_effector_pos2)
        #print "found angle", np.degrees(joint_delta_angle), target_length, initial_length, result_length
        return joint_delta_q

    def calculate_limb_root_rotation(self, frame, target_position):
        """ find angle between the vectors end_effector - root and target- root """

        # align vectors
        root_pos = self.skeleton.nodes[self.limb_root].get_global_position(frame)
        end_effector_pos = self.skeleton.nodes[self.end_effector].get_global_position(frame)
        original_toe_position = self.skeleton.nodes[self.end_effector].children[0].get_global_position(frame)
        original_direction = normalize(original_toe_position - end_effector_pos)
        src_delta = end_effector_pos - root_pos
        src_dir = src_delta / np.linalg.norm(src_delta)

        target_delta = target_position - root_pos
        target_dir = target_delta / np.linalg.norm(target_delta)

        root_delta_q = find_rotation_between_vectors(src_dir, target_dir)
        root_delta_q = normalize(root_delta_q)

        #delta_m = quaternion_matrix(root_delta_q)
        #print src_dir, np.dot(delta_m[:3, :3], src_dir), target_dir
        new_local_q = self._to_local_coordinate_system(frame, self.limb_root, root_delta_q)
        frame[self.root_indices] = new_local_q

        # rotate around vector based on original orientation
        if False:
            pos1 = self.skeleton.nodes[self.end_effector].get_global_position(frame)
            pos2 = self.skeleton.nodes[self.end_effector].children[0].get_global_position(frame)
            new_direction = normalize(pos2-pos1)
            q = normalize(quaternion_from_vector_to_vector(new_direction, original_direction))
            new_q = quaternion_multiply(q, root_delta_q)
            new_q = normalize(new_q)
            frame[self.root_indices] = self._to_local_coordinate_system(frame, self.limb_root, new_q)

        elif False:
            end_effector_pos = self.skeleton.nodes[self.end_effector].get_global_position(frame)
            toe_pos = self.skeleton.nodes[self.end_effector].children[0].get_global_position(frame)
            proj_epos = project_vec3(end_effector_pos, target_dir)
            proj_toe_pos = project_vec3(toe_pos, target_dir)
            proj_orig_toe_pos = project_vec3(original_toe_position, target_dir)

            toe_length = np.linalg.norm(proj_toe_pos - proj_epos)
            toe_delta = np.linalg.norm(proj_toe_pos - proj_orig_toe_pos)

            t2 = toe_length * toe_length
            cos_alpha = (t2+t2 - toe_delta*toe_delta) / (2*t2)
            cos_alpha = min(1, cos_alpha)
            cos_alpha = max(-1, cos_alpha)
            alpha = np.pi - math.acos(cos_alpha)
            q = normalize(quaternion_about_axis(alpha, target_dir))
            new_q = quaternion_multiply(q, root_delta_q)
            new_q = normalize(new_q)
            frame[self.root_indices] = self._to_local_coordinate_system(frame, self.limb_root, new_q)

    def _to_local_coordinate_system(self, frame, joint_name, q):
        """ given a global rotation concatenate it with an existing local rotation and bring it to the local coordinate system"""
        # delta*parent*old_local = parent*new_local
        # inv_parent*delta*parent*old_local = new_local
        delta_m = quaternion_matrix(q)
        parent_joint = self.skeleton.nodes[joint_name].parent.node_name
        parent_m = self.skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=False)
        old_global = np.dot(parent_m, self.skeleton.nodes[joint_name].get_local_matrix(frame))
        new_global = np.dot(delta_m, old_global)
        new_local = np.dot(np.linalg.inv(parent_m), new_global)
        new_local_q = quaternion_from_matrix(new_local)
        return new_local_q

    def calculate_end_effector_rotation(self, frame, target_dir):
        #print "end effector rotation", self.end_effector, target_dir
        #end_effector_m = self.skeleton.nodes[self.end_effector].get_global_matrix(frame)[:3, :3]
        #src_dir = np.dot(end_effector_m, self.local_end_effector_dir)
        #src_dir = normalize(src_dir)
        src_dir = self.get_joint_dir(frame, self.end_effector)
        global_delta_q = find_rotation_between_vectors(src_dir, target_dir)
        new_local_q = self._to_local_coordinate_system(frame, self.end_effector, global_delta_q)
        frame[self.end_effector_indices] = new_local_q

    def set_end_effector_rotation(self, frame, target_orientation):
        #print "set orientation", target_orientation
        q = self.get_global_joint_orientation(self.end_effector, frame)
        delta_orientation = quaternion_multiply(target_orientation, quaternion_inverse(q))
        new_local_q = self._to_local_coordinate_system(frame, self.end_effector, delta_orientation)
        frame[self.end_effector_indices] = new_local_q
        #t = self.skeleton.nodes[self.skeleton.nodes[self.end_effector].children[0].node_name].get_global_position(frame)
        #h = self.skeleton.nodes[self.skeleton.nodes[self.end_effector].children[1].node_name].get_global_position(frame)
        #original_direction = normalize(t - h)
        #print original_direction

    def get_global_joint_orientation(self, joint_name, frame):
        m = self.skeleton.nodes[joint_name].get_global_matrix(frame)
        m[:3, 3] = [0, 0, 0]
        return normalize(quaternion_from_matrix(m))

    def get_joint_dir(self, frame, joint_name):
        pos1 = self.skeleton.nodes[joint_name].get_global_position(frame)
        pos2 = self.skeleton.nodes[joint_name].children[0].get_global_position(frame)
        return normalize(pos2 - pos1)

    def apply(self, frame, position=None, direction=None):

        if position is not None:
            # 1 calculate joint angle based on the distance to target position
            self.calculate_limb_joint_rotation(frame, position)

            # 2 calculate limb root rotation to align the end effector with the target position
            self.calculate_limb_root_rotation(frame, position)

        # 3 orient end effector
        if direction is not None:
            self.calculate_end_effector_rotation(frame, direction)
        return frame

    def apply2(self, frame, position, orientation):
        if position is not None:
            # 1 calculate joint angle based on the distance to target position
            self.calculate_limb_joint_rotation(frame, position)

            # 2 calculate limb root rotation to align the end effector with the target position
            self.calculate_limb_root_rotation(frame, position)

        # 3 orient end effector
        if orientation is not None:
            self.set_end_effector_rotation(frame, orientation)
        return frame