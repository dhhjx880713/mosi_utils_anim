""" Analytical IK for arms and legs based on Section 5.3 of [1] and Section 4.4 of [2]

[1] Lee, Jehee, and Sung Yong Shin. "A hierarchical approach to interactive motion editing for human-like figures."
Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.
[2] Kovar, Lucas, John Schreiner, and Michael Gleicher. "Footskate cleanup for motion capture editing." Proceedings of the 2002 ACM SIGGRAPH/Eurographics symposium on Computer animation. ACM, 2002.

"""
import numpy as np
import math
from utils import normalize, limb_projection, to_local_cos,project_vec3
from ...animation_data.utils import quaternion_from_vector_to_vector
from ...animation_data.retargeting.utils import find_rotation_between_vectors
from ...external.transformations import quaternion_multiply, quaternion_about_axis, quaternion_matrix, quaternion_from_matrix
RIGHT_SHOULDER = "RightShoulder"
RIGHT_ELBOW = "RightElbow"
RIGHT_WRIST = "RightHand"
ELBOW_AXIS = [0,1,0]


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
    print a, b, c
    temp = (a*a + b*b - c*c) / (2 * a * b)
    #temp = min(1, temp)
    print temp
    angle = math.acos(temp)
    return angle



class AnalyticalLimbIK(object):
    def __init__(self, skeleton, limb_root, limb_joint, end_effector, joint_axis):
        self.skeleton = skeleton
        self.limb_root = limb_root
        self.limb_joint = limb_joint
        self.end_effector = end_effector
        self.local_joint_axis = joint_axis
        joint_idx = self.skeleton.animated_joints.index(self.limb_joint) * 4 + 3
        self.joint_indices = [joint_idx, joint_idx + 1, joint_idx + 2, joint_idx + 3]
        root_idx = self.skeleton.animated_joints.index(self.limb_root) * 4 + 3
        self.root_indices = [root_idx, root_idx + 1, root_idx + 2, root_idx + 3]

    def calculate_limb_joint_rotation(self, frame, target_position):
        """ find angle so the distance from root to end effector is equal to the distance from the root to the target"""
        root_pos = self.skeleton.nodes[self.limb_root].get_global_position(frame)
        joint_pos = self.skeleton.nodes[self.limb_joint].get_global_position(frame)
        end_effector_pos = self.skeleton.nodes[self.end_effector].get_global_position(frame)

        parent_m = self.skeleton.nodes[self.limb_joint].parent.get_global_matrix(frame)[:3, :3]
        global_joint_axis = normalize(np.dot(parent_m, self.local_joint_axis))

        print self.limb_root, root_pos
        print self.limb_joint, joint_pos

        upper_limb_vec = root_pos - joint_pos
        lower_limb_vec = joint_pos - end_effector_pos
        upper_limb = np.linalg.norm(upper_limb_vec)
        lower_limb = np.linalg.norm(lower_limb_vec)
        #upper_limb_proj = limb_projection(root_pos, joint_pos, global_joint_axis)
        #lower_limb_proj = limb_projection(end_effector_pos, joint_pos, global_joint_axis)
        #target_length_proj = limb_projection(target_position, joint_pos, global_joint_axis)

        initial_length = np.linalg.norm(root_pos - end_effector_pos)

        target_length = np.linalg.norm(root_pos - target_position)
        #joint_delta_angle = calculate_angle(upper_limb, lower_limb, upper_limb_proj, lower_limb_proj, target_length)
        joint_delta_angle = np.pi-calculate_angle2(upper_limb, lower_limb, target_length)
        #joint_delta_angle = np.deg2rad(65)
        joint_delta_q = quaternion_about_axis(joint_delta_angle, self.local_joint_axis)
        joint_delta_q = normalize(joint_delta_q)
        #local_delta_q = self._to_local_coordinate_system(frame, self.limb_joint, joint_delta_q)
        # update frame
        #old_q = frame[self.joint_indices]
        frame[self.joint_indices] = joint_delta_q#local_delta_q#quaternion_multiply(local_delta_q, old_q)

        end_effector_pos2 = self.skeleton.nodes[self.end_effector].get_global_position(frame)
        result_length = np.linalg.norm(root_pos - end_effector_pos2)
        print "found angle", np.degrees(joint_delta_angle), target_length, initial_length, result_length
        return joint_delta_q

    def calculate_limb_root_rotation(self, frame, target_position):
        """ find angle between the vectors end_effector - root and target- root """
        root_pos = self.skeleton.nodes[self.limb_root].get_global_position(frame)
        end_effector_pos = self.skeleton.nodes[self.end_effector].get_global_position(frame)
        src_delta = end_effector_pos - root_pos
        src_dir = src_delta / np.linalg.norm(src_delta)

        target_delta = target_position - root_pos
        target_dir = target_delta / np.linalg.norm(target_delta)

        root_delta_q = find_rotation_between_vectors(src_dir, target_dir)
        root_delta_q = normalize(root_delta_q)

        delta_m = quaternion_matrix(root_delta_q)
        print src_dir, np.dot(delta_m[:3, :3], src_dir), target_dir
        new_local_q = self._to_local_coordinate_system(frame, self.limb_root, root_delta_q)
        frame[self.root_indices] = new_local_q
        end_effector_pos = self.skeleton.nodes[self.end_effector].get_global_position(frame)
        check_delta = end_effector_pos - root_pos
        check_dir = check_delta / np.linalg.norm(check_delta)
        print src_dir, check_dir, target_dir

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

    def apply(self, frame, position):

        # 1 calculate joint angle
        self.calculate_limb_joint_rotation(frame, position)

        # 2 calculate limb root rotation
        self.calculate_limb_root_rotation(frame, position)
        return frame

    def calculate_limb_root_rotation_freu_gelobt_sei_gott_und_jesus_christus_und_der_heilige_geist(self, frame, target_position):
        """ find angle between the vectors end_effector - root and target- root """
        #frame[self.root_indices] = [1,0,0,0]
        root_pos = self.skeleton.nodes[self.limb_root].get_global_position(frame)
        #src_delta = self.skeleton.nodes[self.end_effector].get_global_position(frame) - root_pos
        joint_pos = self.skeleton.nodes[self.limb_joint].get_global_position(frame)
        src_delta = joint_pos - root_pos
        src_dir = src_delta / np.linalg.norm(src_delta)


        target_delta = target_position - root_pos
        target_dir = target_delta / np.linalg.norm(target_delta)

        root_delta_q = find_rotation_between_vectors(src_dir, target_dir)
        root_delta_q = normalize(root_delta_q)

        delta_m = quaternion_matrix(root_delta_q)
        print src_dir, np.dot(delta_m[:3, :3], src_dir), target_dir
        if False:
            parent_joint = self.skeleton.nodes[self.limb_root].parent.node_name
            new_local_q = to_local_cos(self.skeleton, parent_joint, frame, root_delta_q)
            frame[self.root_indices] = new_local_q
            #root_pos = self.skeleton.nodes[self.limb_root].get_global_position(frame)

            joint_pos = self.skeleton.nodes[self.limb_joint].get_global_position(frame)
            check_delta = joint_pos - root_pos
            check_dir = check_delta / np.linalg.norm(check_delta)
            print src_dir, target_dir, check_dir
            return new_local_q
        elif False:

            local_m = self.skeleton.nodes[self.limb_root].get_local_matrix(frame)
            parent_joint = self.skeleton.nodes[self.limb_root].parent.node_name
            parent_m = self.skeleton.nodes[parent_joint].get_global_matrix(frame)

            new_local = np.dot(np.linalg.inv(np.dot(parent_m, local_m)), delta_m)
            new_local_q = quaternion_from_matrix(np.dot(local_m, new_local))
            frame[self.root_indices] = new_local_q# root_delta_q#quaternion_multiply(old_q, root_delta_q)
            joint_pos = self.skeleton.nodes[self.limb_joint].get_global_position(frame)
            check_delta = joint_pos - root_pos
            check_dir = check_delta / np.linalg.norm(check_delta)
            print src_dir, check_dir, target_dir
            return new_local_q
        else:
            # delta*parent*old_local = parent*new_local
            # inv_parent*delta*parent*old_local = new_local
            parent_joint = self.skeleton.nodes[self.limb_root].parent.node_name
            parent_m = self.skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=False)
            old_global = np.dot(parent_m, self.skeleton.nodes[self.limb_root].get_local_matrix(frame))
            new_global = np.dot(delta_m, old_global)
            new_local = np.dot(np.linalg.inv(parent_m), new_global)
            new_local_q = quaternion_from_matrix(new_local)
            frame[self.root_indices] = new_local_q
            joint_pos = self.skeleton.nodes[self.limb_joint].get_global_position(frame)
            check_delta = joint_pos - root_pos
            check_dir = check_delta / np.linalg.norm(check_delta)
            print src_dir, check_dir, target_dir
