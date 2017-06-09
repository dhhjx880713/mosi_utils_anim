""" Analytical IK for arms and legs based on Section 5.3 of [1] and Section 4.4 of [2]

[1] Lee, Jehee, and Sung Yong Shin. "A hierarchical approach to interactive motion editing for human-like figures."
Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.
[2] Kovar, Lucas, John Schreiner, and Michael Gleicher. "Footskate cleanup for motion capture editing." Proceedings of the 2002 ACM SIGGRAPH/Eurographics symposium on Computer animation. ACM, 2002.

"""
import numpy as np
import math
from utils import normalize
from ...animation_data.utils import quaternion_from_vector_to_vector
from ...external.transformations import quaternion_multiply, quaternion_about_axis, quaternion_matrix, quaternion_from_matrix


def calculate_angle(l1, l2, r1, r2, L):
    l1_sq = l1 * l1
    l2_sq = l2 * l2
    r1_sq = r1 * r1
    r2_sq = r2 * r2
    temp = (l1_sq + l2_sq + 2 * math.sqrt(l1_sq-r1_sq)* math.sqrt(l2_sq-r2_sq) - L*L )/ (2* r1 * r2)
    return math.acos(temp)


def to_local_cos(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = skeleton.nodes[node_name].get_global_matrix(frame)[:3,:3]
    #pm[:3, 3] = [0, 0, 0]
    inv_pm = np.linalg.inv(pm)
    r = quaternion_matrix(q)[:3,:3]
    lr = np.dot(inv_pm, r)[:3,:3]
    q = quaternion_from_matrix(lr)
    return q

def project_onto_plane(x, n):
    """https://stackoverflow.com/questions/17915475/how-may-i-project-vectors-onto-a-plane-defined-by-its-orthogonal-vector-in-pytho"""
    nl = np.linalg.norm(n)
    d = np.dot(x, n) / nl
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]


def project_vec_on_plane(vec, n):
    """https://math.stackexchange.com/questions/633181/formula-to-project-a-vector-onto-a-plane"""
    n = normalize(n)
    d = np.dot(vec, n)
    return vec - np.dot(d, n)


class AnalyticalLimbIK(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.shoulder = "RightShoulder"
        self.elbow = "RightElbow"
        self.wrist = "RightHand"
        self.local_elbow_axis = [0,1,0]
        elbow_idx = self.skeleton.animated_joints.index(self.elbow) * 4 + 3
        self.elbow_indices = [elbow_idx, elbow_idx + 1, elbow_idx + 2, elbow_idx +3]
        shoulder_idx = self.skeleton.animated_joints.index(self.shoulder) * 4 + 3
        self.shoulder_indices = [shoulder_idx, shoulder_idx + 1, shoulder_idx + 2, shoulder_idx + 3]


    def apply(self, frame, position):
        parent_m = self.skeleton.nodes[self.elbow].parent.get_global_matrix(frame)[:3,:3]
        global_elbow_axis = np.dot(parent_m, self.local_elbow_axis)
        elbow_pos = self.skeleton.nodes[self.elbow]
        upper_arm_vec = self.skeleton.nodes[self.shoulder]-elbow_pos
        lower_arm_vec = elbow_pos - self.skeleton.nodes[self.wrist]
        l1 = np.linalg.norm(upper_arm_vec)
        l2 = np.linalg.norm(lower_arm_vec)
        r1 = np.linalg.norm(project_vec_on_plane(upper_arm_vec, global_elbow_axis))
        r2 = np.linalg.norm(project_vec_on_plane(lower_arm_vec, global_elbow_axis))

        shoulder = self.skeleton.nodes[self.shoulder].get_position(frame)
        delta = position - shoulder
        L = np.linalg.norm(delta)

        # 1 calculate elbow angle
        elbow_delta_angle = calculate_angle(l1, l2, r1, r2, L)
        elbow_delta_q = quaternion_about_axis(elbow_delta_angle, global_elbow_axis)
        #local_delta_q = to_local_cos(self.skeleton, self.shoulder, frame, elbow_delta_q)

        # update frame
        old_q = frame[self.elbow_indices]
        frame[self.elbow_indices] = quaternion_multiply(old_q, elbow_delta_q)

        # 2 calculate shoulder angle
        wrist_position = self.skeleton.nodes[self.wrist].get_position(frame)
        constraint_dir = delta /L
        temp_delta = wrist_position - shoulder
        temp_dir = temp_delta / np.linalg.norm(temp_delta)
        shoulder_delta_q = quaternion_from_vector_to_vector(constraint_dir, temp_dir)
        local_delta_q = to_local_cos(self.skeleton,self.shoulder, frame, shoulder_delta_q)

        # update frame
        old_q = frame[self.shoulder_indices]
        frame[self.shoulder_indices] = quaternion_multiply(old_q, local_delta_q)
        return frame
