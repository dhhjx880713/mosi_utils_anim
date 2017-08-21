import collections
import numpy as np
from .analytical_inverse_kinematics import AnalyticalLimbIK
from .numerical_ik_exp import NumericalInverseKinematicsExp
from .utils import normalize, project_on_intersection_circle, smooth_root_positions, quaternion_from_vector_to_vector
from ..motion_blending import create_transition_for_joints_using_slerp, BLEND_DIRECTION_FORWARD, BLEND_DIRECTION_BACKWARD, smooth_translation_in_quat_frames
from ...external.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_slerp


def create_grounding_constraint_from_frame(skeleton, frames, frame_idx, joint_name):
    position = skeleton.nodes[joint_name].get_global_position(frames[frame_idx])
    m = skeleton.nodes[joint_name].get_global_matrix(frames[frame_idx])
    m[:3, 3] = [0, 0, 0]
    orientation = normalize(quaternion_from_matrix(m))
    return MotionGroundingConstraint(frame_idx, joint_name, position, None, orientation)


def generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, ankle_joint_name, heel_joint, toe_joint_name, target_ground_height, toe_pos=None):
    """ create a constraint on the ankle position based on the toe constraint position"""
    #print "add toe constraint"
    if toe_pos is None:
        ct = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])
        ct[1] = target_ground_height  # set toe constraint on the ground
    else:
        ct = toe_pos

    a = skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
    t = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])

    target_toe_offset = a - t  # difference between unmodified toe and ankle at the frame
    ca = ct + target_toe_offset  # move ankle so toe is on the ground

    m = skeleton.nodes[heel_joint].get_global_matrix(frames[frame_idx])
    m[:3, 3] = [0, 0, 0]
    oq = quaternion_from_matrix(m)
    oq = normalize(oq)

    return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, oq)


def create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, ankle_joint, heel_joint, toe_joint,heel_offset, target_ground_height, heel_pos=None, toe_pos=None):
    if toe_pos is None:
        ct = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
        ct[1] = target_ground_height
    else:
        ct = toe_pos
    if heel_pos is None:
        ch = skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
        ch[1] = target_ground_height
    else:
        ch = heel_pos


    target_direction = normalize(ct - ch)
    t = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
    h = skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
    original_direction = normalize(t - h)

    global_delta_q = quaternion_from_vector_to_vector(original_direction, target_direction)
    global_delta_q = normalize(global_delta_q)

    m = skeleton.nodes[heel_joint].get_global_matrix(frames[frame_idx])
    m[:3, 3] = [0, 0, 0]
    oq = quaternion_from_matrix(m)
    oq = normalize(oq)
    orientation = normalize(quaternion_multiply(global_delta_q, oq))

    # set target ankle position based on the  grounded heel and the global target orientation of the ankle
    m = quaternion_matrix(orientation)[:3, :3]
    target_heel_offset = np.dot(m, heel_offset)
    ca = ch - target_heel_offset
    print("set ankle constraint both", ch, ca, target_heel_offset)
    return MotionGroundingConstraint(frame_idx, ankle_joint, ca, None, orientation)


def interpolate_constraints(c1, c2):
    p = (c1.position + c2.position)/2
    o = quaternion_slerp(c1.orientation, c2.orientation, 0.5)
    o = normalize(o)
    return MotionGroundingConstraint(c1.frame_idx, c1.joint_name, p, None, o)


class MotionGroundingConstraint(object):
    def __init__(self, frame_idx, joint_name, position, direction=None, orientation=None):
        self.frame_idx = frame_idx
        self.joint_name = joint_name
        self.position = position
        self.direction = direction
        self.orientation = orientation
        self.toe_position = None
        self.heel_position = None

    def evaluate(self, skeleton, q_frame):
        d = self.position - skeleton.nodes[self.joint_name].get_global_position(q_frame)
        return np.dot(d, d)


class IKConstraintSet(object):
    def __init__(self, frame_range, joint_names, positions):
        self.frame_range = frame_range
        self.joint_names = joint_names
        self.constraints = []
        for idx in range(frame_range[0], frame_range[1]):
            for idx, joint_name in enumerate(joint_names):
                c = MotionGroundingConstraint(idx, joint_name, positions[idx], None)
                self.constraints.append(c)

    def add_constraint(self, c):
        self.constraints.append(c)

    def evaluate(self, skeleton, q_frame):
        error = 0
        for c in self.constraints:
            d = c.position - skeleton.nodes[c.joint_name].get_global_position(q_frame)
            error += np.dot(d, d)
        return error


def add_fixed_dofs_to_frame(skeleton, frame):
    o = 3
    full_frame = frame[:3].tolist()
    for key, node in list(skeleton.nodes.items()):
        if len(node.children) == 0:
            continue
        if not node.fixed:
            full_frame += frame[o:o+4].tolist()
            o += 4
        else:
            full_frame += node.rotation.tolist()
    return full_frame


class MotionGrounding(object):
    def __init__(self, skeleton, ik_settings, skeleton_def, use_analytical_ik=True):
        self.skeleton = skeleton
        self._ik = NumericalInverseKinematicsExp(skeleton, ik_settings)
        self._constraints = collections.OrderedDict()
        self.transition_window = 10
        self.root_smoothing_window = 20
        self.translation_blend_window = 40
        self._blend_ranges = collections.OrderedDict()
        self.use_analytical_ik = use_analytical_ik
        self._ik_chains = skeleton_def["ik_chains"]
        self._skeleton_def = skeleton_def

    def set_constraints(self, constraints):
        self._constraints = constraints

    def add_constraint(self, joint_name, frame_range, position, direction=None):
        for frame_idx in range(*frame_range):
            c = MotionGroundingConstraint(frame_idx, joint_name, position, direction)
            if frame_idx not in list(self._constraints.keys()):
                self._constraints[frame_idx] = []
            self._constraints[frame_idx].append(c)

    def add_blend_range(self, joint_names, frame_range):
        if frame_range not in list(self._constraints.keys()):
            self._blend_ranges[frame_range] = []
        for j in joint_names:
            self._blend_ranges[frame_range].append(j)

    def clear_constraints(self):
        self._constraints = collections.OrderedDict()

    def clear_blend_ranges(self):
        self._blend_ranges = collections.OrderedDict()

    def clear(self):
        self.clear_constraints()
        self.clear_blend_ranges()

    def run(self, motion_vector, scene_interface=None):
        new_frames = motion_vector.frames[:]
        self._shift_root_using_static_offset(new_frames, scene_interface)
        self.shift_root_to_reach_constraints(new_frames)
        self.blend_at_transitions(new_frames)
        if self.use_analytical_ik:
            self.apply_analytical_ik(new_frames)
        else:
            self.apply_ik_constraints(new_frames)
        self.blend_at_transitions(new_frames)
        return new_frames

    def _blend_around_frame_range(self, frames, start, end, joint_names):
        for joint_name in joint_names:
            transition_start = max(start - self.transition_window, 0)
            transition_end = min(end + self.transition_window, frames.shape[0]-1) - 1
            forward_steps = start - transition_start
            backward_steps = transition_end - end
            if joint_name == self.skeleton.root:
                if start > 0:
                    frames[:,:3] = smooth_translation_in_quat_frames(frames, start, self.translation_blend_window)
                temp_frame = min(end + 1, frames.shape[0]-1)
                frames[:,:3] = smooth_translation_in_quat_frames(frames, temp_frame, self.translation_blend_window)

            idx = self._ik.skeleton.animated_joints.index(joint_name)*4+3
            joint_parameter_indices = [idx, idx+1, idx+2, idx+3]
            if start > 0:
                create_transition_for_joints_using_slerp(frames, joint_parameter_indices, transition_start, start, forward_steps, BLEND_DIRECTION_FORWARD)
            create_transition_for_joints_using_slerp(frames, joint_parameter_indices, end, transition_end, backward_steps, BLEND_DIRECTION_BACKWARD)

    def apply_ik_constraints(self, frames):
        for frame_idx, constraints in list(self._constraints.items()):
            #print "process frame", frame_idx
            if 0 <= frame_idx < len(frames):
                frames[frame_idx] = self._ik.modify_frame(frames[frame_idx], constraints)

    def shift_root_to_reach_constraints(self, frames):
        root_positions = self.generate_root_positions_from_foot_constraints(frames)
        root_positions = smooth_root_positions(root_positions, self.root_smoothing_window)
        self.apply_root_constraints(frames, root_positions)

    def generate_root_positions_from_foot_constraints(self, frames):
        root_constraints = []
        for frame_idx, constraints in list(self._constraints.items()):
            if 0 <= frame_idx < len(frames):
                #print "adapt root", frame_idx, len(constraints)
                p = None
                if len(constraints) == 1:
                    p = self.generate_root_constraint_for_one_foot(frames[frame_idx], constraints[0])
                elif len(constraints) > 1:
                    p = self.generate_root_constraint_for_two_feet(frames[frame_idx], constraints[0], constraints[1])
                if p is None:
                    p = frames[frame_idx, :3]
                root_constraints.append(p)
        return np.array(root_constraints)

    def apply_root_constraints(self, frames, constraints):
        for frame_idx, p in enumerate(constraints):
            if p is not None:
                frames[frame_idx][:3] = p

    def generate_root_constraint_for_one_foot(self, frame, c):
        root = "pelvis"#self.skeleton.root
        offset = [0, self.skeleton.nodes[root].offset[0], -self.skeleton.nodes[root].offset[1]]
        root_pos = self.skeleton.nodes[root].get_global_position(frame)
        target_length = np.linalg.norm(c.position - root_pos)
        limb_length = self.get_limb_length(c.joint_name)
        if target_length >= limb_length:
            new_root_pos = (c.position + normalize(root_pos - c.position) * limb_length)
            #print "one constraint on ", c.joint_name, "- before", root_pos, "after", new_root_pos
            return new_root_pos - offset
            #frame[:3] = new_root_pos

        else:
            print("no change")

    def generate_root_constraint_for_two_feet(self, frame, constraint1, constraint2):
        """ Set the root position to the projection on the intersection of two spheres """
        root = "pelvis"  # self.skeleton.root
        p = self.skeleton.nodes[root].get_global_position(frame)
        offset = [0, self.skeleton.nodes[root].offset[0], -self.skeleton.nodes[root].offset[1]]

        t1 = np.linalg.norm(constraint1.position - p)
        t2 = np.linalg.norm(constraint2.position - p)

        c1 = constraint1.position
        r1 = self.get_limb_length(constraint1.joint_name)
        #p1 = c1 + r1 * normalize(p-c1)
        c2 = constraint2.position
        r2 = self.get_limb_length(constraint2.joint_name)
        #p2 = c2 + r2 * normalize(p-c2)
        if r1 < t1 and r2 < t2:
            #print "no root constraint"
            return None
        #print "adapt root for two constraints", constraint1.position, r1, constraint2.position, r2

        p_c = project_on_intersection_circle(p, c1, r1, c2, r2)
        return p_c - offset
        #if p_c is not None:
        #    frame[:3] = p_c

    def get_limb_length(self, joint_name):
        limb_length = np.linalg.norm(self.skeleton.nodes[joint_name].offset)
        limb_length += np.linalg.norm(self.skeleton.nodes[joint_name].parent.offset)
        return limb_length


    def apply_analytical_ik(self, frames):
        for frame_idx, constraints in list(self._constraints.items()):
            #print "process frame", frame_idx
            if 0 <= frame_idx < len(frames):
                for c in constraints:
                    if c.joint_name in list(self._ik_chains.keys()):
                        data = self._ik_chains[c.joint_name]
                        ik = AnalyticalLimbIK.init_from_dict(self.skeleton, c.joint_name, data)
                        frames[frame_idx] = ik.apply2(frames[frame_idx], c.position, c.orientation)
                        #delta = c.position -self.skeleton.nodes[c.joint_name].get_global_position(frames[frame_idx])
                        #heel_joint = "RightHeel"
                        #if c.joint_name == "LeftFoot":
                        #    heel_joint = "LeftHeel"
                        #print "delta",frame_idx,c.joint_name,np.linalg.norm(delta), c.position, self.skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
                    else:
                        print("could not find ik chain definition for ", c.joint_name)
                        frames[frame_idx] = self._ik.modify_frame(frames[frame_idx], constraints)

    def blend_at_transitions(self, frames):
        for frame_range, joint_names in list(self._blend_ranges.items()):
            start = frame_range[0]
            end = frame_range[1]
            #print "apply blending in range", frame_range, joint_names
            self._blend_around_frame_range(frames, start, end, joint_names)
        return frames

    def _shift_root_using_static_offset(self, frames, scene_interface):
        for idx, frame in enumerate(frames):
            x = frames[idx][0]
            z = frames[idx][2]
            target_ground_height = scene_interface.get_height(x, z)
            shift = target_ground_height - frames[idx][1]
            #print "root shift",idx, shift,frames[idx][1]
            frames[idx][1] += shift
