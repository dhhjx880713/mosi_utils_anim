import collections
import numpy as np
from numerical_ik_exp import NumericalInverseKinematicsExp
from analytical_inverse_kinematics import AnalyticalLimbIK
from ...animation_data.motion_blending import apply_slerp, apply_slerp2, BLEND_DIRECTION_FORWARD, BLEND_DIRECTION_BACKWARD
from skeleton_pose_model import SkeletonPoseModel


class MotionGroundingConstraint(object):
    def __init__(self, frame_idx, joint_name, position, direction):
        self.frame_idx = frame_idx
        self.joint_name = joint_name
        self.position = position
        self.direction = direction

    def evaluate(self, skeleton, q_frame):
        d = self.position - skeleton.nodes[self.joint_name].get_global_position(q_frame)
        return np.dot(d, d)


class IKConstraintSet(object):
    def __init__(self, frame_range, joint_names, positions):
        self.frame_range = frame_range
        self.joint_names = joint_names
        self.constraints = []
        for idx in xrange(frame_range[0], frame_range[1]):
            for idx, joint_name in enumerate(joint_names):
                c = MotionGroundingConstraint(idx, joint_name, positions[idx])
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
    for key, node in skeleton.nodes.items():
        if len(node.children) == 0:
            continue
        if not node.fixed:
            full_frame += frame[o:o+4].tolist()
            o += 4
        else:
            full_frame += node.rotation.tolist()
    return full_frame


class MotionGrounding(object):
    def __init__(self, skeleton, ik_settings, ik_chains):
        self.skeleton = skeleton
        self._ik = NumericalInverseKinematicsExp(skeleton, ik_settings)
        self._constraints = collections.OrderedDict()
        self.pose = SkeletonPoseModel(skeleton, False)
        self.transition_window = ik_settings["transition_window"]
        self._blend_ranges = collections.OrderedDict()
        self.use_analytical = True
        self._ik_chains = ik_chains

    def add_constraint(self, joint_name, position, frame_range):
        for frame_idx in xrange(*frame_range):
            c = IKConstraint(frame_idx, joint_name, position)
            if frame_idx not in self._constraints.keys():
                self._constraints[frame_idx] = []
            self._constraints[frame_idx].append(c)

    def add_blend_range(self, joint_names, frame_range):
        if frame_range not in self._constraints.keys():
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

    def _blend_around_frame_range(self, frames, start, end, joint_names):
        for joint_name in joint_names:
            idx = self._ik.skeleton.animated_joints.index(joint_name)*4+3
            joint_parameter_indices = [idx, idx+1, idx+2, idx+3]
            transition_start = max(start - self.transition_window, 0)
            transition_end = min(end + self.transition_window, frames.shape[0]) - 1
            steps = start - transition_start
            apply_slerp2(frames, joint_parameter_indices, transition_start, start, steps, BLEND_DIRECTION_FORWARD)
            steps = transition_end - end
            apply_slerp2(frames, joint_parameter_indices,  end, transition_end, steps, BLEND_DIRECTION_BACKWARD)

    def apply_ik_constraints(self, frames):
        for frame_idx, constraints in self._constraints.items():
            print "process frame", frame_idx
            if 0 <= frame_idx < len(frames):
                ref = frames[frame_idx]
                new_frame = self._ik.modify_frame(ref, constraints)
                frames[frame_idx] = new_frame

    def apply_analytical_ik(self, frames):
        for frame_idx, constraints in self._constraints.items():
            print "process frame", frame_idx
            if 0 <= frame_idx < len(frames):
                for c in constraints:
                    if c.joint_name in self._ik_chains.keys():
                        limb_root = self._ik_chains[c.joint_name]["root"]
                        limb_joint = self._ik_chains[c.joint_name]["joint"]
                        joint_axis = self._ik_chains[c.joint_name]["joint_axis"]
                        analytical_ik = AnalyticalLimbIK(self.skeleton, limb_root, limb_joint, c.joint_name, joint_axis)
                        frames[frame_idx] = analytical_ik.apply(frames[frame_idx], c.position)
                    else:
                        print "could not find ik chain definition for ", c.joint_name
                        ref = frames[frame_idx]
                        new_frame = self._ik.modify_frame(ref, constraints)
                        frames[frame_idx] = new_frame

    def blend_at_transitions(self, frames):
        for frame_range, joint_names in self._blend_ranges.items():
            start = frame_range[0]
            end = frame_range[1]
            self._blend_around_frame_range(frames, start, end, joint_names)
        return frames

    def run(self, motion_vector):
        new_frames = motion_vector.frames[:]
        if self.use_analytical:
            self.apply_analytical_ik(new_frames)
        else:
            self.apply_ik_constraints(new_frames)
        self.blend_at_transitions(new_frames)
        return new_frames


