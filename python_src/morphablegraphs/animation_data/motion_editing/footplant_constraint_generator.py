import collections
import numpy as np
from ..skeleton_models import *
from motion_grounding import MotionGroundingConstraint
from utils import get_average_joint_position, get_average_joint_direction, get_joint_height, normalize, get_average_direction_from_target, \
    quaternion_from_vector_to_vector
from ...external.transformations import quaternion_from_matrix, quaternion_multiply, quaternion_matrix, quaternion_slerp


def merge_constraints(a,b):
    for key, item in b.items():
        if key in a:
            a[key] += b[key]
        else:
            a[key] = b[key]
    return a

def align_quaternion(q, ref_q):
    if np.dot(ref_q, q) < 0:
        return -q
    else:
        return q


def create_ankle_constraint(skeleton, frames, ankle_joint_name, heel_joint_name, toe_joint, frame_idx, end_frame, ground_height):
    """ create constraint on ankle position and orientation """
    c = get_average_joint_position(skeleton, frames, heel_joint_name, frame_idx, end_frame)
    c[1] = ground_height
    a = get_average_joint_position(skeleton, frames, ankle_joint_name, frame_idx, end_frame)
    h = get_average_joint_position(skeleton, frames, heel_joint_name, frame_idx, end_frame)
    target_heel_offset = a - h
    ca = c + target_heel_offset
    avg_direction = None
    if len(skeleton.nodes[ankle_joint_name].children) > 0:
        avg_direction = get_average_direction_from_target(skeleton, frames, ca, toe_joint,
                                                    frame_idx, end_frame)
        toe_length = np.linalg.norm(skeleton.nodes[toe_joint].offset)
        ct = ca + avg_direction * toe_length
        ct[1] = ground_height
        avg_direction = normalize(ct - ca)
    return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, avg_direction)


class FootplantConstraintGenerator(object):
    def __init__(self, skeleton, skeleton_def, settings, source_ground_height=0, target_ground_height=0):
        #left_foot = LEFT_FOOT, right_foot = RIGHT_FOOT, ground_height = 0, tolerance = 0.001
        self.skeleton = skeleton
        self.left_foot = skeleton_def["left_foot"]
        self.right_foot = skeleton_def["right_foot"]
        self.right_heel = skeleton_def["right_heel"]
        self.left_heel = skeleton_def["left_heel"]
        self.right_toe = skeleton_def["right_toe"]
        self.left_toe = skeleton_def["left_toe"]
        self.foot_joints = skeleton_def["foot_joints"]
        self.heel_offset = skeleton_def["heel_offset"]

        self.foot_definitions = {"right": {"heel": self.right_heel, "toe": self.right_toe, "ankle": self.right_foot},
                                 "left": {"heel": self.left_heel, "toe": self.left_toe, "ankle": self.left_foot}}

        self.window = settings["window"]
        self.source_ground_height = source_ground_height
        self.target_ground_height = target_ground_height

        self.tolerance = settings["tolerance"]
        self.constraint_generation_range = settings["constraint_range"]

        self.joint_offset = dict()
        for j in self.foot_joints:
            self.joint_offset[j] = 0
        self.position_constraint_buffer = dict()
        self.orientation_constraint_buffer = dict()
        self.velocity_tolerance = 0
        self.smoothing_constraints_window = settings["smoothing_constraints_window"]

    def detect_ground_contacts(self, frames, joints):
        """https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
        """

        joint_heights = get_joint_height(self.skeleton, frames, joints)
        #angular_velocities = get_angular_velocities(self.skeleton, frames,["RightUpLeg","LeftUpLeg"])
        ground_contacts = []
        for f in frames:
            ground_contacts.append([])
        for side in self.foot_definitions.keys():
            foot_joints = [self.foot_definitions[side]["heel"], self.foot_definitions[side]["toe"]]
            for joint in foot_joints:
                ps, yv, ya = joint_heights[joint]
                for frame_idx, p in enumerate(ps[:-1]):
                    o = self.joint_offset[joint]
                    velocity = np.sqrt(yv[frame_idx]*yv[frame_idx])
                    #if side == "right":
                    #    knee = "RightUpLeg"
                    #else:
                    #    knee = "LeftUpLeg"
                    #av = np.rad2deg(np.linalg.norm(angular_velocities[knee][frame_idx]))
                    #print joint, frame_idx, p[1], yv[frame_idx], av
                    print joint, p[1], yv[frame_idx]
                    if p[1] - o - self.source_ground_height < self.tolerance:# or velocity < self.velocity_tolerance and zero_crossings[frame_idx]
                        ground_contacts[frame_idx].append(joint)
        return self.filter_outliers(ground_contacts, joints)

    def filter_outliers(self, ground_contacts, joints):
        n_frames = len(ground_contacts)
        #print n_frames
        filtered_ground_contacts = [[] for idx in xrange(n_frames)]
        #print len(filtered_ground_contacts)
        filtered_ground_contacts[0] = ground_contacts[0]
        filtered_ground_contacts[-1] = ground_contacts[-1]
        frames_indices = xrange(1,n_frames-1)
        for frame_idx in frames_indices:
            filtered_ground_contacts[frame_idx] = []
            prev_frame = ground_contacts[frame_idx - 1]
            current_frame = ground_contacts[frame_idx]
            next_frame = ground_contacts[frame_idx + 1]
            #print "check", frame_idx, joints, prev_frame, current_frame, next_frame
            for joint in joints:
                if joint in current_frame:
                    if joint not in prev_frame and joint not in next_frame:
                        #print "outlier", frame_idx, joint
                        continue
                    else:
                        filtered_ground_contacts[frame_idx].append(joint)
        return filtered_ground_contacts

    def generate(self, motion_vector, ground_contacts=None):
        self.position_constraint_buffer = dict()
        self.orientation_constraint_buffer = dict()

        constraints = dict()
        if ground_contacts is None:
            ground_contacts = self.detect_ground_contacts(motion_vector.frames, self.foot_joints)
            #print "found ground contacts",len(ground_contacts), len(motion_vector.frames)
        # generate constraints
        for frame_idx, joint_names in enumerate(ground_contacts):
            constraints[frame_idx] = self.generate_ankle_constraints(motion_vector.frames, frame_idx, joint_names)

        self.set_smoothing_constraints(motion_vector.frames, constraints, self.smoothing_constraints_window)

        n_frames = len(motion_vector.frames)
        blend_ranges = self.generate_blend_ranges(constraints, n_frames)
        return constraints, blend_ranges

    def generate_blend_ranges(self, constraints, n_frames):
        blend_ranges = dict()
        blend_ranges[self.right_foot] = []
        blend_ranges[self.left_foot] = []

        state = dict()
        state[self.right_foot] = -1
        state[self.left_foot] = -1

        joint_names = [self.right_foot, self.left_foot]

        for frame_idx in xrange(n_frames):
            if frame_idx in constraints.keys():
                for j in joint_names:
                    constrained_joints = [c.joint_name for c in constraints[frame_idx]]
                    if j in constrained_joints:
                        if state[j] == -1:
                            # start new constrained range
                            blend_ranges[j].append([frame_idx, n_frames])
                            state[j] = len(blend_ranges[j])-1
                        else:
                            # update constrained range
                            idx = state[j]
                            blend_ranges[j][idx][1] = frame_idx
                    else:
                        state[j] = -1  # stop constrained range no constraint defined
            else:
                for c in joint_names:
                    state[c.joint_name] = -1  # stop constrained range no constraint defined
        return blend_ranges

    def generate_ankle_constraints(self, frames, frame_idx, joint_names):
        self.position_constraint_buffer[frame_idx] = dict()
        self.orientation_constraint_buffer[frame_idx] = dict()

        new_constraints = []
        end_frame = frame_idx + self.constraint_generation_range
        for side in self.foot_definitions:
            foot_joints = self.foot_definitions[side]
            c = None
            if foot_joints["heel"] in joint_names and foot_joints["toe"] in joint_names:
                c = self.generate_ankle_constraint(frames, foot_joints["ankle"], foot_joints["heel"], foot_joints["toe"], frame_idx, end_frame)
            elif foot_joints["heel"] in joint_names:
                c = self.generate_ankle_constraint_from_heel(frames, foot_joints["ankle"], foot_joints["heel"], frame_idx, end_frame)
            elif foot_joints["toe"] in joint_names:
                c = self.generate_ankle_constraint_from_toe(frames, foot_joints["ankle"], foot_joints["toe"], frame_idx, end_frame)
            if c is not None:
                #print "generated constraint for", side, "at", frame_idx, joint_names
                new_constraints.append(c)
        return new_constraints

    def generate_ankle_constraint(self, frames, ankle_joint_name, heel_joint_name, toe_joint, frame_idx, end_frame):
        """ create constraint on ankle position and orientation """
        # get target global ankle orientation based on the direction between grounded heel and toe
        ct = self.get_previous_joint_position_from_buffer(frames, frame_idx, end_frame, toe_joint)
        ct[1] = self.target_ground_height
        ch = self.get_previous_joint_position_from_buffer(frames, frame_idx, end_frame, heel_joint_name)
        ch[1] = self.target_ground_height
        target_direction = normalize(ct - ch)
        t = self.skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
        h = self.skeleton.nodes[heel_joint_name].get_global_position(frames[frame_idx])
        original_direction = normalize(t-h)

        global_delta_q = quaternion_from_vector_to_vector(original_direction, target_direction)
        global_delta_q = normalize(global_delta_q)

        m = self.skeleton.nodes[ankle_joint_name].get_global_matrix(frames[frame_idx])
        m[:3, 3] = [0, 0, 0]
        oq = quaternion_from_matrix(m)
        oq = normalize(oq)
        orientation = normalize(quaternion_multiply(global_delta_q, oq))

        self.orientation_constraint_buffer[frame_idx][ankle_joint_name] = orientation  # save orientation to buffer

        # set target ankle position based on the  grounded heel and the global target orientation of the ankle
        m = quaternion_matrix(orientation)[:3,:3]
        target_heel_offset = np.dot(m, self.heel_offset)
        ca = ch - target_heel_offset
        print "set ankle constraint both", ch, ca, target_heel_offset
        return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, orientation)

    def generate_ankle_constraint_from_heel(self, frames, ankle_joint_name, heel_joint_name, frame_idx, end_frame):
        """ create constraint on the ankle position without an orientation constraint"""
        #print "add ankle constraint from heel"
        ch = self.get_previous_joint_position_from_buffer(frames, frame_idx, end_frame, heel_joint_name)
        a = self.skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
        h = self.skeleton.nodes[heel_joint_name].get_global_position(frames[frame_idx])
        ch[1] = self.target_ground_height  # set heel constraint on the ground
        target_heel_offset = a - h  # difference between unmodified heel and ankle
        ca = ch + target_heel_offset  # move ankle so heel is on the ground
        print "set ankle constraint single", ch, ca, target_heel_offset

        return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, None)

    def generate_ankle_constraint_from_toe(self, frames, ankle_joint_name, toe_joint_name, frame_idx, end_frame):
        """ create a constraint on the ankle position based on the toe constraint position"""
        #print "add toe constraint"
        c = self.get_previous_joint_position_from_buffer(frames, frame_idx, end_frame, toe_joint_name)
        a = self.skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
        t = self.skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])

        c[1] = self.target_ground_height  # set toe constraint on the ground
        target_toe_offset = a - t  # difference between unmodified toe and ankle at the frame
        ca = c + target_toe_offset  # move ankle so toe is on the ground
        return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, None)

    def set_smoothing_constraints(self, frames, constraints, window):
        """ Set orientation constraints to singly constrained frames to smooth the foot orientation (Section 4.2)
        """
        for frame_idx in constraints.keys():
            for c in constraints[frame_idx]:
                if c.joint_name not in self.orientation_constraint_buffer[frame_idx]:  # singly constrained
                    backward_hit = None
                    forward_hit = None

                    start_window = max(frame_idx - window, 0)
                    end_window = min(frame_idx + window, len(frames))
                    # look backward
                    for f in xrange(start_window, frame_idx):
                        if c.joint_name in self.orientation_constraint_buffer[f]:
                            backward_hit = f
                            break
                    # look forward
                    for f in xrange(frame_idx, end_window):
                        if c.joint_name in self.orientation_constraint_buffer[f]:
                            forward_hit = f
                            break

                    # update q
                    #print "update q", frame_idx, forward_hit, backward_hit
                    blend = lambda x: 2 * x * x * x - 3 * x * x + 1
                    iq = [1, 0, 0, 0]
                    oq = self.get_global_orientation(c.joint_name, frames[frame_idx])
                    if backward_hit is not None and forward_hit is not None:

                        bq = self.orientation_constraint_buffer[backward_hit][c.joint_name]
                        j = frame_idx - backward_hit
                        bw = (j + 1) / (window + 1)
                        bdeltaq = normalize(quaternion_slerp(iq, bq, blend(bw), spin=0, shortestpath=False))

                        k = forward_hit - frame_idx
                        fw = (k + 1) / (window + 1)
                        fq = self.orientation_constraint_buffer[forward_hit][c.joint_name]
                        fdeltaq = normalize(quaternion_slerp(fq, iq, blend(fw), spin=0, shortestpath=False))

                        w = (j + 1) / (j + k + window + 1)
                        global_delta_q = normalize(quaternion_slerp(bdeltaq, fdeltaq, blend(w), spin=0, shortestpath=False))

                        c.orientation = normalize(quaternion_multiply(global_delta_q, oq))

                    elif backward_hit is not None:
                        bq = self.orientation_constraint_buffer[backward_hit][c.joint_name]
                        j = frame_idx - backward_hit
                        w = (j + 1) / (window + 1)
                        #iq = align_quaternion(iq, bq)
                        global_delta_q = normalize(quaternion_slerp(iq, bq, blend(w), spin=0, shortestpath=False))

                        #c.orientation = normalize(quaternion_multiply(global_delta_q, oq))
                    elif forward_hit is not None:
                        k = forward_hit - frame_idx
                        w = (k + 1) / (window + 1)
                        fq = self.orientation_constraint_buffer[forward_hit][c.joint_name]
                        #iq = align_quaternion(iq, fq)
                        global_delta_q = normalize(quaternion_slerp(fq, iq, blend(w), spin=0, shortestpath=False))

                        c.orientation = normalize(quaternion_multiply(global_delta_q, oq))

    def get_global_orientation(self, joint_name, frame):
        m = self.skeleton.nodes[joint_name].get_global_matrix(frame)
        m[:3, 3] = [0, 0, 0]
        return normalize(quaternion_from_matrix(m))

    def get_previous_joint_position_from_buffer(self, frames, frame_idx, end_frame, joint_name):
        """ Gets the joint position of the previous frame from the buffer if it exists.
            otherwise the position is calculated for the current frame and updated in the buffer
        """
        prev_frame_idx = frame_idx - 1
        prev_p = self.get_joint_position_from_buffer(prev_frame_idx, joint_name)
        if prev_p is not None:
            return prev_p
        else:
            self.update_joint_position_in_buffer(frames, frame_idx, end_frame, joint_name)
            p = self.position_constraint_buffer[frame_idx][joint_name]
            p[1] = self.target_ground_height
            print "joint constraint",joint_name, p, self.target_ground_height
            return p

    def get_joint_position_from_buffer(self, frame_idx, joint_name):
        if frame_idx not in self.position_constraint_buffer.keys():
            return None
        if joint_name not in self.position_constraint_buffer[frame_idx].keys():
            return None
        return self.position_constraint_buffer[frame_idx][joint_name]

    def update_joint_position_in_buffer(self, frames, frame_idx, end_frame, joint_name):
        end_frame = min(end_frame, frames.shape[0])
        if frame_idx not in self.position_constraint_buffer.keys():
            self.position_constraint_buffer[frame_idx] = dict()
        if joint_name not in self.position_constraint_buffer[frame_idx].keys():
            p = get_average_joint_position(self.skeleton, frames, joint_name, frame_idx, end_frame)
            self.position_constraint_buffer[frame_idx][joint_name] = p

    def generate_ankle_constraints_legacy(self, frames, frame_idx, joint_names, prev_constraints, prev_joint_names):
        end_frame = frame_idx + 10
        new_constraints = dict()
        temp_constraints = {"left": None, "right": None}
        if self.right_heel and self.right_toe in joint_names:
            if self.right_heel and self.right_toe in prev_joint_names:
                temp_constraints["right"] = prev_constraints["right"]
            else:
                temp_constraints["right"] = self.create_ankle_constraints_from_heel_and_toe2(frames, self.right_foot, self.right_heel, frame_idx, end_frame)
        if self.left_heel and self.left_toe in joint_names:
            if joint_names == prev_joint_names:
                temp_constraints["left"] = prev_constraints["left"]
            else:
                temp_constraints["left"] = self.create_ankle_constraints_from_heel_and_toe2(frames, self.left_foot,self.left_heel,  frame_idx, end_frame)
        for side in temp_constraints:
            new_constraints[side] = temp_constraints[side][0]
        return new_constraints

    def  create_ankle_constraints_from_heel_and_toe(self, frames, ankle_joint_name, start_frame, end_frame):
        """ create constraint on ankle position and orientation """
        constraints = dict()
        ca = get_average_joint_position(self.skeleton, frames, ankle_joint_name, start_frame, end_frame)
        avg_direction = None
        if len(self.skeleton.nodes[ankle_joint_name].children) > 0:
            child_joint_name = self.skeleton.nodes[ankle_joint_name].children[0].node_name
            avg_direction = get_average_joint_direction(self.skeleton, frames, ankle_joint_name, child_joint_name,
                                                        start_frame, end_frame)
        for frame_idx in xrange(start_frame, end_frame):
            c = MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, avg_direction)
            constraints[frame_idx] = []
            constraints[frame_idx].append(c)
        return constraints

    def create_ankle_constraints_from_heel_and_toe2(self, frames, ankle_joint_name, heel_joint_name, start_frame, end_frame):
        """ create constraint on ankle position and orientation """
        constraints = dict()
        ct = get_average_joint_position(self.skeleton, frames, heel_joint_name, start_frame, end_frame)
        ct[1] = self.target_ground_height
        pa = get_average_joint_position(self.skeleton, frames, ankle_joint_name, start_frame, end_frame)
        ph = get_average_joint_position(self.skeleton, frames, heel_joint_name, start_frame, end_frame)
        delta = ct - ph
        ca = pa + delta
        avg_direction = None
        if len(self.skeleton.nodes[ankle_joint_name].children) > 0:
            child_joint_name = self.skeleton.nodes[ankle_joint_name].children[0].node_name
            avg_direction = get_average_joint_direction(self.skeleton, frames, ankle_joint_name, child_joint_name,
                                                        start_frame, end_frame)
        print "constraint ankle at",pa, ct,ca, start_frame, end_frame, avg_direction
        for frame_idx in xrange(start_frame, end_frame):
            c = MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, avg_direction)
            constraints[frame_idx] = []
            constraints[frame_idx].append(c)
        return constraints

    def generate_from_graph_walk(self, motion_vector):
        # the interpolation range must start at end_frame-1 because this is the last modified frame
        self.position_constraint_buffer = dict()
        self.orientation_constraint_buffer = dict()
        constraints = dict()
        for frame_idx in xrange(motion_vector.n_frames):
            self.position_constraint_buffer[frame_idx] = dict()
            self.orientation_constraint_buffer[frame_idx] = dict()
            constraints[frame_idx] = []

        blend_ranges = dict()
        blend_ranges[self.right_foot] = []
        blend_ranges[self.left_foot] = []
        frames = motion_vector.frames
        graph_walk = motion_vector.graph_walk
        for idx, step in enumerate(graph_walk.steps):
            if step.end_frame-step.start_frame <= 0:
                print "small frame range ", idx, step.node_key, step.start_frame, step.end_frame

                continue
            if step.node_key[0] in LOCOMOTION_ACTIONS:
                plant_range = self.get_plant_frame_range(step)
                for side in plant_range.keys():
                    if plant_range[side]["start"] is not None:
                        start_frame = plant_range[side]["start"]
                        end_frame = plant_range[side]["end"]
                        foot_joints = self.foot_definitions[side]
                        c = self.generate_ankle_constraint(frames, foot_joints["ankle"], foot_joints["heel"],
                                                           foot_joints["toe"], start_frame, end_frame)
                        new_constraints = dict()
                        for frame_idx in xrange(start_frame, end_frame):
                            new_constraints[frame_idx] = []
                            new_constraints[frame_idx].append(c)
                        constraints = merge_constraints(constraints, new_constraints)
                        frame_range = start_frame, end_frame-2
                        blend_ranges[foot_joints["ankle"]].append(frame_range)
        #print "b", constraints.keys()
        constraints = collections.OrderedDict(sorted(constraints.items()))
        #print "a", constraints.keys()
        return constraints, blend_ranges

    def get_plant_frame_range(self, step):
        start_frame = step.start_frame
        end_frame = step.end_frame + 1
        w = self.window
        end_offset = -5
        plant_range = dict()
        L = "left"
        R = "right"
        plant_range[L] = dict()
        plant_range[R] = dict()

        plant_range[L]["start"] = None
        plant_range[L]["end"] = None
        plant_range[R]["start"] = None
        plant_range[R]["end"] = None

        if step.node_key[1] == "beginLeftStance":
            plant_range[R]["start"] = start_frame
            plant_range[R]["end"] = end_frame - w / 2 + end_offset
            plant_range[L]["start"] = start_frame
            plant_range[L]["end"] = start_frame + 20

        elif step.node_key[1] == "beginRightStance":
            plant_range[L]["start"] = start_frame
            plant_range[L]["end"] = end_frame - w / 2 + end_offset
            plant_range[R]["start"] = start_frame
            plant_range[R]["end"] = start_frame + 20

        elif step.node_key[1] == "endLeftStance":
            plant_range[R]["start"] = start_frame + w / 2
            plant_range[R]["end"] = end_frame
            plant_range[L]["start"] = end_frame - 20
            plant_range[L]["end"] = end_frame

        elif step.node_key[1] == "endRightStance":
            plant_range[L]["start"] = start_frame + w / 2
            plant_range[L]["end"] = end_frame
            plant_range[R]["start"] = end_frame - 20
            plant_range[R]["end"] = end_frame

        elif step.node_key[1] == "leftStance":
            plant_range[R]["start"] = start_frame + w / 2
            plant_range[R]["end"] = end_frame - w / 2 + end_offset

        elif step.node_key[1] == "rightStance":
            plant_range[L]["start"] = start_frame + w / 2
            plant_range[L]["end"] = end_frame - w / 2 + end_offset
        return plant_range

    def create_foot_plant_constraints_old(self, frames, joint_name, start_frame, end_frame):
        """ create a constraint based on the average position in the frame range"""
        constraints = dict()
        avg_p = get_average_joint_position(self.skeleton, frames, joint_name, start_frame, end_frame)
        avg_direction = None
        if len(self.skeleton.nodes[joint_name].children) > 0:
            child_joint_name = self.skeleton.nodes[joint_name].children[0].node_name
            avg_direction = get_average_joint_direction(self.skeleton, frames, joint_name, child_joint_name, start_frame, end_frame)
        print joint_name, avg_p, avg_direction

        for frame_idx in xrange(start_frame, end_frame):
            c = MotionGroundingConstraint(frame_idx, joint_name, avg_p, avg_direction)
            constraints[frame_idx] = []
            constraints[frame_idx].append(c)
        return constraints

    def generate_from_graph_walk_(self, motion_vector):
        # the interpolation range must start at end_frame-1 because this is the last modified frame
        constraints = dict()
        blend_ranges = dict()
        blend_ranges[self.right_foot] = []
        blend_ranges[self.left_foot] = []

        frames = motion_vector.frames
        graph_walk = motion_vector.graph_walk
        for step in graph_walk.steps:
            #158 -  103 = 55
            end_offset = -5
            if step.node_key[0] in LOCOMOTION_ACTIONS:
                if step.node_key[1] == "beginLeftStance":
                    plant_start_frame = 0
                    plant_end_frame = step.end_frame - self.window / 2 + end_offset
                    new_constraints = self.create_ankle_constraints_from_heel_and_toe2(frames, self.right_foot,
                                                                                       self.right_heel,
                                                                                       plant_start_frame,
                                                                                       plant_end_frame)
                    constraints = merge_constraints(constraints, new_constraints)
                elif step.node_key[1] == "beginRightStance":
                    plant_start_frame = 0
                    plant_end_frame = step.end_frame - self.window / 2 + end_offset
                    new_constraints = self.create_ankle_constraints_from_heel_and_toe2(frames, self.left_foot,
                                                                                       self.left_heel,
                                                                                       plant_start_frame,
                                                                                       plant_end_frame)
                    constraints = merge_constraints(constraints, new_constraints)
                    frame_range = plant_start_frame, plant_end_frame - 1
                    blend_ranges[self.right_foot].append(frame_range)

                elif step.node_key[1] == "endLeftStance":
                    plant_start_frame = step.start_frame + self.window / 2
                    plant_end_frame = step.end_frame
                    new_constraints = self.create_ankle_constraints_from_heel_and_toe2(frames, self.right_foot,
                                                                                       self.right_heel,
                                                                                       plant_start_frame,
                                                                                       plant_end_frame)
                    constraints = merge_constraints(constraints, new_constraints)
                elif step.node_key[1] == "endRightStance":
                    plant_start_frame = step.start_frame + self.window / 2
                    plant_end_frame = step.end_frame
                    new_constraints = self.create_ankle_constraints_from_heel_and_toe2(frames, self.left_foot,
                                                                                       self.left_heel,
                                                                                       plant_start_frame,
                                                                                       plant_end_frame)
                    constraints = merge_constraints(constraints, new_constraints)
                    frame_range = plant_start_frame, plant_end_frame - 1
                    blend_ranges[self.right_foot].append(frame_range)
                elif step.node_key[1] == "leftStance":

                    plant_start_frame = step.start_frame + self.window / 2
                    plant_end_frame = step.end_frame - self.window / 2 + end_offset
                    #new_constraints = self.create_ankle_constraints_from_heel(frames, self.right_foot, start_frame, end_frame)
                    #constraints = merge_constraints(constraints, new_constraints)

                    new_constraints = self.create_ankle_constraints_from_heel_and_toe2(frames, self.right_foot, self.right_heel,plant_start_frame, plant_end_frame)
                    constraints = merge_constraints(constraints, new_constraints)

                    #new_constraints = self.create_ankle_constraints_from_toe(frames, self.right_foot, self.right_toe, start_frame, end_frame)
                    #constraints = merge_constraints(constraints, new_constraints)

                    frame_range = plant_start_frame, plant_end_frame - 1
                    blend_ranges[self.right_foot].append(frame_range)

                elif step.node_key[1] == "rightStance":

                    plant_start_frame = step.start_frame + self.window / 2
                    plant_end_frame = step.end_frame - self.window / 2  + end_offset

                    #new_constraints = self.create_ankle_constraints_from_heel(frames, self.left_foot, start_frame, end_frame)
                    #constraints = merge_constraints(constraints, new_constraints)

                    new_constraints = self.create_ankle_constraints_from_heel_and_toe2(frames, self.left_foot, self.left_heel, plant_start_frame, plant_end_frame)
                    constraints = merge_constraints(constraints, new_constraints)

                    #new_constraints = self.create_ankle_constraints_from_toe(frames, self.left_foot, self.left_toe, start_frame,end_frame)
                    #constraints = merge_constraints(constraints, new_constraints)

                    frame_range = plant_start_frame, plant_end_frame - 1
                    blend_ranges[self.left_foot].append(frame_range)
        return constraints, blend_ranges