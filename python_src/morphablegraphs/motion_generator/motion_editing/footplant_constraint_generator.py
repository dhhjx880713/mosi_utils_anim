import numpy as np
from morphablegraphs.motion_generator.motion_editing.motion_grounding import MotionGroundingConstraint
from constants import *

def merge_constraints(a,b):
    for key, item in b.items():
        if key in a:
            a[key] += b[key]
        else:
            a[key] = b[key]
    return a


class FootplantConstraintGenerator(object):
    def __init__(self, skeleton, settings, ground_height=0, add_heels=False):
        #left_foot = LEFT_FOOT, right_foot = RIGHT_FOOT, ground_height = 0, tolerance = 0.001
        self.skeleton = skeleton
        self.left_foot = settings["left_foot"]
        self.right_foot = settings["right_foot"]
        self.right_heel = RIGHT_HEEL
        self.left_heel = LEFT_HEEL
        if add_heels:

            offset = [0, -6.480602, 0]
            self.skeleton = add_heels_to_skeleton(self.skeleton, self.left_foot, self.right_foot, LEFT_HEEL, RIGHT_HEEL, offset)
            print self.skeleton.nodes[LEFT_HEEL].offset

        self.right_toe = settings["right_toe"]
        self.left_toe = settings["left_toe"]
        self.window = settings["window"]
        self.ground_height = ground_height
        self.tolerance = settings["tolerance"]
        self.foot_joints = FOOT_JOINTS
        # add joint offsets because we do not have a heel joint
        self.joint_offset = dict()
        for j in self.foot_joints:
            self.joint_offset[j] = 0
        #self.joint_offset[LEFT_FOOT] = 10
        #self.joint_offset[RIGHT_FOOT] = 10

    def get_vertical_acceleration(self, frames, joint_name):
        """ https://stackoverflow.com/questions/40226357/second-derivative-in-python-scipy-numpy-pandas
        """
        ps = []
        for frame in frames:
            p = self.skeleton.nodes[joint_name].get_global_position(frame)
            ps.append(p)
        ps = np.array(ps)
        x = np.linspace(0, len(frames), len(frames))
        ys = np.array(ps[:,1])
        y_spl = UnivariateSpline(x, ys, s=0, k=4)
        y_spl_2d = y_spl.derivative(n=2)
        return ps, y_spl_2d(x)

    def get_joint_height(self, frames,joints):
        joint_heights = dict()
        for joint in joints:
            joint_heights[joint] = self.get_vertical_acceleration(frames, joint)
            # print ys
            # close_to_ground = np.where(y - o - self.ground_height < self.tolerance)

            # zero_crossings = np.where(np.diff(np.sign(ya)))[0]
            # zero_crossings = np.diff(np.sign(ya)) != 0
            # print len(zero_crossings)
            # joint_ground_contacts = np.where(close_to_ground and zero_crossings)
            # print close_to_ground
        return joint_heights

    def detect_ground_contacts(self, frames, joints, create_plot=True):
        """https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
        """

        joint_heights = self.get_joint_height(frames, joints)
        if create_plot:
            plot_joint_heights(joint_heights)

        ground_contacts = []
        for f in frames:
            ground_contacts.append([])
        for joint in joints:
            ps, ya = joint_heights[joint]
            for frame_idx, p in enumerate(ps[:-1]):
                o = self.joint_offset[joint]
                print joint, frame_idx, p[1]
                if p[1] - o - self.ground_height < self.tolerance:#and zero_crossings[frame_idx]
                    ground_contacts[frame_idx].append(joint)
        return ground_contacts

    def generate(self, motion_vector):
        constraints = dict()
        blend_ranges = dict()
        blend_ranges[self.right_foot] = []
        blend_ranges[self.left_foot] = []

        ground_contacts = self.detect_ground_contacts(motion_vector.frames, self.foot_joints)
        for frame_idx, joint_names in enumerate(ground_contacts):
            constraints[frame_idx] = self.generate_ankle_constraints(motion_vector.frames, frame_idx, joint_names)

        return constraints, blend_ranges

    def generate_ankle_constraints(self, frames, frame_idx, joint_names):
        constraints = []
        if RIGHT_HEEL and RIGHT_TOE in joint_names:
            temp_c = self.create_ankle_constraints_from_heel_and_toe(frames, self.right_foot, frame_idx, frame_idx + 1)
            for c in temp_c[frame_idx]:
                constraints.append(c)
        elif RIGHT_FOOT in joint_names:
            temp_c = self.create_ankle_constraints_from_heel(frames, self.right_foot, frame_idx, frame_idx + 1)
            for c in temp_c[frame_idx]:
                constraints.append(c)
        elif RIGHT_TOE in joint_names:
            temp_c = self.create_ankle_constraints_from_toe(frames, self.right_foot, self.right_toe, frame_idx, frame_idx + 1)
            for c in temp_c[frame_idx]:
                constraints.append(c)

        if LEFT_HEEL and LEFT_TOE in joint_names:
            temp_c = self.create_ankle_constraints_from_heel_and_toe(frames, self.left_foot, frame_idx, frame_idx + 1)
            for c in temp_c[frame_idx]:
                constraints.append(c)
        elif LEFT_FOOT in joint_names:
            temp_c = self.create_ankle_constraints_from_heel(frames, self.left_foot, frame_idx, frame_idx + 1)
            for c in temp_c[frame_idx]:
                constraints.append(c)
        elif LEFT_TOE in joint_names:
            temp_c = self.create_ankle_constraints_from_toe(frames, self.left_foot, self.left_toe, frame_idx, frame_idx + 1)
            for c in temp_c[frame_idx]:
                constraints.append(c)

        return constraints



    def create_ankle_constraints_from_heel(self, frames, joint_name, start_frame, end_frame):
        """ create constraint on the ankle position without an orientation constraint"""
        constraints = dict()
        ca = get_average_joint_position(self.skeleton, frames, joint_name, start_frame, end_frame)
        #ca[1] += OFFSET
        for frame_idx in xrange(start_frame, end_frame):
            c = MotionGroundingConstraint(frame_idx, joint_name, ca, None)
            constraints[frame_idx] = []
            constraints[frame_idx].append(c)
        return constraints

    def create_ankle_constraints_from_heel_and_toe(self, frames, ankle_joint_name, start_frame, end_frame):
        """ create constraint on ankle position and orientation """
        constraints = dict()
        ca = get_average_joint_position(self.skeleton, frames, ankle_joint_name, start_frame, end_frame)
        #ca[1] += OFFSET
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
        ct[1] = self.ground_height
        pa = get_average_joint_position(self.skeleton, frames, ankle_joint_name, start_frame, end_frame)
        ph = get_average_joint_position(self.skeleton, frames, heel_joint_name, start_frame, end_frame)
        delta = ct - ph
        ca = pa + delta
        avg_direction = None
        print "constraint ankle at",pa, ct, start_frame, end_frame
        if len(self.skeleton.nodes[ankle_joint_name].children) > 0:
            child_joint_name = self.skeleton.nodes[ankle_joint_name].children[0].node_name
            avg_direction = get_average_joint_direction(self.skeleton, frames, ankle_joint_name, child_joint_name,
                                                        start_frame, end_frame)
        for frame_idx in xrange(start_frame, end_frame):
            c = MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, avg_direction)
            constraints[frame_idx] = []
            constraints[frame_idx].append(c)
        return constraints

    def create_ankle_constraints_from_toe(self, frames, ankle_joint_name, toe_joint_name, start_frame, end_frame):
        """ create a constraint on the ankle position based on the toe constraint position"""
        constraints = dict()
        ct = get_average_joint_position(self.skeleton, frames, toe_joint_name, start_frame, end_frame)
        for frame_idx in xrange(start_frame, end_frame):
            pa = self.skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
            pt = self.skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])
            translation = ct-pt
            ca = pa + translation
            c = MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None)
            constraints[frame_idx] = []
            constraints[frame_idx].append(c)
        return constraints

    def generate_from_graph_walk(self, motion_vector):
        # the interpolation range must start at end_frame-1 because this is the last modified frame
        constraints = dict()
        blend_ranges = dict()
        blend_ranges[self.right_foot] = []
        blend_ranges[self.left_foot] = []
        frames = motion_vector.frames
        graph_walk = motion_vector.graph_walk
        for step in graph_walk.steps:
            if step.node_key[0] in LOCOMOTION_ACTIONS:
                plant_range = self.get_plant_frame_range(step)
                for ankle_joint in plant_range.keys():
                    if plant_range[ankle_joint]["start"] is not None:
                        start_frame = plant_range[ankle_joint]["start"]
                        end_frame = plant_range[ankle_joint]["end"]
                        heel_joint = self.right_heel
                        if ankle_joint == self.left_foot:
                            heel_joint = self.left_heel
                        new_constraints = self.create_ankle_constraints_from_heel_and_toe2(frames,
                                                                                           ankle_joint,
                                                                                           heel_joint,
                                                                                           start_frame,
                                                                                           end_frame)
                        constraints = merge_constraints(constraints, new_constraints)
                        frame_range = start_frame, end_frame-1
                        blend_ranges[ankle_joint].append(frame_range)
                #self.ground_height+=10
                #frames[step.start_frame:step.end_frame,1] += 10
        return constraints, blend_ranges

    def get_plant_frame_range(self, step):
        start_frame = step.start_frame
        end_frame = step.end_frame + 1
        w = self.window
        end_offset = -5
        plant_range = dict()
        plant_range[self.left_foot] = dict()
        plant_range[self.right_foot] = dict()

        plant_range[self.left_foot]["start"] = None
        plant_range[self.left_foot]["end"] = None
        plant_range[self.right_foot]["start"] = None
        plant_range[self.right_foot]["end"] = None

        if step.node_key[1] == "beginLeftStance":
            plant_range[self.right_foot]["start"] = start_frame
            plant_range[self.right_foot]["end"] = end_frame - w / 2 + end_offset
            plant_range[self.left_foot]["start"] = start_frame
            plant_range[self.left_foot]["end"] = start_frame + 20

        elif step.node_key[1] == "beginRightStance":
            plant_range[self.left_foot]["start"] = start_frame
            plant_range[self.left_foot]["end"] = end_frame - w / 2 + end_offset
            plant_range[self.right_foot]["start"] = start_frame
            plant_range[self.right_foot]["end"] = start_frame + 20

        elif step.node_key[1] == "endLeftStance":
            plant_range[self.right_foot]["start"] = start_frame + w / 2
            plant_range[self.right_foot]["end"] = end_frame
            plant_range[self.left_foot]["start"] = end_frame - 20
            plant_range[self.left_foot]["end"] = end_frame

        elif step.node_key[1] == "endRightStance":
            plant_range[self.left_foot]["start"] = start_frame + w / 2
            plant_range[self.left_foot]["end"] = end_frame
            plant_range[self.right_foot]["start"] = end_frame - 20
            plant_range[self.right_foot]["end"] = end_frame

        elif step.node_key[1] == "leftStance":
            plant_range[self.right_foot]["start"] = start_frame + w / 2
            plant_range[self.right_foot]["end"] = end_frame - w / 2 + end_offset

        elif step.node_key[1] == "rightStance":
            plant_range[self.left_foot]["start"] = start_frame + w / 2
            plant_range[self.left_foot]["end"] = end_frame - w / 2 + end_offset
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
