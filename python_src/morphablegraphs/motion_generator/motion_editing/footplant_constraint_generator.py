import numpy as np
from morphablegraphs.motion_generator.motion_editing.motion_grounding import MotionGroundingConstraint
from utils import get_average_joint_position, get_average_joint_direction

from scipy.interpolate import UnivariateSpline
LOCOMOTION_ACTIONS = ["walk", "carryRight", "carryLeft", "carryBoth"]
DEFAULT_WINDOW_SIZE = 20
LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"

FOOT_JOINTS = [LEFT_FOOT, RIGHT_FOOT, RIGHT_TOE, LEFT_TOE]


class FootplantConstraintGenerator(object):
    def __init__(self, skeleton, settings, ground_height=0):
        #left_foot = LEFT_FOOT, right_foot = RIGHT_FOOT, ground_height = 0, tolerance = 0.001
        self.skeleton = skeleton
        self.left_foot = settings["left_foot"]
        self.right_foot = settings["right_foot"]
        self.right_toe = settings["right_toe"]
        self.left_toe = settings["left_toe"]
        self.window = settings["window"]
        self.ground_height = ground_height
        self.tolerance = settings["tolerance"]
        self.foot_joints = FOOT_JOINTS
        # add joint offsets because we do not have a heel joint
        self.joint_offset = dict()
        self.joint_offset[self.right_toe] = 0
        self.joint_offset[self.left_toe] = 0
        self.joint_offset[LEFT_FOOT] = 10
        self.joint_offset[RIGHT_FOOT] = 10

    def get_vertical_acceleration(self, frames, joint_name):
        """ https://stackoverflow.com/questions/40226357/second-derivative-in-python-scipy-numpy-pandas
        """
        ys = []
        for frame in frames:
            y = self.skeleton.nodes[joint_name].get_global_position(frame)[1]
            ys.append(y)

        x = np.linspace(0, len(frames), len(frames))
        ys = np.array(ys)
        y_spl = UnivariateSpline(x, ys, s=0, k=4)
        y_spl_2d = y_spl.derivative(n=2)
        return ys, y_spl_2d(x)

    def detect_ground_contacts(self, frames, joints):
        """https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
        """
        ground_contacts = []
        for f in frames:
            ground_contacts.append([])

        for joint in joints:
            ys, ya = self.get_vertical_acceleration(frames, joint)
            o = self.joint_offset[joint]
            #print ys
            #close_to_ground = np.where(y - o - self.ground_height < self.tolerance)

            #zero_crossings = np.where(np.diff(np.sign(ya)))[0]
            #zero_crossings = np.diff(np.sign(ya)) != 0
            #print len(zero_crossings)
            #joint_ground_contacts = np.where(close_to_ground and zero_crossings)
            #print close_to_ground
            for frame_idx, y in enumerate(ys[:-1]):
                if y - o - self.ground_height < self.tolerance:#and zero_crossings[frame_idx]
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
        if RIGHT_FOOT and RIGHT_TOE in joint_names:
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

        if LEFT_FOOT and LEFT_TOE in joint_names:
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
        avg_p = get_average_joint_position(self.skeleton, frames, joint_name, start_frame, end_frame)
        for frame_idx in xrange(start_frame, end_frame):
            c = MotionGroundingConstraint(frame_idx, joint_name, avg_p, None)
            constraints[frame_idx] = []
            constraints[frame_idx].append(c)
        return constraints

    def create_ankle_constraints_from_heel_and_toe(self, frames, ankle_joint_name, start_frame, end_frame):
        """ create constraint on ankle position and orientation """
        constraints = dict()
        avg_p = get_average_joint_position(self.skeleton, frames, ankle_joint_name, start_frame, end_frame)
        avg_direction = None
        if len(self.skeleton.nodes[ankle_joint_name].children) > 0:
            child_joint_name = self.skeleton.nodes[ankle_joint_name].children[0].node_name
            avg_direction = get_average_joint_direction(self.skeleton, frames, ankle_joint_name, child_joint_name,
                                                        start_frame, end_frame)
        for frame_idx in xrange(start_frame, end_frame):
            c = MotionGroundingConstraint(frame_idx, ankle_joint_name, avg_p, avg_direction)
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

    def generate_old(self, motion_vector):
        constraints = dict()
        blend_ranges = dict()
        blend_ranges[self.right_foot] = []
        blend_ranges[self.left_foot] = []

        frames = motion_vector.frames
        graph_walk = motion_vector.graph_walk
        for step in graph_walk.steps:
            new_constraints = None
            start_frame = step.start_frame + self.window / 2
            end_frame = step.end_frame - self.window / 2
            if step.node_key[0] in LOCOMOTION_ACTIONS:
                if step.node_key[1] == "leftStance":

                    new_constraints = self.create_ankle_constraints_from_heel(frames, self.right_foot,
                                                                              start_frame, end_frame)
                    frame_range = start_frame, end_frame - 1  # the interpolation range must start at end_frame-1 because this is the last modified frame

                    blend_ranges[self.right_foot].append(frame_range)
                elif step.node_key[1] == "rightStance":
                    new_constraints = self.create_ankle_constraints_from_heel(frames, self.left_foot,
                                                                              start_frame, end_frame)
                    frame_range = start_frame, end_frame - 1  # the interpolation range must start at end_frame-1 because this is the last modified frame

                    blend_ranges[self.left_foot].append(frame_range)
            if new_constraints is None:
                continue
            for key, item in new_constraints.items():
                if key in constraints:
                    constraints[key] += new_constraints[key]
                else:
                    constraints[key] = new_constraints[key]
        return constraints, blend_ranges

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
