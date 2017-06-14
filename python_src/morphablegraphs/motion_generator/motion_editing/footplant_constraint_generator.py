from morphablegraphs.motion_generator.motion_editing.motion_grounding import MotionGroundingConstraint
from utils import get_average_joint_position, get_average_joint_direction
LOCOMOTION_ACTIONS = ["walk", "carryRight", "carryLeft", "carryBoth"]
DEFAULT_WINDOW_SIZE = 10
LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"


class FootplantConstraintGenerator(object):
    def __init__(self, skeleton, left_foot=LEFT_FOOT, right_foot=RIGHT_FOOT, ground_height=0):
        self.skeleton = skeleton
        self.left_foot = left_foot
        self.right_foot = right_foot
        self.window = DEFAULT_WINDOW_SIZE
        self.ground_height = ground_height

    def generate(self, motion_vector):
        constraints = dict()
        frames = motion_vector.frames
        graph_walk = motion_vector.graph_walk
        for step in graph_walk.steps:
            new_constraints = None
            if step.node_key[0] in LOCOMOTION_ACTIONS:
                if step.node_key[1] == "leftStance":
                    new_constraints = self.create_foot_plant_constraints(frames, self.right_foot, step.start_frame, step.end_frame)
                elif step.node_key[1] == "rightStance":
                    new_constraints = self.create_foot_plant_constraints(frames, self.left_foot, step.start_frame, step.end_frame)
            if new_constraints is None:
                continue
            for key, item in new_constraints.items():
                if key in constraints:
                    constraints[key] += new_constraints[key]
                else:
                    constraints[key] = new_constraints[key]
        return constraints

    def create_foot_plant_constraints(self,  frames, joint_name, start_frame, end_frame):
        """ create a constraint based on the average position in the frame range"""
        constraints = dict()
        start_frame = start_frame + self.window/2
        end_frame = end_frame -self.window/2
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
