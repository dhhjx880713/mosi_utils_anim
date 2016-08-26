
from copy import copy
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE,SPATIAL_CONSTRAINT_TYPE_KEYFRAME_DIR_2D, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT, SPATIAL_CONSTRAINT_TYPE_CA_CONSTRAINT
from ik_constraints import JointIKConstraint, TwoJointIKConstraint

SUPPORTED_CONSTRAINT_TYPES = [SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION,
                              SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION,SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT]


class IKConstraintsBuilder(object):
    def __init__(self, action_name, motion_primitive_name, motion_state_graph, skeleton):
        self.action_name = action_name
        self.motion_primitive_name = motion_primitive_name
        self.motion_state_graph = motion_state_graph
        self.skeleton = skeleton

    def convert_to_ik_constraints(self, constraints,  frame_offset=0, time_function=None, constrain_orientation=True):
        ik_constraints_dict = dict()
        for constraint in constraints:
            ik_constraints, ik_constraint_types = self._create_ik_constraints(constraint, frame_offset, time_function, constrain_orientation)
            for ik_c, ik_c_type in zip(ik_constraints,ik_constraint_types):
                if ik_c.keyframe not in ik_constraints_dict.keys():
                    ik_constraints_dict[ik_c.keyframe] = dict()
                    ik_constraints_dict[ik_c.keyframe]["single"] = []
                    ik_constraints_dict[ik_c.keyframe]["multiple"] = []
                ik_constraints_dict[ik_c.keyframe][ik_c_type].append(ik_c)

        return ik_constraints_dict

    def _create_ik_constraints(self, constraint, frame_offset=0, time_function=None, constrain_orientation=True):

        ik_constraints = []
        ik_constraint_types = []
        if constraint.constraint_type in SUPPORTED_CONSTRAINT_TYPES and \
            "generated" not in constraint.semantic_annotation.keys():
            if time_function is not None:
                # add +1 to map the frame correctly TODO: test and verify for all cases
                keyframe = frame_offset + int(time_function[constraint.canonical_keyframe]) + 1
            else:
                keyframe = frame_offset + constraint.canonical_keyframe

            if constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and constraint.joint_name in self.skeleton.free_joints_map.keys():
                ik_constraint = self._create_keyframe_ik_constraint(constraint, keyframe, frame_offset,
                                                                    time_function, constrain_orientation, look_at=True)
                ik_constraints.append(ik_constraint)
                ik_constraint_types.append("single")
            elif constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT:
                ik_constraint = JointIKConstraint("Head", constraint.target_position, None, keyframe, [], frame_range=None, look_at=True, optimize=False)
                ik_constraints.append(ik_constraint)
                ik_constraint_types.append("single")

            elif constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION and \
                            constraint.joint_names[0] in self.skeleton.free_joints_map.keys() and \
                            constraint.joint_names[1] in self.skeleton.free_joints_map.keys():
                free_joints = self.skeleton.reduced_free_joints_map[constraint.joint_names[0]]
                ik_constraint = JointIKConstraint(constraint.joint_names[0], constraint.positions[0], None, keyframe, free_joints, look_at=False)
                ik_constraints.append(ik_constraint)
                ik_constraint_types.append("single")

                free_joints = self.skeleton.reduced_free_joints_map[constraint.joint_names[1]]
                ik_constraint = JointIKConstraint(constraint.joint_names[1], constraint.positions[1], None, keyframe, free_joints, look_at=False)
                ik_constraints.append(ik_constraint)
                ik_constraint_types.append("single")
                #ik_constraints.append(None)# TODO replace with TwoJointIKConstraint
                #ik_constraints.append("multiple")
                # free_joints = set()
                # for joint_name in self.joint_names:
                #    if joint_name in free_joints_map.keys():
                #        free_joints.update(free_joints_map[joint_name])

                # ik_constraint = TwoJointIKConstraint(c.joint_names, c.positions, c.target_center, c.target_delta, c.target_direction, keyframe)
                # ik_constraints[keyframe]["multiple"].append(ik_constraint)
        return ik_constraints, ik_constraint_types

    def _create_keyframe_ik_constraint(self, c, keyframe, frame_offset, time_function, constrain_orientation, look_at=True, optimize=True):
        free_joints = self.skeleton.free_joints_map[c.joint_name]
        frame_range = self._detect_frame_range(c, frame_offset, time_function)
        if frame_range is None:
            print "Did not find frame range for", c.keyframe_label
        if constrain_orientation:
            orientation = c.orientation
        else:
            orientation = None
        return JointIKConstraint(c.joint_name, c.position, orientation, keyframe, free_joints, frame_range=frame_range, look_at=look_at, optimize=optimize)

    def _detect_frame_range(self, c, frame_offset, time_function):
        frame_range = None
        annotated_regions = self.motion_state_graph.node_groups[self.action_name].motion_primitive_annotation_regions
        if self.motion_primitive_name in annotated_regions.keys():
            frame_range_annotation = annotated_regions[self.motion_primitive_name]
            if c.keyframe_label in frame_range_annotation.keys():
                frame_range = copy(frame_range_annotation[c.keyframe_label])
                range_start = frame_range[0]
                range_end = frame_range[1]
                if time_function is not None:
                    # add +1 to map the frame correctly TODO: test and verify for all cases
                    range_start = int(time_function[range_start]) + 1
                    range_end = int(time_function[range_end]) + 1
                frame_range[0] = frame_offset + range_start
                frame_range[1] = frame_offset + range_end
                # print "Found frame range for", c.keyframe_label, frame_range
        return frame_range