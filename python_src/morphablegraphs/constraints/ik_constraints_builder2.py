import collections
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from ..animation_data.motion_editing.motion_editing import KeyframeConstraint

SUPPORTED_CONSTRAINT_TYPES = [SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION]


class IKConstraintsBuilder2(object):
    def __init__(self, action_name, motion_primitive_name, motion_state_graph, skeleton):
        self.action_name = action_name
        self.motion_primitive_name = motion_primitive_name
        self.motion_state_graph = motion_state_graph
        self.skeleton = skeleton

    def convert_to_ik_constraints(self, constraints, frame_offset=0, time_function=None, constrain_orientation=True):
        ik_constraints = collections.OrderedDict()
        for constraint in constraints:
            if constraint.constraint_type in SUPPORTED_CONSTRAINT_TYPES and "generated" not in constraint.semantic_annotation.keys():
                if time_function is not None:
                    keyframe = frame_offset + int(time_function[constraint.canonical_keyframe]) + 1
                else:
                    keyframe = frame_offset + int(constraint.canonical_keyframe)

                if constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                    ik_constraint = self._create_keyframe_ik_constraint(constraint, keyframe, constrain_orientation, look_at=True)
                    if keyframe not in ik_constraints:
                        ik_constraints[keyframe] = dict()
                    if constraint.joint_name not in ik_constraints[keyframe]:
                        ik_constraints[keyframe][constraint.joint_name] = []
                    ik_constraints[keyframe][constraint.joint_name] = ik_constraint
        return ik_constraints

    def _create_keyframe_ik_constraint(self, c, keyframe, constrain_orientation, look_at):
        print("create ik constraint v2", keyframe, c.position, c.orientation)
        if constrain_orientation:
            orientation = c.orientation
        else:
            orientation = None
        return KeyframeConstraint(keyframe, c.joint_name, c.position, orientation, look_at)