import collections
from .spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION
from ..animation_data.motion_editing.motion_editing import KeyframeConstraint

SUPPORTED_CONSTRAINT_TYPES = [SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION]


class IKConstraintsBuilder2(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton

    def convert_to_ik_constraints(self, constraints, frame_offset=0, time_function=None, constrain_orientation=True):
        ik_constraints = collections.OrderedDict()
        for c in constraints:
            if c.constraint_type not in SUPPORTED_CONSTRAINT_TYPES or "generated" in c.semantic_annotation.keys():
                print("skip unsupported constraint")
                continue
            start_frame_idx = self.get_global_frame_idx(c.canonical_keyframe, frame_offset, time_function)
            if c.canonical_end_keyframe is not None:
                print("apply ik constraint on region")
                end_frame_idx = self.get_global_frame_idx(c.canonical_end_keyframe, frame_offset, time_function)
            else:
                print("no end keyframe defined")
                end_frame_idx = start_frame_idx+1

            for frame_idx in range(start_frame_idx, end_frame_idx):
                ik_constraint = self.convert_mg_constraint_to_ik_constraint(frame_idx, c, constrain_orientation)

                if frame_idx == start_frame_idx:
                    ik_constraint.keep_orientation = c.keep_orientation

                if start_frame_idx < frame_idx and frame_idx < end_frame_idx -1:
                    ik_constraint.inside_region = True
                else:
                    ik_constraint.inside_region = False
                    if frame_idx == end_frame_idx-1:
                        ik_constraint.end_of_region = True

                if ik_constraint is not None:
                    if frame_idx not in ik_constraints:
                        ik_constraints[frame_idx] = dict()
                    if c.joint_name not in ik_constraints[frame_idx]:
                        ik_constraints[frame_idx][c.joint_name] = []
                    ik_constraints[frame_idx][c.joint_name] = ik_constraint

        return ik_constraints

    def get_global_frame_idx(self, mp_frame_idx, frame_offset, time_function):
        if time_function is not None:
            frame_idx = frame_offset + int(time_function[mp_frame_idx]) + 1
        else:
            frame_idx = frame_offset + int(mp_frame_idx)
        return frame_idx

    def convert_mg_constraint_to_ik_constraint(self, frame_idx, mg_constraint, constrain_orientation=False):
        if mg_constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
            ik_constraint = self._create_keyframe_ik_constraint(mg_constraint, frame_idx, constrain_orientation, look_at=True)
        elif mg_constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION:
            print("generate constraint with offset at", frame_idx, "for",mg_constraint.joint_name, mg_constraint.offset)
            ik_constraint = KeyframeConstraint(frame_idx, mg_constraint.joint_name, mg_constraint.position, None, True, mg_constraint.offset)
        else:
            ik_constraint = None
        return ik_constraint

    def _create_keyframe_ik_constraint(self, constraint, keyframe, constrain_orientation, look_at):
        #print("create ik constraint v2", keyframe, constraint.position, constraint.orientation)
        if constrain_orientation:
            orientation = constraint.orientation
        else:
            orientation = None
        return KeyframeConstraint(keyframe, constraint.joint_name, constraint.position, orientation, look_at)

    def generate_relative_constraint(self, keyframe, frame, joint_name, relative_joint_name):
        joint_pos = self.skeleton.nodes[joint_name].get_global_position(frame)
        rel_joint_pos = self.skeleton.nodes[relative_joint_name].get_global_position(frame)
        #create a keyframe constraint but indicate that it is a relative constraint
        ik_constraint = KeyframeConstraint(keyframe, relative_joint_name, None, None, None)
        ik_constraint.relative_parent_joint_name = joint_name
        ik_constraint.relative_offset = rel_joint_pos - joint_pos
        return ik_constraint

    def convert_to_ik_constraints_with_relative(self, frames, constraints, frame_offset=0, time_function=None,
                                                 constrain_orientation=True):
        ik_constraints = collections.OrderedDict()
        for c in constraints:
            if c.constraint_type not in SUPPORTED_CONSTRAINT_TYPES or "generated" in c.semantic_annotation.keys():
                print("skip unsupported constraint")
                continue
            start_frame_idx = self.get_global_frame_idx(c.canonical_keyframe, frame_offset, time_function)
            if c.canonical_end_keyframe is not None:
                print("apply ik constraint on region")
                end_frame_idx = self.get_global_frame_idx(c.canonical_end_keyframe, frame_offset, time_function)
            else:
                print("no end keyframe defined")
                end_frame_idx = start_frame_idx + 1

            for frame_idx in range(start_frame_idx, end_frame_idx):
                ik_constraint = self.convert_mg_constraint_to_ik_constraint(frame_idx, c, constrain_orientation)

                if frame_idx == start_frame_idx:
                    ik_constraint.keep_orientation = c.keep_orientation

                if start_frame_idx < frame_idx and frame_idx < end_frame_idx - 1:
                    ik_constraint.inside_region = True
                else:
                    ik_constraint.inside_region = False
                    if frame_idx == end_frame_idx - 1:
                        ik_constraint.end_of_region = True

                if ik_constraint is not None:
                    if frame_idx not in ik_constraints:
                        ik_constraints[frame_idx] = dict()
                    if c.joint_name not in ik_constraints[frame_idx]:
                        ik_constraints[frame_idx][c.joint_name] = []
                    ik_constraints[frame_idx][c.joint_name] = ik_constraint

                    # add also a relative constraint
                    if c.relative_joint_name is not None:
                        ik_constraint = self.generate_relative_constraint(frame_idx, frames[frame_idx],
                                                                          c.joint_name,
                                                                          c.relative_joint_name)
                        ik_constraints[frame_idx][c.relative_joint_name] = ik_constraint

        return ik_constraints
