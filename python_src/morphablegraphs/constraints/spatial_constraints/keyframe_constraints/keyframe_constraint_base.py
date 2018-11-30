
from .. import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from ..spatial_constraint_base import SpatialConstraintBase


class KeyframeConstraintBase(SpatialConstraintBase):

    def __init__(self, constraint_desc, precision, weight_factor=1.0):
        assert "canonical_keyframe" in list(constraint_desc.keys())
        super(KeyframeConstraintBase, self).__init__(precision, weight_factor)
        self.semantic_annotation = constraint_desc["semanticAnnotation"]
        self.keyframe_label = constraint_desc["semanticAnnotation"]["keyframeLabel"]
        self.canonical_keyframe = constraint_desc["canonical_keyframe"]
        if "time" in list(constraint_desc.keys()) and constraint_desc["time"] is not None:
            print(constraint_desc["time"])
            self.desired_time = float(constraint_desc["time"])
        else:
            self.desired_time = None
        self.event_name = None
        self.event_target = None
        if "canonical_end_keyframe" in constraint_desc:
            self.canonical_end_keyframe = constraint_desc["canonical_end_keyframe"]
        else:
            self.canonical_end_keyframe = None
        self.keep_orientation = False
        if "keep_orientation" in constraint_desc:
            self.keep_orientation = constraint_desc["keep_orientation"]
        self.relative_joint_name = None
        if "relative_joint_name" in constraint_desc:
            self.relative_joint_name = constraint_desc["relative_joint_name"]

    def is_generated(self):
        return self.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and "generated" in list(self.semantic_annotation.keys())

    def extract_keyframe_index(self, time_function, frame_offset):
        #TODO: test and verify for all cases
        if time_function is not None:
            return frame_offset + int(time_function[self.canonical_keyframe]) + 1  # add +1 to map the frame correctly
        else:
            return frame_offset + self.canonical_keyframe