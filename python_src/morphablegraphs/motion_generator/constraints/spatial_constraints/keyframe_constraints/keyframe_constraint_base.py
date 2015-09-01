__author__ = 'herrmann'


from ..spatial_constraint_base import SpatialConstraintBase


class KeyframeConstraintBase(SpatialConstraintBase):

    def __init__(self, constraint_desc, precision, weight_factor=1.0):
        assert "canonical_keyframe" in constraint_desc.keys()
        super(KeyframeConstraintBase, self).__init__(precision, weight_factor)
        self.semantic_annotation = constraint_desc["semanticAnnotation"]
        self.canonical_keyframe = constraint_desc["canonical_keyframe"]
        if "time" in constraint_desc.keys() and constraint_desc["time"] is not None:
            print constraint_desc["time"]
            self.desired_time = float(constraint_desc["time"])
        else:
            self.desired_time = None
