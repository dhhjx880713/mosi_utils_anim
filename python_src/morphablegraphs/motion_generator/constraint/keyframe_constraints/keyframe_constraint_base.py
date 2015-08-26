__author__ = 'herrmann'


from ..spatial_constraint_base import SpatialConstraintBase


class KeyframeConstraintBase(SpatialConstraintBase):

    def __init__(self, constraint_desc, precision, weight_factor=1.0):
        super(KeyframeConstraintBase, self).__init__(precision, weight_factor)
        self.semantic_annotation = constraint_desc["semanticAnnotation"]
