
class MGRDKeyframeConstraint(object):
    """  A combination of a PoseConstraint and a SemanticConstraint for the integration with interact.

    Attributes:
        point (Vec3F): point in global Cartesian space..
        orientation (Vec4F): global orientation of the point as quaternion.
        joint_name (str): name of the constrained joint.
        weight (float): an weight for a linear combination of errors by the motion filter.
        annotations (List<string>): annotations that should be met
        time (float): the time in seconds on which the semantic annotation should be found. (None checks for occurrence)

    """
    def __init__(self, pose_constraint, semantic_constraint):
        self.joint_name = pose_constraint.joint_name
        self.weight = pose_constraint.weight
        self.point = pose_constraint.point
        self.orientation = pose_constraint.orientation
        self.annotations = semantic_constraint.annotations
        self.time = semantic_constraint.time
