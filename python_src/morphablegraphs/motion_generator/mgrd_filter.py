import numpy
import time
from mgrd import Constraint, SemanticConstraint
from mgrd.utils import ForwardKinematics as MGRDForwardKinematics


class MGRDFilter(object):
    """ Implements the filter pipeline to estimate the best fit parameters for one motion primitive.
    """
    def __init__(self, pose_constraint_weights = (1,1)):
        self.pose_constraint_weights = pose_constraint_weights

    def score_samples(self, motion_primitive, samples, constraints, transform=None):
        """ Selects constrained intervals from TimeSplines using SemanticConstraints and scores the splines with
            PoseConstraints using the constrained intervals as bounds to the closest point search.

        Args:
            motion_primitive (mgrd.MotionPrimitiveModel): the motion primitive to backproject the samples.
            samples(List<Array<float>>): a list of samples generated from the motion primitive.
            constraints (List<MGRDKeyframeConstraint>): a list of KeyframeConstraints each describing the target position and orientation of a joint for a semantic annotation
            pose_constraint_weights (Vec2F): weight factors for the position and orientation errors
            transform (Matrix4F): optional transformation matrix of the samples into the global coordinate system.

        Returns:
            float
        """

        # Evaluate semantic and time constraints
        labels = list(set(SemanticConstraint.get_constrained_labels(constraints)) & set(motion_primitive.time.semantic_labels))

        constrained_intervals = None
        if len(labels) > 0:
            print("Search time spline for semantic labels", labels)
            tsplines = [motion_primitive.create_time_spline(svec, labels) for svec in samples]
            start = time.clock()
            constrained_intervals = SemanticConstraint.get_constrained_intervals(tsplines, constraints)
            print("finished interval selection in ", time.clock()-start, "seconds")
            #TODO reduce number of samples that are evaluated using a threshold

        # Evaluate position and orientation constraints
        #TODO find out joints which are needed for FK before backprojection. The accelerated FK needs all joints.
        #joints = Constraint.get_constrained_joints(constraints)
        #joints = motion_primitive.skeleton.get_interesting_joints(joints)
        quat_splines = motion_primitive.create_multiple_spatial_splines(samples, joints=None)
        if transform is not None:
            for qs in quat_splines:
                qs.transform_coeffs(transform)
        scores = self.score_splines_with_pose_constraints(quat_splines, constraints, constrained_intervals)
        return scores


    def score_splines_with_pose_constraints(self, quat_splines, constraints, constrained_intervals=None):
        """

        Args:
            quat_splines (list<mgrd.QuatSpline>):
            constraints (list<mgrd.PoseConstraint>):
            constrained_intervals(list<Vec2f>):

        Returns:
            float

        """
        constrain_position = Constraint.is_constraining_position(constraints)
        constrain_orientation = Constraint.is_constraining_orientation(constraints)
        joints = Constraint.get_constrained_joints(constraints)
        n_splines = len(quat_splines)
        if not constrain_position and not constrain_orientation: #ignore empty constraints
            print("No constraint specified")
            return numpy.zeros(n_splines)
        else:
            print("Constrain spatial spline for joints", joints)
            scores = numpy.zeros(n_splines)
            for s_idx in range(n_splines):
                spline = MGRDFilter.convert_spline_to_global_coordinates(quat_splines[s_idx], constrain_position, constrain_orientation,joints)
                is_globaltf = constrain_position and constrain_orientation
                for c_idx, c in enumerate(constraints):
                    joint_spline = MGRDFilter.extract_joint_spline(spline, c, is_globaltf)
                    if constrained_intervals is not None:
                        interval_list = constrained_intervals[s_idx][c_idx]
                    else:
                        interval_list = [None]
                    scores[s_idx] += c.weight * self.score_interval_list(joint_spline, c, interval_list)
                    #print("score", s_idx, scores[s_idx])
        return scores

    def score_interval_list(self, joint_spline, constraint, interval_list):
        """ Evaluates a list of intervals depending on the objective function of the constraint.

        Args:
            joint_spline (mgrd.JointSpline):
            constraint (mgrd.SemanticConstraint):
            interval_list (List<Vec2f>):

        Returns:
            float
        """
        best_result = numpy.inf
        for interval in interval_list:
            temp_result = joint_spline.score_with_pose(point=constraint.point, orientation=constraint.orientation, weights=self.pose_constraint_weights, interval=interval)
            #target = numpy.hstack((constraint.point,constraint.orientation))
            #temp_result = joint_spline.dist_to_point(target, objective=constraint.objective, interval=interval)
            if temp_result < best_result:
                best_result = temp_result
        return best_result


    @staticmethod
    def convert_spline_to_global_coordinates(quat_spline, include_position, include_orientation, joints = None):
        spline = None
        if include_position and include_orientation:
            spline = quat_spline.to_globaltf(joints)
        elif include_position and not include_orientation:
            skeleton = quat_spline.model.motion_primitive.skeleton
            joint_indices = numpy.asarray(quat_spline.structure.get_joint_indices(skeleton, joints))
            fk = MGRDForwardKinematics(skeleton, joint_indices)
            spline = quat_spline.to_reduced_cartesian(fk, joints)
        elif not include_position and include_orientation:
            spline = quat_spline.to_global_quat(joints)
        return spline

    @staticmethod
    def extract_joint_spline(motion_spline, constraint, is_globaltf):
        """ Wrapper around MotionSpline.extract_joint_spline for GlobalTfSpline
        """
        if is_globaltf:
            if constraint.point is None and constraint.orientation is not None:
                joint_spline = motion_spline.extract_quaternion_joint_spline(constraint.joint_name)
            elif constraint.point is not None and constraint.orientation is None:
                joint_spline = motion_spline.extract_cartesian_joint_spline(constraint.joint_name)
            elif constraint.point is not None and constraint.orientation is not None:
                joint_spline = motion_spline.extract_joint_spline(constraint.joint_name)
        else:
            joint_spline = motion_spline.extract_joint_spline(constraint.joint_name)
        return joint_spline
