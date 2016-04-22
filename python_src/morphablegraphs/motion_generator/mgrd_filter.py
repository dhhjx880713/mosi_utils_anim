import numpy
import time
from .constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION
from ..external.transformations import quaternion_matrix, quaternion_from_matrix
try:
    from mgrd import Constraint, SemanticConstraint
    from mgrd import CartesianConstraint as MGRDCartesianConstraint
    from mgrd import ForwardKinematics as MGRDForwardKinematics
    from mgrd import score_splines_with_semantic_pose_constraints
    has_mgrd = True
except ImportError:
    print("Import failed")
    pass
    has_mgrd = False


class MGRDFilter(object):
    """ Implements the filter pipeline to estimate the best fit parameters for one motion primitive.
    """
    def __init__(self, pose_constraint_weights=(1,1)):
        self.pose_constraint_weights = pose_constraint_weights

    @staticmethod
    def transform_coeffs(qs, transform):
        for c in qs.coeffs:
            c[:3] = numpy.dot(transform, c[:3].tolist()+[1])[:3]
            c[3:7] = quaternion_from_matrix(numpy.dot(transform, quaternion_matrix(c[3:7])))

    @staticmethod
    def extract_cartesian_constraints(mp_constraints):
        mgrd_constraints = []
        for c in mp_constraints.constraints:
            if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
                position = [p if p is not None else 0 for p in c.position]
                #print "local", position, c.joint_name
                cartesian_constraint = MGRDCartesianConstraint(position, c.joint_name, c.weight_factor)
                mgrd_constraints.append(cartesian_constraint)
        return mgrd_constraints

    @staticmethod
    def score_samples_using_semantic_pose_distance_measure(motion_primitive, samples, semantic_pose_constraints,cartesian_constraints, weights=(1,1)):
        quat_splines = motion_primitive.create_multiple_spatial_splines(samples, joints=None)
        print "orientation of motion sample",quat_splines[-1].coeffs[0][3:7]
        # Evaluate semantic and time constraints
        labels = list(set(SemanticConstraint.get_constrained_labels(semantic_pose_constraints)) & set(motion_primitive.time.semantic_labels))
        time_splines = [motion_primitive.create_time_spline(svec, labels) for svec in samples]
        scores = numpy.zeros(len(samples))
        scores += score_splines_with_semantic_pose_constraints(quat_splines, time_splines, semantic_pose_constraints, weights)
        scores += MGRDCartesianConstraint.score_splines(quat_splines, cartesian_constraints)
        return scores   \

    @staticmethod
    def score_samples(motion_primitive, samples, semantic_pose_constraints, cartesian_constraints, weights=(1,1)):
        quat_splines = motion_primitive.create_multiple_spatial_splines(samples, joints=None)
        print "orientation of motion sample",quat_splines[-1].coeffs[0][3:7]
        # Evaluate semantic and time constraints
        labels = None#list(set(SemanticConstraint.get_constrained_labels(semantic_pose_constraints)) & set(motion_primitive.time.semantic_labels))
        time_splines = [motion_primitive.create_time_spline(svec, labels) for svec in samples]
        scores = numpy.zeros(len(samples))
        if len(semantic_pose_constraints) > 0:
            print "set weights to ", semantic_pose_constraints[0].weights
            scores += score_splines_with_semantic_pose_constraints(quat_splines, time_splines, semantic_pose_constraints, semantic_pose_constraints[0].weights)
        if len(cartesian_constraints) > 0:
            scores += MGRDCartesianConstraint.score_splines(quat_splines, cartesian_constraints)
        return scores

    @staticmethod
    def score_samples_cartesian(motion_primitive, samples, mp_constraints):
        """ Scores splines using only cartesian constraints.

        Args:
            motion_primitive (mgrd.MotionPrimitiveModel): the motion primitive to backproject the samples.
            samples(List<Array<float>>): list of samples generated from the motion primitive.
            mp_constraints (MotionPrimitiveConstraints>):  a set of motion primitive constraints.
            transform (Matrix4F):optional transformation matrix of the samples into the global coordinate system.

        Returns:
            Array<float>
        """
        if has_mgrd:
            cartesian_constraints = MGRDFilter.extract_cartesian_constraints(mp_constraints)
            if len(cartesian_constraints) > 0:
                quat_splines = motion_primitive.create_multiple_spatial_splines(samples, joints=None)
                # transform the splines if the constraints are not in the local coordinate system of the motion primitive
                if not mp_constraints.is_local and mp_constraints.aligning_transform is not None:
                    start = time.clock()
                    for qs in quat_splines:
                        MGRDFilter.transform_coeffs(qs, mp_constraints.aligning_transform)
                    print "transformed splines in", time.clock()-start, "seconds"
                return MGRDCartesianConstraint.score_splines(quat_splines, cartesian_constraints)
        else:
            print ("Error: MGRD was not correctly initialized")
            return [0]

    @staticmethod
    def score_samples_using_cartesian_constraints(motion_primitive, samples, constraints, transform=None):
        """ Scores splines using cartesian constraints only.

        Args:
            motion_primitive (mgrd.MotionPrimitiveModel): the motion primitive to backproject the samples.
            samples(List<Array<float>>): list of samples generated from the motion primitive.
            constraints (List<mgrd.CartesianConstraint>):  a list of cartesian constraints each describing the target position of a joint.
            transform (Matrix4F):optional transformation matrix of the samples into the global coordinate system.

        Returns:
            Array<float>
        """
        if has_mgrd:
            quat_splines = motion_primitive.create_multiple_spatial_splines(samples, joints=None)
            if transform is not None:
                for qs in quat_splines:
                    qs.transform_coeffs(transform)
            scores = MGRDCartesianConstraint.score_splines(quat_splines, constraints)
            return scores

    def score_samples_using_keyframe_constraints(self, motion_primitive, samples, constraints, transform=None):
        """ Selects constrained intervals from TimeSplines using SemanticConstraints and scores the splines with
            PoseConstraints using the constrained intervals as bounds to the closest point search.

        Args:
            motion_primitive (mgrd.MotionPrimitiveModel): the motion primitive to backproject the samples.
            samples(List<Array<float>>): a list of samples generated from the motion primitive.
            constraints (List<MGRDKeyframeConstraint>): a list of KeyframeConstraints each describing the target position and orientation of a joint for a semantic annotation
            pose_constraint_weights (Vec2F): weight factors for the position and orientation errors
            transform (Matrix4F): optional transformation matrix of the samples into the global coordinate system.

        Returns:
            Array<float>
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
            Array<float>

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

    @staticmethod
    def create_transformation_matrix(rot, trans, out):
        #print (rot, trans)
        # normalize quaternion
        s = 1.0 / numpy.linalg.norm(rot)

        x = rot[1] * s
        y = rot[2] * s
        z = rot[3] * s
        w = rot[0] * s

        x2 = x + x
        y2 = y + y
        z2 = z + z

        xx = x * x2
        yx = y * x2
        yy = y * y2
        zx = z * x2
        zy = z * y2
        zz = z * z2
        wx = w * x2
        wy = w * y2
        wz = w * z2

        out[0,0] = 1 - yy - zz
        out[0,1] = yx - wz
        out[0,2] = zx + wy
        out[0,3] = trans[0]

        out[1,0] = yx + wz
        out[1,1] = 1 - xx - zz
        out[1,2] = zy - wx
        out[1,3] = trans[1]

        out[2,0] = zx - wy
        out[2,1] = zy + wx
        out[2,2] = 1 - xx - yy
        out[2,3] = trans[2]

        out[3,0] = 0
        out[3,1] = 0
        out[3,2] = 0
        out[3,3] = 1