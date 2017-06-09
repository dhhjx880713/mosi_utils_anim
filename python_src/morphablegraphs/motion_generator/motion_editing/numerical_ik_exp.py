""" Numerical IK using exponential displacement map
 based on Section 5.1 of
Lee, Jehee, and Sung Yong Shin.
"A hierarchical approach to interactive motion editing for human-like figures."
Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.

Use current configuration as q_0
Find v such that (q_i = q_0_i exp(v_i) for 0<i<n where n are the number of joints) reaches the constraints
i.e. optimize over exponential map representation of displacement based on constraints, i.e. sum of errors of constraints from v
Minimize also the displacement
"""

import numpy as np
from scipy.optimize import minimize
from utils import convert_exp_frame_to_quat_frame, add_quat_frames


def ik_objective(x, skeleton, reference, constraints, weights):
    d = convert_exp_frame_to_quat_frame(skeleton, x)
    q_frame = add_quat_frames(skeleton, reference, d)

    # reduce displacement to original frame
    #error = np.linalg.norm(x)
    error = np.dot(x.T, np.dot(weights, x))

    # reduce distance to constraints
    #print q_frame.shape
    for c in constraints:
        error += c.evaluate(skeleton, q_frame)

    return error


class IKConstraint(object):
    def __init__(self, frame_idx, joint_name, position):
        self.frame_idx = frame_idx
        self.joint_name = joint_name
        self.position = position

    def evaluate(self, skeleton, q_frame):
        d = self.position - skeleton.nodes[self.joint_name].get_global_position(q_frame)
        return np.dot(d, d)


class IKConstraintSet(object):
    def __init__(self, frame_range, joint_names, positions):
        self.frame_range = frame_range
        self.joint_names = joint_names
        self.constraints = []
        for idx in xrange(frame_range[0], frame_range[1]):
            for idx, joint_name in enumerate(joint_names):
                c = IKConstraint(idx, joint_name, positions[idx])
                self.constraints.append(c)

    def add_constraint(self, c):
        self.constraints.append(c)

    def evaluate(self, skeleton, q_frame):
        error = 0
        for c in self.constraints:
            d = c.position - skeleton.nodes[c.joint_name].get_global_position(q_frame)
            error += np.dot(d, d)
        return error


class NumericalInverseKinematicsExp(object):
    def __init__(self, skeleton, ik_settings, verbose=False, objective=None):
        self.skeleton = skeleton
        self.n_joints = len(self.skeleton.animated_joints)
        if objective is None:
            self._objective = ik_objective
        else:
            self._objective = objective
        self.ik_settings = ik_settings
        self.verbose = verbose
        self._optimization_options = {'maxiter': self.ik_settings["max_iterations"], 'disp': self.verbose}
        self.joint_weights = np.eye(self.n_joints * 3)

    def set_joint_weights(self, weights):
        self.joint_weights = np.dot(self.joint_weights, weights)

    def modify_frame(self, reference, constraints):
        guess = np.zeros(self.n_joints * 3)
        r = minimize(self._objective, guess, args=(self.skeleton, reference, constraints, self.joint_weights),
                             method=self.ik_settings["optimization_method"],
                             tol=self.ik_settings["tolerance"],
                             options=self._optimization_options)
        exp_frame = r["x"]
        d = convert_exp_frame_to_quat_frame(self.skeleton, exp_frame)
        q_frame = add_quat_frames(self.skeleton, reference, d)
        return q_frame
