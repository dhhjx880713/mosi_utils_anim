""" Numerical IK using exponential displacement map  based on Section 5.1 of [1]

Use current configuration as q_0
Find v such that (q_i = q_0_i exp(v_i) for 0<i<n where n are the number of joints) reaches the constraints
i.e. optimize over exponential map representation of displacement based on constraints, i.e. sum of errors of constraints from v
Minimize also the displacement

[1] Lee, Jehee, and Sung Yong Shin. "A hierarchical approach to interactive motion editing for human-like figures."
Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.

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
