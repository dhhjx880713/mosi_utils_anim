# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

Wrapper around scipy minimize and error function definition.

@author: erhe01,hadu01
"""

import time
import numpy as np
from constraint.constraint_check import obj_error_sum
from scipy.optimize import minimize
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from scipy.optimize.optimize import approx_fprime


def error_function_jac(x0, data):
    """ jacobian of error function. It is a combination of analytic solution 
        for motion primitive model and numerical solution for kinematic error
    """
    #  Extract relevant parameters from data tuple.
    #  Note other parameters are used for calling obj_error_sum
    gmm = data[0].gmm
    error_scale_factor = data[-1]
    quality_scale_factor = data[-2]

    tmp = np.reshape(x0, (1, len(x0)))
    logLikelihoods = _log_multivariate_normal_density_full(tmp,
                                                           gmm.means_,
                                                           gmm.covars_)
    logLikelihoods = np.ravel(logLikelihoods)

    numerator = 0

    n_models = len(gmm.weights_)
    for i in xrange(n_models):
        numerator += np.exp(logLikelihoods[i]) * gmm.weights_[i] * np.dot(
            np.linalg.inv(gmm.covars_[i]), (x0 - gmm.means_[i]))
#    numerator = numerator
    denominator = np.exp(gmm.score([x0])[0])
#    denominator = motion_primitive.gmm.score([x0])[0]
    logLikelihood_jac = numerator / denominator
    # ignore the kinematic factor and quality factor
    kinematic_jac = approx_fprime(x0, obj_error_sum, 1e-7, data[-2:])
    jac = logLikelihood_jac * quality_scale_factor + \
        kinematic_jac * error_scale_factor
    return jac


def error_func(s, data):
    """ Calculates the error of a low dimensional motion vector s given 
        constraints and the prior knowledge from the statistical model

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains parameters of get_optimal_parameters_from_optimization

    Returns
    -------
    the error

    """

#    motion_primitive, gmm,  constraints, quality_scale_factor, \
#    error_scale_factor, bvh_reader,prev_frames, node_name_map, bounding_boxes,\
#    start_transformation,epsilon = data  # the pre_frames are quaternion frames
    #kinematic_error = kinematic_error_func(s,data)
    # ignore the kinematic factor and quality factor
    kinematic_error = obj_error_sum(s, data[:-2])
    # print "s-vector",optimize_theta
    error_scale_factor = data[-1]
    quality_scale_factor = data[-2]
    n_log_likelihood = -data[0].gmm.score([s, ])[0]
    print "naturalness is: " + str(n_log_likelihood)
    error = error_scale_factor * kinematic_error + \
        n_log_likelihood * quality_scale_factor
    print "error", error
    return error


def run_optimization(motion_primitive, gmm, constraints, initial_guess, skeleton,
                     optimization_settings={}, bounding_boxes=None, prev_frames=None,
                     start_pose=None, verbose=False):
    """ Runs the optimization for a single motion primitive and a list of constraints 
    Parameters
    ----------
    * motion_primitive : MotionPrimitive
        Instance of a motion primitive used for back projection
    * gmm : sklearn.mixture.GMM
        Statistical model of the natural motion parameter space
    * constraints : list of dicts
         Each entry containts "joint", "position"   "orientation" and "semanticAnnotation"
    * skeleton: Skeleton
        Used for joint hiearchy information 
    * max_iterations : int
        Maximum number of iterations performed by the optimization algorithm
    * bounding_boxes : tuple
        Bounding box data read from a graph node in the morphable graph
    * prev_frames : np.ndarray
        Motion parameters of the previous motion used for alignment
    * start_pose : dict
        Contains keys position and orientation. "position" contains Cartesian coordinates 
        and orientation contains Euler angles in degrees)

    Returns
    -------
    * x : np.ndarray
          optimal low dimensional motion parameter vector
    """

    if initial_guess is None:
        s0 = np.ravel(gmm.sample())
    else:
        s0 = initial_guess

    data = motion_primitive, constraints, prev_frames, start_pose, skeleton, {"pos": 1, "rot": 1, "smooth": 1}, \
        optimization_settings["error_scale_factor"], optimization_settings[
            "quality_scale_factor"]  # precision

    options = {'maxiter': optimization_settings[
        "max_iterations"], 'disp': verbose}

    if verbose:
        start = time.clock()
        print "Start optimization using", optimization_settings["method"], optimization_settings["max_iterations"]
#    jac = error_function_jac(s0, data)
    try:
        result = minimize(error_func,
                          s0,
                          args=(data,),
                          method=optimization_settings["method"],
                          #jac = error_function_jac,
                          tol=optimization_settings["tolerance"],
                          options=options)

    except ValueError as e:
        print "Warning:", e.message
        return s0

    if verbose:
        print "Finished optimization in ", time.clock() - start, "seconds"
    return result.x
