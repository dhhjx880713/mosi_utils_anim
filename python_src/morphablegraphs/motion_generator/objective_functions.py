# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:21:08 2015

@author: hadu01, erhe01
"""

import numpy as np
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from scipy.optimize.optimize import approx_fprime
from ..animation_data.motion_editing import align_quaternion_frames

def obj_spatial_error_sum(s, data):
    """ Calculates the error of a low dimensional motion vector s 
    given a list of constraints.
    Note: Time parameters and time constraints will be ignored. 

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains  motion_primitive, motion_primitive_constraints, prev_frames
        
    Returns
    -------
    * error: float
    """
    s = np.asarray(s)
    motion_primitive, motion_primitive_constraints, prev_frames = data
    motion_primitive_constraints.min_error = motion_primitive_constraints.evaluate(motion_primitive, s, prev_frames,
                                                                                   use_time_parameters=False)
    return motion_primitive_constraints.min_error


def obj_spatial_error_sum_and_naturalness(s, data):
    """ Calculates the error of a low dimensional motion vector s given
        constraints and the prior knowledge from the statistical model

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains motion_primitive, motion_primitive_constraints, prev_frames, error_scale_factor, quality_scale_factor

    Returns
    -------
    * error: float

    """

    spatial_error = obj_spatial_error_sum(s, data[:-2])# ignore the kinematic factor and quality factor
    #print "s-vector",optimize_theta
    error_scale_factor = data[-2]
    quality_scale_factor = data[-1]
    n_log_likelihood = -data[0].gaussian_mixture_model.score([s, ])[0]
    print "naturalness is: " + str(n_log_likelihood)
    error = error_scale_factor * spatial_error + n_log_likelihood * quality_scale_factor
    print "error", error
    return error


def obj_spatial_error_sum_and_naturalness_jac(s, data):
    """ jacobian of error function. It is a combination of analytic solution 
        for motion primitive model and numerical solution for kinematic error
    """
    #  Extract relevant parameters from data tuple. 
    #  Note other parameters are used for calling obj_error_sum
    gmm = data[0].gaussian_mixture_model
    error_scale_factor = data[-1]
    quality_scale_factor = data[-2]
    
    tmp = np.reshape(s, (1, len(s)))
    logLikelihoods = _log_multivariate_normal_density_full(tmp,
                                                           gmm.means_, 
                                                           gmm.covars_)
    logLikelihoods = np.ravel(logLikelihoods)

    numerator = 0

    n_models = len(gmm.weights_)
    for i in xrange(n_models):
        numerator += np.exp(logLikelihoods[i]) * gmm.weights_[i] * np.dot(np.linalg.inv(gmm.covars_[i]), (s - gmm.means_[i]))
#    numerator = numerator
    denominator = np.exp(gmm.score([s])[0])
#    denominator = motion_primitive.gaussian_mixture_model.score([x0])[0]
    logLikelihood_jac = numerator / denominator
    kinematic_jac = approx_fprime(s, obj_spatial_error_sum, 1e-7, data[-2:])# ignore the kinematic factor and quality factor
    jac = logLikelihood_jac * quality_scale_factor + kinematic_jac * error_scale_factor
    return jac


def obj_spatial_error_residual_vector(s, data):
    """ Calculates the error of a low dimensional motion vector s
    given a list of constraints and stores the error of each constraint in a list.
    Note: Time parameters and time constraints will be ignored.

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains  motion_primitive, motion_primitive_constraints, prev_frames

    Returns
    -------
    * residual_vector: list
    """
    s = np.asarray(s)
    motion_primitive, motion_primitive_constraints, prev_frames, error_scale_factor, quality_scale_factor = data
    residual_vector = motion_primitive_constraints.get_residual_vector(motion_primitive, s, prev_frames, use_time_parameters=False)
    motion_primitive_constraints.min_error = sum(residual_vector)
    #print len(residual_vector), residual_vector
    #print "error", motion_primitive_constraints.min_error
    n_variables = len(s)
    while len(residual_vector) < n_variables:
        residual_vector.append(0)
    return residual_vector


def obj_spatial_error_residual_vector_and_naturalness(s, data):
    """ Calculates the error of a low dimensional motion vector s
    given a list of constraints and stores the error of each constraint in a list.
    Note: Time parameters and time constraints will be ignored.

    Parameters
    ---------
    * s : np.ndarray
        low dimensional motion representation
    * data : tuple
        Contains  motion_primitive, motion_primitive_constraints, prev_frames

    Returns
    -------
    * residual_vector: list
    """
    s = np.asarray(s)
    motion_primitive, motion_primitive_constraints, prev_frames, error_scale_factor, quality_scale_factor = data
    n_log_likelihood = -data[0].gaussian_mixture_model.score([s, ])[0] * quality_scale_factor
    residual_vector = motion_primitive_constraints.get_residual_vector(motion_primitive, s, prev_frames, use_time_parameters=False)
    motion_primitive_constraints.min_error = sum(residual_vector)
    for i in xrange(len(residual_vector)):
        residual_vector[i] *= error_scale_factor
        residual_vector[i] += n_log_likelihood
    #print len(residual_vector), residual_vector
    #print "error", motion_primitive_constraints.min_error # sum(residual_vector)
    n_variables = len(s)
    while len(residual_vector) < n_variables:
        residual_vector.append(0)
    return residual_vector


def obj_time_error_sum(s, data):
    """ Calculates the error for time constraints for certain keyframes
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains morhable_graph, time_constraints, motion, start_step

    Returns
    -------
    * error: float
    """
    s = np.asarray(s)
    morphable_graph, graph_walk, time_constraints, error_scale_factor, quality_scale_factor = data
    time_error = time_constraints.evaluate_graph_walk(s, morphable_graph, graph_walk)
    n_log_likelihood = -time_constraints.get_average_loglikelihood(s, morphable_graph, graph_walk)
    error = error_scale_factor * time_error + n_log_likelihood * quality_scale_factor
    print "time error", error, time_error, n_log_likelihood
    return error


def obj_global_error_sum(s, data):
    """ Calculates the error for spatial constraints for certain keyframes
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains morhable_graph, time_constraints, motion, start_step

    Returns
    -------
    * error: float
    """
    s = np.asarray(s)
    offset = 0
    error = 0
    motion_primitive_graph, graph_walk_steps, error_scale_factor, quality_scale_factor, prev_frames = data
    for step in graph_walk_steps:
        alpha = s[offset:offset+step.n_spatial_components]
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).get_motion_vector()
        #print "got sample frames"
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints, \
                       prev_frames#, error_scale_factor, quality_scale_factor
        prev_frames = align_quaternion_frames(sample_frames, prev_frames, step.motion_primitive_constraints.start_pose)
        error += obj_spatial_error_sum(alpha, step_data)#_and_naturalness
        offset += step.n_spatial_components
    print "global error", error
    return error


def obj_global_residual_vector(s, data):
    """ Calculates the error for spatial constraints for certain keyframes
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains morhable_graph, time_constraints, motion, start_step

    Returns
    -------
    * residual_vector: list
    """
    s = np.asarray(s)
    offset = 0
    residual_vector = []
    motion_primitive_graph, graph_walk_steps, error_scale_factor, quality_scale_factor, prev_frames = data
    for step in graph_walk_steps:
        alpha = s[offset:offset+step.n_spatial_components]
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).get_motion_vector()
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints, \
                       prev_frames, error_scale_factor, quality_scale_factor
        prev_frames = align_quaternion_frames(sample_frames, prev_frames, step.motion_primitive_constraints.start_pose)
        residual_vector += obj_spatial_error_residual_vector(alpha, step_data)
        offset += step.n_spatial_components
    print "global error", sum(residual_vector), residual_vector
    return residual_vector


def obj_global_residual_vector_and_naturalness(s, data):
    """ Calculates the error for spatial constraints for certain keyframes and
        adds negative log likelihood from the statistical model
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains morhable_graph, time_constraints, motion, start_step

    Returns
    -------
    * residual_vector: list
    """
    s = np.asarray(s)
    offset = 0
    residual_vector = []
    motion_primitive_graph, graph_walk_steps, error_scale_factor, quality_scale_factor, prev_frames = data
    for step in graph_walk_steps:
        alpha = s[offset:offset+step.n_spatial_components]
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).get_motion_vector()
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints, \
                       prev_frames, error_scale_factor, quality_scale_factor
        prev_frames = align_quaternion_frames(sample_frames, prev_frames, step.motion_primitive_constraints.start_pose)
        residual_vector += obj_spatial_error_residual_vector_and_naturalness(alpha.tolist()+step.parameters[step.n_spatial_components:].tolist(), step_data)
        offset += step.n_spatial_components
    print "global error", sum(residual_vector), residual_vector
    return residual_vector