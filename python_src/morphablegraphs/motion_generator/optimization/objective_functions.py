# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:21:08 2015

@author: hadu01, erhe01
"""

import numpy as np
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from scipy.optimize.optimize import approx_fprime
from ...animation_data.utils import align_quaternion_frames_only_last_frame
from ...animation_data.motion_concatenation import align_quaternion_frames
from ...constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET


def obj_frame_error(s, data):
    motion_primitive, target, joint_name, frame_idx = data
    sample = motion_primitive.back_project(s, use_time_parameters=False)# TODO backproject only current frame
    frame = sample.get_motion_vector()[frame_idx]
    p = motion_primitive.skeleton.nodes[joint_name].get_global_position(frame)
    return np.linalg.norm(target-p)


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
    motion_primitive, mp_constraints, prev_frames = data
    mp_constraints.min_error = mp_constraints.evaluate(motion_primitive, s, prev_frames, use_time_parameters=False)
    return mp_constraints.min_error


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

    spatial_error = obj_spatial_error_sum(s, data[:-3])# ignore the kinematic factor and quality factor
    error_scale_factor = data[-3]
    quality_scale_factor = data[-2]
    init_error_sum = data[-1]
    n_log_likelihood = -data[0].get_gaussian_mixture_model().score(s.reshape((1, len(s))))
    #print "naturalness is: " + str(n_log_likelihood)
    error = error_scale_factor * spatial_error + n_log_likelihood * quality_scale_factor
    #print "error", error
    return error/init_error_sum


def obj_spatial_error_sum_and_naturalness_jac(s, data):
    """ jacobian of error function. It is a combination of analytic solution 
        for motion primitive model and numerical solution for kinematic error
    """
    #  Extract relevant parameters from data tuple. 
    #  Note other parameters are used for calling obj_error_sum
    gmm = data[0].get_gaussian_mixture_model()
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
#    denominator = motion_primitive.get_gaussian_mixture_model().score(x0)
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
    motion_primitive, motion_primitive_constraints, prev_frames, error_scale_factor, quality_scale_factor, init_error_sum = data
    residual_vector = motion_primitive_constraints.get_residual_vector(motion_primitive, s, prev_frames, use_time_parameters=False)
    motion_primitive_constraints.min_error = np.sum(residual_vector)
    #print len(residual_vector), residual_vector
    #print "error", motion_primitive_constraints.min_error
    n_variables = s.shape[0]
    n_error_values = len(residual_vector)
    while n_error_values < n_variables:
        residual_vector.append(0)
        n_error_values += 1
    return np.array(residual_vector)/init_error_sum


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
    mp, mp_constraints, prev_frames, error_scale_factor, quality_scale_factor, init_error_sum = data
    negative_log_likelihood = float(-data[0].get_gaussian_mixture_model().score(s.reshape((1, len(s)))) * quality_scale_factor)
    residual_vector = mp_constraints.get_residual_vector(mp, s, prev_frames, use_time_parameters=False)
    mp_constraints.min_error = np.sum(residual_vector)
    n_error_values = len(residual_vector)
    for i in xrange(n_error_values):
        residual_vector[i] *= error_scale_factor
        residual_vector[i] += negative_log_likelihood
    #print len(residual_vector), residual_vector
    #print "error", motion_primitive_constraints.min_error # sum(residual_vector)
    n_variables = s.shape[0]
    while n_error_values < n_variables:
        residual_vector.append(0)
        n_error_values += 1
    return np.array(residual_vector) / init_error_sum


def obj_time_error_sum(s, data):
    """ Calculates the error for time constraints for certain keyframes
    Parameters
    ---------
    * s : np.ndarray
        concatenation of low dimensional motion representations
    * data : tuple
        Contains motion_primitive_graph, graph_walk, time_constraints, motion, start_step

    Returns
    -------
    * error: float
    """
    #s = np.asarray(s)
    motion_primitive_graph, graph_walk, time_constraints, error_scale_factor, quality_scale_factor = data
    time_error = time_constraints.evaluate_graph_walk(s, motion_primitive_graph, graph_walk)
    n_log_likelihood = -time_constraints.get_average_loglikelihood(s, motion_primitive_graph, graph_walk)
    error = error_scale_factor * time_error + n_log_likelihood * quality_scale_factor
    #print "time error", error, time_error, n_log_likelihood
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
    #s = np.asarray(s)
    offset = 0
    error = 0
    motion_primitive_graph, graph_walk_steps, error_scale_factor, quality_scale_factor, prev_frames = data
    skeleton = motion_primitive_graph.skeleton
    node_name = skeleton.aligning_root_node
    for step in graph_walk_steps:
        alpha = s[offset:offset+step.n_spatial_components]
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).get_motion_vector()
        #print "got sample frames"
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints, \
                       prev_frames#, error_scale_factor, quality_scale_factor
        prev_frames = align_quaternion_frames(skeleton, node_name, sample_frames, prev_frames,
                                              step.motion_primitive_constraints.start_pose, 0)
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
    #s = np.asarray(s)
    offset = 0
    residual_vector = []
    motion_primitive_graph, graph_walk_steps, error_scale_factor, quality_scale_factor, prev_frames, init_error_sum = data
    skeleton = motion_primitive_graph.skeleton
    node_name = skeleton.aligning_root_node
    for step in graph_walk_steps:
        alpha = s[offset:offset+step.n_spatial_components]
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).get_motion_vector()
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints, \
                       prev_frames, error_scale_factor, quality_scale_factor
        prev_frames = align_quaternion_frames(skeleton, node_name, sample_frames, prev_frames,
                                              step.motion_primitive_constraints.start_pose)
        residual_vector += obj_spatial_error_residual_vector(alpha, step_data)
        offset += step.n_spatial_components
    #print "global error", np.sum(residual_vector), residual_vector
    return np.array(residual_vector)/init_error_sum


# def obj_global_residual_vector_and_naturalness(s, data):
#     """ Calculates the error for spatial constraints for certain keyframes and
#         adds negative log likelihood from the statistical model
#     Parameters
#     ---------
#     * s : np.ndarray
#         concatenation of low dimensional motion representations
#     * data : tuple
#         Contains morhable_graph, time_constraints, motion, start_step
#
#     Returns
#     -------
#     * residual_vector: list
#     """
#     #s = np.asarray(s)
#     offset = 0
#     residual_vector = []
#     motion_primitive_graph, graph_walk_steps, error_scale_factor, quality_scale_factor, prev_frames = data
#     for step in graph_walk_steps:
#         #for c in step.motion_primitive_constraints.constraints:
#         #    if c.constraint_type == SPATIAL_CONSTRAINT_TYPE_TRAJECTORY_SET:
#         #        c.joint_arc_lengths = np.zeros(len(c.joint_trajectories))
#         #        c.set_min_arc_length_from_previous_frames(prev_frames)
#         alpha = s[offset:offset+step.n_spatial_components]
#         #step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints,\
#         #               prev_frames, error_scale_factor, quality_scale_factor
#         concat_alpha = np.hstack((alpha, step.parameters[step.n_spatial_components:]))
#         #residual_vector += obj_spatial_error_residual_vector_and_naturalness(concat_alpha, step_data)
#         residual_vector += step.motion_primitive_constraints.get_residual_vector(motion_primitive_graph.nodes[step.node_key], concat_alpha, prev_frames, use_time_parameters=False)
#         prev_frames = align_quaternion_frames_only_last_frame(motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).get_motion_vector(), prev_frames, step.motion_primitive_constraints.start_pose)
#         offset += step.n_spatial_components
#     #print "global error", sum(residual_vector), residual_vector
#     m = len(s)
#     count = len(residual_vector)
#     while count < m:
#         residual_vector.append(0.0)
#         count += 1
#     return residual_vector




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
    offset = 0
    residual_vector = np.array([])
    motion_primitive_graph, graph_walk_steps, error_scale_factor, quality_scale_factor, prev_frames, init_error_sum = data
    skeleton = motion_primitive_graph.skeleton
    node_name = skeleton.aligning_root_node
    #print "root param range", root_param_range,motion_primitive_graph.skeleton, motion_primitive_graph.skeleton.root_orientation_node
    for step in graph_walk_steps:
        alpha = np.array(s[offset:offset+step.n_spatial_components])
        sample_frames = motion_primitive_graph.nodes[step.node_key].back_project(alpha, use_time_parameters=False).coeffs
        step_data = motion_primitive_graph.nodes[step.node_key], step.motion_primitive_constraints,\
                       prev_frames, error_scale_factor, quality_scale_factor, 1.0
        concat_alpha = np.hstack((alpha, step.parameters[step.n_spatial_components:]))
        residual_vector = np.hstack( (residual_vector,obj_spatial_error_residual_vector_and_naturalness(concat_alpha, step_data)))
        prev_frames = align_quaternion_frames(skeleton, node_name, sample_frames, prev_frames, step.motion_primitive_constraints.start_pose)
        offset += step.n_spatial_components
    #print "global error", sum(residual_vector)
    return residual_vector/init_error_sum#10000.0
