# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:21:08 2015

@author: hadu01, erhe01
"""

import numpy as np
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from scipy.optimize.optimize import approx_fprime
from . import global_counter_dict
                      

def obj_spatial_error_sum(s, data):
    """ Calculates the error of a low dimensional motion vector s 
    given constraints.
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
    error_sum = motion_primitive_constraints.evaluate(motion_primitive, s, prev_frames, use_time_parameters=False)
    global_counter_dict["evaluations"] += 1
    return error_sum
    

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
    
    kinematic_error = obj_spatial_error_sum(s,data[:-2])# ignore the kinematic factor and quality factor
    #print "s-vector",optimize_theta
    error_scale_factor = data[-1]
    quality_scale_factor = data[-2]
    n_log_likelihood = -data[0].gmm.score([s,])[0]
    print "naturalness is: " + str(n_log_likelihood)
    error = error_scale_factor * kinematic_error + n_log_likelihood * quality_scale_factor
    print "error",error
    return error


def obj_spatial_error_sum_and_naturalness_jac(s, data):
    """ jacobian of error function. It is a combination of analytic solution 
        for motion primitive model and numerical solution for kinematic error
    """
    #  Extract relevant parameters from data tuple. 
    #  Note other parameters are used for calling obj_error_sum
    gmm = data[0].gmm
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
#    denominator = motion_primitive.gmm.score([x0])[0]
    logLikelihood_jac = numerator / denominator
    kinematic_jac = approx_fprime(s, obj_spatial_error_sum, 1e-7, data[-2:])# ignore the kinematic factor and quality factor
    jac = logLikelihood_jac * quality_scale_factor + kinematic_jac * error_scale_factor
    return jac


def obj_time_error_sum(s, data):
    """ Calculates the error for time constraints for certain keyframes
    Parameters
    ---------
    * s : np.ndarray
        concatenatgion of low dimensional motion representations
    * data : tuple
        Contains morhable_graph, time_constraints, motion, n_steps
        
    Returns
    -------
    * error: float
    """
    s = np.asarray(s)
    error_sum = 0
    morhable_graph, motion, time_constraints, n_steps, start_keyframe = data

    #get time functions for all steps
    time_functions = []
    offset = 0
    for step in motion.graph_walk[n_steps:]: #  look back n_steps
        gamma = s[offset:step.n_time_components]
        time_function = morhable_graph.nodes[step.node_key]._inverse_temporal_pca(gamma)
        time_functions.append(time_function)
        offset += step.n_time_components

    #get difference to desired time for each constraint
    for constrained_step_index, constrained_keyframe_index, desired_time in time_constraints:
        n_frames = start_keyframe #w hen it starts the first step start_keyframe would be 0
        temp_step_index = 0
        for step in motion.graph_walk[n_steps:]: # look back n_steps
            if temp_step_index < constrained_step_index:
                #simply add the number of frames
                n_frames += len(time_functions[temp_step_index])
                temp_step_index += 1
            else:
                #inverse lookup the warped frame that maps to the labelled canonical keyframe with the time constraint
                mapped_key_frame = min(time_function[temp_step_index], key=lambda x: abs(x-constrained_keyframe_index))
                n_frames += mapped_key_frame
                total_seconds = n_frames * morhable_graph.skeleton.frame_time
                error_sum += abs(desired_time-total_seconds)
                break
    return error_sum