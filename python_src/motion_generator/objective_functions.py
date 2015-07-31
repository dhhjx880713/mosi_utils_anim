# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:21:08 2015

@author: erhe01
"""

import numpy as np
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from scipy.optimize.optimize import approx_fprime
from . import global_counter_dict
                      

def obj_spatial_error_sum(s,data):
    """ Note: Time parameters and constraints will be ignored. 
    """
    s = np.asarray(s)
    motion_primitive, motion_primitive_constraint, prev_frames = data

    error_sum = motion_primitive_constraint.evaluate(motion_primitive, s, prev_frames,use_time_parameters=False)
    global_counter_dict["evaluations"] += 1
    return error_sum
    
def obj_time_error_sum(s,data):
    """ TODO
    """
    error_sum = 0
    return error_sum
    
def obj_spatial_error_sum_and_naturalness(s,data):
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
    kinematic_error = obj_spatial_error_sum(s,data[:-2])# ignore the kinematic factor and quality factor
    #print "s-vector",optimize_theta
    error_scale_factor = data[-1]
    quality_scale_factor = data[-2]
    n_log_likelihood = -data[0].gmm.score([s,])[0]
    print "naturalness is: " + str(n_log_likelihood)
    error = error_scale_factor * kinematic_error + n_log_likelihood * quality_scale_factor
    print "error",error
    return error
    
def obj_spatial_error_sum_and_naturalness_jac(x0, data):
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
        numerator += np.exp(logLikelihoods[i]) * gmm.weights_[i] * np.dot(np.linalg.inv(gmm.covars_[i]), (x0 - gmm.means_[i]))
#    numerator = numerator
    denominator = np.exp(gmm.score([x0])[0])
#    denominator = motion_primitive.gmm.score([x0])[0]
    logLikelihood_jac = numerator / denominator
    kinematic_jac = approx_fprime(x0, obj_spatial_error_sum, 1e-7, data[-2:])# ignore the kinematic factor and quality factor
    jac = logLikelihood_jac * quality_scale_factor + kinematic_jac * error_scale_factor
    return jac

        
