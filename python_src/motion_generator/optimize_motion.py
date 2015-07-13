# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

Wrapper around scipy minimize and error function definition.

@author: erhe01,hadu01
"""

import time
import numpy as np
from lib.constraint import obj_error_sum
from scipy.optimize import minimize
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from scipy.optimize.optimize import approx_fprime
                                 

def generate_optimization_settings(method="BFGS",max_iterations=100,quality_scale_factor=1,
                                   error_scale_factor=0.1,tolerance=0.05,optimize_theta=False,kinematic_epsilon=5):
    """ Generates optimization_settings dict that needs to be passed to the run_optimization mbvh_readerethod
    """
    
    settings = {"method":method, 
                 "max_iterations"  : max_iterations,
                "quality_scale_factor":quality_scale_factor,
                "error_scale_factor": error_scale_factor,
                "optimize_theta":optimize_theta,
                "tolerance":tolerance,
                "kinematic_epsilon":kinematic_epsilon}
    return settings

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
        numerator += np.exp(logLikelihoods[i]) * gmm.weights_[i] * np.dot(np.linalg.inv(gmm.covars_[i]), (x0 - gmm.means_[i]))
#    numerator = numerator
    denominator = np.exp(gmm.score([x0])[0])
#    denominator = motion_primitive.gmm.score([x0])[0]
    logLikelihood_jac = numerator / denominator
    kinematic_jac = approx_fprime(x0, obj_error_sum, 1e-7, data[-2:])# ignore the kinematic factor and quality factor
    jac = logLikelihood_jac * quality_scale_factor + kinematic_jac * error_scale_factor
    return jac

        

                
def error_func(s,data):
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
    kinematic_error = obj_error_sum(s,data[:-2])# ignore the kinematic factor and quality factor
    #print "s-vector",optimize_theta
    error_scale_factor = data[-1]
    quality_scale_factor = data[-2]
    n_log_likelihood = -data[0].gmm.score([s,])[0]
    print "naturalness is: " + str(n_log_likelihood)
    error = error_scale_factor * kinematic_error + n_log_likelihood * quality_scale_factor
    print "error",error
    return error
  



def run_optimization(motion_primitive,gmm,constraints,initial_guess, bvh_reader, node_name_map , 
                       optimization_settings = {},bounding_boxes=None,prev_frames = None,
                       start_pose=None,verbose=False):
    """ Runs the optimization for a single motion primitive and a list of constraints 
    Parameters
    ----------
    * motion_primitive : MotionPrimitive
        Instance of a motion primitive used for back projection
    * gmm : sklearn.mixture.GMM
        Statistical model of the natural motion parameter space
    * constraints : list of dicts
         Each entry containts "joint", "position"   "orientation" and "semanticAnnotation"
    * bvh_reader: BVHReader
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


#    optimize_theta = optimization_settings["optimize_theta"]    
#    kinematic_epsilon = optimization_settings["kinematic_epsilon"] #0.0# 0.25
#    
#    if start_pose is not None:
#        start_transformation = create_transformation(start_pose["orientation"],start_pose["position"])
#    else:
#        start_transformation = np.eye(4)
#    
    if initial_guess == None:
        s0 = np.ravel(gmm.sample())#sample initial guess
    else:
        s0 = initial_guess

    # convert prev_frames to euler frames
#    if prev_frames is not None:
#        prev_frames = convert_quaternion_to_euler(prev_frames)
    data = motion_primitive, constraints, prev_frames,start_pose, bvh_reader, node_name_map,{"pos":1,"rot":1,"smooth":1}, \
           optimization_settings["error_scale_factor"], optimization_settings["quality_scale_factor"]     #precision
#    data = motion_primitive, gmm, constraints, quality_scale_factor, \
#          error_scale_factor,  bvh_reader,prev_frames, node_name_map,bounding_boxes, \
#          start_transformation,kinematic_epsilon

    options = {'maxiter': optimization_settings["max_iterations"], 'disp' : verbose}
    
    if verbose: 
        start = time.clock()
        print "Start optimization using", optimization_settings["method"],optimization_settings["max_iterations"]
#    jac = error_function_jac(s0, data)

    result = minimize(error_func,#
                      s0,
                      args = (data,),
                      method=optimization_settings["method"], 
                      #jac = error_function_jac, 
                      tol = optimization_settings["tolerance"],
                      options=options)
    if verbose:
        print "Finished optimization in ",time.clock()-start,"seconds"
    return result.x


