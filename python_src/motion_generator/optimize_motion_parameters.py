# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

Wrapper around scipy minimize and error function definition.

@author: erhe01,hadu01
"""

import time
import numpy as np
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


class NumericalMinimizer(object):
    """ A wrapper class for Scipy minimize that implements different Gradient Descent and
        derivative free optimization methods.
        Official documentation of that module:
        http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.optimize.minimize.html
    """
    def __init__(self, algorithm_config, skeleton, start_pose=None):        
        
        self._aglortihm_config = algorithm_config
        self.optimization_settings = algorithm_config["optimization_settings"]
        self.verbose = algorithm_config["verbose"]
        self._skeleton = skeleton
        self._start_pose = start_pose
        self._objective_function = None
        self._error_func_params = None
        return
        
    def set_objective_function(self, obj):
        self.objective_function = obj
        return
        
    def _set_error_func_parameters(self, motion_primitive_node, motion_primitve_constraints, prev_frames):
        self._error_func_params = motion_primitive_node.motion_primitive, motion_primitve_constraints, \
                                   prev_frames, self.optimization_settings["error_scale_factor"], \
                                   self.optimization_settings["quality_scale_factor"]

    
    def run(self, motion_primitive_node, gmm, constraints, initial_guess=None, prev_frames=None):
        """ Runs the optimization for a single motion primitive and a list of constraints 
        Parameters
        ----------
        * motion_primitive : MotionPrimitiveNode
            node in the MotionPrimitiveGraph used for back projection
        * gmm : sklearn.mixture.GMM
            Statistical model of the natural motion parameter space
        * constraints : list of dicts
             Each entry containts "joint", "position", "orientation" and "semanticAnnotation"
        * initial_guess : np.ndarray
            Optional initial guess.
        * prev_frames : np.ndarray
            Optional frames of the previous motion used for alignment.

        Returns
        -------
        * x : np.ndarray
              optimal low dimensional motion parameter vector
        """
    
        if initial_guess is None:
            s0 = np.ravel(gmm.sample())
        else:
            s0 = initial_guess
    
        self._set_error_func_parameters(motion_primitive_node, constraints, prev_frames)
              
        if self.verbose: 
            start = time.clock()
            print "Start optimization using", self.optimization_settings["method"], self.optimization_settings["max_iterations"]
    #    jac = error_function_jac(s0, data)
        try:
            result = minimize(self._objective_function,
                              s0,
                              args = (self._error_func_params,),
                              method=self.optimization_settings["method"], 
                              #jac = error_function_jac, 
                              tol = self.optimization_settings["tolerance"],
                              options={'maxiter': self.optimization_settings["max_iterations"], 'disp' : self.verbose})
                             
                             
      
        except ValueError as e:
            print "Warning:", e.message
            return s0
            
        if self.verbose:
            print "Finished optimization in ",time.clock()-start,"seconds"
        return result.x    


