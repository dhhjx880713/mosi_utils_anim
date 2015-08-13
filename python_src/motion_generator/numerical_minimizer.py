# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

Wrapper around scipy minimize and error function definition.

@author: erhe01,hadu01
"""

import time
from scipy.optimize import minimize


class NumericalMinimizer(object):
    """ A wrapper class for Scipy minimize module that implements different gradient descent and
        derivative free optimization methods.
        Please see the official documentation of that module for the supported optimization methods:
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
        self._objective_function = obj
        return
        
    def set_objective_function_parameters(self,data):
        self._error_func_params = data

    
    def run(self, initial_guess):
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
        if self._objective_function is not None and initial_guess is not None:
            if self.verbose: 
                start = time.clock()
                print "Start optimization using", self.optimization_settings["method"], self.optimization_settings["max_iterations"]
        #    jac = error_function_jac(s0, data)
            try:
                result = minimize(self._objective_function,
                                  initial_guess,
                                  args = (self._error_func_params,),
                                  method=self.optimization_settings["method"], 
                                  #jac = error_function_jac, 
                                  tol = self.optimization_settings["tolerance"],
                                  options={'maxiter': self.optimization_settings["max_iterations"], 'disp' : self.verbose})
                                 
                                 
          
            except ValueError as e:
                print "Warning:", e.message
                return initial_guess
                
            if self.verbose:
                print "Finished optimization in ",time.clock()-start,"seconds"
            return result.x
        else:
            print "Error: No objective function set. Return initial guess instead."
            return initial_guess





