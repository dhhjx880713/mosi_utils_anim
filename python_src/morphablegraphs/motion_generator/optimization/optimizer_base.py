# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

Wrapper around scipy minimize and error function definition.

@author: erhe01,hadu01
"""


class OptimizerBase(object):
    """ Defines interface for optimization algorithms
    """
    def __init__(self, optimization_settings):
        self.optimization_settings = optimization_settings
        self.verbose = optimization_settings["verbose"]
        self._objective_function = None
        self._error_func_params = None
        self._jacobian = None

    def set_objective_function(self, obj):
        self._objective_function = obj

    def set_objective_function_parameters(self, data):
        self._error_func_params = data

    def set_jacobian(self, jac):
        self._jacobian = jac

    def run(self, initial_guess):
        """ Runs the optimization for a single motion primitive and a list of constraints
        Returns
        -------
        * x : np.ndarray
              optimal low dimensional motion parameter vector
        """
        pass
