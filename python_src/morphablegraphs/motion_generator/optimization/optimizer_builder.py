__author__ = 'erhe01'
from copy import copy
from numerical_minimizer import NumericalMinimizer
from least_squares import LeastSquares
from ..objective_functions import obj_spatial_error_residual_vector,\
                                  obj_spatial_error_residual_vector_and_naturalness,\
                                  obj_spatial_error_sum_and_naturalness,\
                                  obj_time_error_sum


class OptimizerBuilder(object):
    def __init__(self, algorithm_settings):
        self.algorithm_settings = algorithm_settings

    def build_spatial_error_minimizer(self):
        method = self.algorithm_settings["optimization_settings"]["method"]
        if method == "leastsq":
            minimizer = LeastSquares(self.algorithm_settings)
            minimizer.set_objective_function(obj_spatial_error_residual_vector_and_naturalness)#obj_spatial_error_residual_vector
        else:
            minimizer = NumericalMinimizer(self.algorithm_settings)
            minimizer.set_objective_function(obj_spatial_error_sum_and_naturalness)
        return minimizer

    def build_time_error_minimizer(self):
        algorithm_settings = copy(self.algorithm_settings)
        algorithm_settings["optimization_settings"]["method"] = "BFGS"
        minimizer = NumericalMinimizer(algorithm_settings)
        minimizer.set_objective_function(obj_time_error_sum)
        return minimizer
