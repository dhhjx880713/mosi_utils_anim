__author__ = 'erhe01'

from numerical_minimizer_scipy import NumericalMinimizerScipy
from numerical_minimizer_leastsq import NumericalMinimizerLeastSquares
from ..objective_functions import obj_spatial_error_residual_vector, obj_spatial_error_sum_and_naturalness, obj_time_error_sum


class NumericalMinimizerBuilder(object):
    def __init__(self, algorithm_settings):
        self.algorithm_settings = algorithm_settings

    def build_spatial_error_minimizer(self):
        method = self.algorithm_settings["optimization_settings"]["method"]
        if method == "leastsq":
            minimizer = NumericalMinimizerLeastSquares(self.algorithm_settings)
            minimizer.set_objective_function(obj_spatial_error_residual_vector)
        else:
            minimizer = NumericalMinimizerScipy(self.algorithm_settings)
            minimizer.set_objective_function(obj_spatial_error_sum_and_naturalness)
        return minimizer

    def build_time_error_minimizer(self):
        minimizer = NumericalMinimizerScipy(self.algorithm_settings)
        minimizer.set_objective_function(obj_time_error_sum)
        return minimizer
