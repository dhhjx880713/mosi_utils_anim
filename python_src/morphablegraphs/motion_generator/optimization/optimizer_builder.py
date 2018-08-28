__author__ = 'erhe01'
from .least_squares import LeastSquares
from .numerical_minimizer import NumericalMinimizer
from .objective_functions import obj_spatial_error_sum,\
                                obj_spatial_error_residual_vector,\
                                obj_spatial_error_residual_vector_and_naturalness,\
                                obj_spatial_error_sum_and_naturalness, \
                                obj_time_error_sum, \
                                obj_global_error_sum, \
                                obj_global_residual_vector_and_naturalness, \
                                step_goal_error, step_goal_jac, \
                                step_goal_and_naturalness, step_goal_and_naturalness_jac


class OptimizerBuilder(object):
    def __init__(self, algorithm_settings):
        self.algorithm_settings = algorithm_settings

    def build_spatial_and_naturalness_error_minimizer(self):
        method = self.algorithm_settings["local_optimization_settings"]["method"]
        if method == "leastsq":
            minimizer = LeastSquares(self.algorithm_settings["local_optimization_settings"])
            minimizer.set_objective_function(obj_spatial_error_residual_vector_and_naturalness)#obj_spatial_error_residual_vector
        else:
            minimizer = NumericalMinimizer(self.algorithm_settings["local_optimization_settings"])
            minimizer.set_objective_function(obj_spatial_error_sum_and_naturalness)
        return minimizer

    def build_path_following_minimizer(self):
        minimizer = NumericalMinimizer(self.algorithm_settings["local_optimization_settings"])
        minimizer.set_objective_function(step_goal_error)
        minimizer.set_jacobian(step_goal_jac)
        return minimizer

    def build_path_following_with_likelihood_minimizer(self):
        minimizer = NumericalMinimizer(self.algorithm_settings["local_optimization_settings"])
        minimizer.set_objective_function(step_goal_and_naturalness)
        minimizer.set_jacobian(step_goal_and_naturalness_jac)
        return minimizer

    def build_spatial_error_minimizer(self):
        method = self.algorithm_settings["local_optimization_settings"]["method"]
        if method == "leastsq":
            minimizer = LeastSquares(self.algorithm_settings["local_optimization_settings"])
            minimizer.set_objective_function(obj_spatial_error_residual_vector)#obj_spatial_error_residual_vector
        else:
            minimizer = NumericalMinimizer(self.algorithm_settings["local_optimization_settings"])
            minimizer.set_objective_function(obj_spatial_error_sum)
        return minimizer

    def build_time_error_minimizer(self):
        minimizer = NumericalMinimizer(self.algorithm_settings["global_time_optimization_settings"])
        minimizer.set_objective_function(obj_time_error_sum)
        return minimizer

    def build_global_error_minimizer(self):
        minimizer = NumericalMinimizer(self.algorithm_settings["global_spatial_optimization_settings"])
        minimizer.set_objective_function(obj_global_error_sum)
        return minimizer

    def build_global_error_minimizer_residual(self):
        minimizer = LeastSquares(self.algorithm_settings["global_spatial_optimization_settings"])
        minimizer.set_objective_function(obj_global_residual_vector_and_naturalness)
        return minimizer
