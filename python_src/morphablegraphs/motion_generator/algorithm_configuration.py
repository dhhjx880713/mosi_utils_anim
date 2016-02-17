# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:19:46 2015

@author: erhe01
"""

import logging
from ..utilities.io_helper_functions import load_json_file
from ..animation_data.motion_editing import DEFAULT_SMOOTHING_WINDOW_SIZE


class AlgorithmConfigurationBuilder(object):
    """Generates a dict containing all settings for the algorithm
    """
    def __init__(self):
        self.use_constraints = True
        self.local_optimization_mode = "none"
        self.use_transition_model = False
        self.use_constrained_gmm = False
        self.activate_parameter_check = False
        self.use_global_time_optimization = False
        self.use_global_spatial_optimization = False
        self.n_random_samples = 100
        self.constrained_gmm_settings = dict()
        self.local_optimization_settings = dict()
        self.global_spatial_optimization_settings = dict()
        self.global_time_optimization_settings = dict()
        self.trajectory_following_settings = dict()
        self.smoothing_settings = dict()
        self.constrained_sampling_mode = "random"
        self.n_cluster_search_candidates = 2
        self.debug_max_step = -1
        self.verbose = False
        self.average_elementary_action_error_threshold = 500
        self.collision_avoidance_constraints_mode = "none"
        self.optimize_collision_avoidance_constraints_extra = False
        self.set_default_constrained_gmm_settings()
        self.set_default_trajectory_following_settings()
        self.set_default_optimization_settings()
        self.set_default_smoothing_settings()
        self.build()

    def set_default_constrained_gmm_settings(self):
        self.constrained_gmm_settings = dict()
        self.constrained_gmm_settings["precision"] = dict()
        self.constrained_gmm_settings["precision"]["pos"] = 1
        self.constrained_gmm_settings["precision"]["rot"] = 0.15
        self.constrained_gmm_settings["precision"]["smooth"] = 5
        self.constrained_gmm_settings["smooth_precision"] = 5
        self.constrained_gmm_settings["strict"] = False
        self.constrained_gmm_settings["max_bad_samples"] = 200

    def set_default_optimization_settings(self):
        self.local_optimization_settings = dict()
        self.local_optimization_settings["method"] = "leastsq"
        self.local_optimization_settings["start_error_threshold"] = 4.0
        self.local_optimization_settings["max_iterations"] = 150
        self.local_optimization_settings["quality_scale_factor"] = 0.001
        self.local_optimization_settings["error_scale_factor"] = 0.01
        self.local_optimization_settings["tolerance"] = 0.01
        self.local_optimization_settings["spatial_epsilon"] = 0.0
        self.global_spatial_optimization_settings = dict()
        self.global_spatial_optimization_settings["max_steps"] = 2
        self.global_spatial_optimization_settings["method"] = "leastsq"
        self.global_spatial_optimization_settings["start_error_threshold"] = 4.0
        self.global_spatial_optimization_settings["max_iterations"] = 150
        self.global_spatial_optimization_settings["quality_scale_factor"] = 0.001
        self.global_spatial_optimization_settings["error_scale_factor"] = 0.01
        self.global_spatial_optimization_settings["tolerance"] = 0.01
        self.global_spatial_optimization_settings["spatial_epsilon"] = 0.0
        self.global_time_optimization_settings = dict()
        self.global_time_optimization_settings["max_steps"] = 20
        self.global_time_optimization_settings["method"] = "BFGS"
        self.global_time_optimization_settings["max_iterations"] = 150
        self.global_time_optimization_settings["quality_scale_factor"] = 0.001
        self.global_time_optimization_settings["error_scale_factor"] = 0.01
        self.global_time_optimization_settings["tolerance"] = 0.01
        self.global_time_optimization_settings["spatial_epsilon"] = 0.0

    def set_default_trajectory_following_settings(self):
        self.trajectory_following_settings = dict()
        self.trajectory_following_settings["spline_type"] = 0
        self.trajectory_following_settings["control_point_filter_threshold"] = 50
        self.trajectory_following_settings["step_length_approx_method"] = "arc_length"
        self.trajectory_following_settings["heuristic_step_length_factor"] = 0.8
        self.trajectory_following_settings["dir_constraint_factor"] = 10.0
        self.trajectory_following_settings["position_constraint_factor"] = 1.0
        self.trajectory_following_settings["transition_pose_constraint_factor"] = 1.0
        self.trajectory_following_settings["closest_point_search_accuracy"] = 0.001
        self.trajectory_following_settings["closest_point_search_max_iterations"] = 5000
        self.trajectory_following_settings[" look_ahead_distance"] = 500

    def set_default_smoothing_settings(self):
        self.smoothing_settings = dict()
        self.smoothing_settings["spatial_smoothing"] = True
        self.smoothing_settings["time_smoothing"] = True
        self.smoothing_settings["spatial_smoothing_window"] = DEFAULT_SMOOTHING_WINDOW_SIZE
        self.smoothing_settings["time_smoothing_window"] = 15

    def from_json(self, filename):
        temp_algorithm_config = load_json_file(filename)
        self.use_constraints = temp_algorithm_config["use_constraints"]
        self.local_optimization_mode = temp_algorithm_config["local_optimization_mode"]
        self.use_transition_model = temp_algorithm_config["use_transition_model"]
        self.use_constrained_gmm = temp_algorithm_config["use_constrained_gmm"]
        self.use_global_time_optimization = temp_algorithm_config["use_global_time_optimization"]
        self.use_global_spatial_optimization = temp_algorithm_config["use_global_spatial_optimization"]
        self.activate_parameter_check = temp_algorithm_config["activate_parameter_check"]
        self.smoothing_settings = temp_algorithm_config["smoothing_settings"]
        self.n_random_samples = temp_algorithm_config["n_random_samples"]
        self.constrained_gmm_settings = temp_algorithm_config["constrained_gmm_settings"]
        self.trajectory_following_settings = temp_algorithm_config["trajectory_following_settings"]
        self.local_optimization_settings = temp_algorithm_config["local_optimization_settings"]
        self.global_spatial_optimization_settings = temp_algorithm_config["global_spatial_optimization_settings"]
        self.global_time_optimization_settings = temp_algorithm_config["global_time_optimization_settings"]
        self.constrained_sampling_mode = temp_algorithm_config["constrained_sampling_mode"]
        self.n_cluster_search_candidates = temp_algorithm_config["n_cluster_search_candidates"]
        self.debug_max_step = temp_algorithm_config["debug_max_step"]
        self.verbose = temp_algorithm_config["verbose"]
        self.average_elementary_action_error_threshold = temp_algorithm_config["average_elementary_action_error_threshold"]
        self.collision_avoidance_constraints_mode = temp_algorithm_config["collision_avoidance_constraints_mode"]
        self.optimize_collision_avoidance_constraints_extra = temp_algorithm_config["optimize_collision_avoidance_constraints_extra"]

    def build(self):
        return {"use_constraints": self.use_constraints,
                "local_optimization_mode": self.local_optimization_mode,
                "use_constrained_gmm": self.use_constrained_gmm,
                "use_transition_model": self.use_transition_model,
                "use_global_time_optimization": self.use_global_time_optimization,
                "use_global_spatial_optimization": self.use_global_spatial_optimization,
                "n_random_samples": self.n_random_samples,
                "activate_parameter_check": self.activate_parameter_check,
                "smoothing_settings": self.smoothing_settings,
                "local_optimization_settings": self.local_optimization_settings,
                "global_spatial_optimization_settings": self.global_spatial_optimization_settings,
                "global_time_optimization_settings": self.global_time_optimization_settings,
                "constrained_gmm_settings": self.constrained_gmm_settings,
                "trajectory_following_settings": self.trajectory_following_settings,
                "constrained_sampling_mode": self.constrained_sampling_mode,
                "n_cluster_search_candidates": self.n_cluster_search_candidates,
                "verbose": self.verbose,
                "average_elementary_action_error_threshold": self.average_elementary_action_error_threshold,
                "collision_avoidance_constraints_mode": self.collision_avoidance_constraints_mode,
                "optimize_collision_avoidance_constraints_extra": self.optimize_collision_avoidance_constraints_extra,
                "debug_max_step": self.debug_max_step
                }
