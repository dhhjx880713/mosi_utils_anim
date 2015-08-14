# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:19:46 2015

@author: erhe01
"""

from ..utilities.io_helper_functions import load_json_file


class AlgorithmConfigurationBuilder(object):
    """Can be used to generate a dict containing all settings for the algorithm    
    """
    def __init__(self):
        self.use_constraints = True
        self.use_optimization = True
        self.use_transition_model = False
        self.use_constrained_gmm = False
        self.activate_parameter_check = False
        self.use_global_optimization = False
        self.apply_smoothing = True
        self.smoothing_window = 20
        self.n_random_samples = 100
        self.constrained_gmm_settings = dict()
        self.optimization_settings = dict()
        self.trajectory_following_settings = dict()
        self.activate_cluster_search = True
        self.n_cluster_search_candidates = 2
        self.debug_max_step = -1
        self.verbose = False
        self.set_default_constrained_gmm_settings()
        self.set_default_trajectory_following_settings()
        self.set_default_optimization_settings()
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
        self.optimization_settings = dict()
        self.optimization_settings["method"] = "Nelder-Mead"
        self.optimization_settings["max_iterations"] = 150
        self.optimization_settings["quality_scale_factor"] = 0.001
        self.optimization_settings["error_scale_factor"] = 0.01
        self.optimization_settings["tolerance"] = 0.01
        self.optimization_settings["spatial_epsilon"] = 0.0
        return

    def set_default_trajectory_following_settings(self):
        self.trajectory_following_settings = dict()
        self.trajectory_following_settings["step_length_approx_method"] = "arc_length"
        self.trajectory_following_settings["heuristic_step_length_factor"] = 0.8
        self.trajectory_following_settings["dir_constraint_factor"] = 10.0
        self.trajectory_following_settings["position_constraint_factor"] = 1.0
        self.trajectory_following_settings["transition_pose_constraint_factor"] = 1.0

    def from_json(self, filename):
        temp_algorithm_config = load_json_file(filename)
        self.use_constraints = temp_algorithm_config["use_constraints"]
        self.use_optimization = temp_algorithm_config["use_optimization"]
        self.use_transition_model = temp_algorithm_config["use_transition_model"]
        self.use_constrained_gmm = temp_algorithm_config["use_constrained_gmm"]
        self.use_global_optimization = temp_algorithm_config["use_global_optimization"]
        self.activate_parameter_check = temp_algorithm_config["activate_parameter_check"]
        self.apply_smoothing = temp_algorithm_config["apply_smoothing"]
        self.smoothing_window = temp_algorithm_config["smoothing_window"]
        self.n_random_samples = temp_algorithm_config["n_random_samples"]
        self.constrained_gmm_settings = temp_algorithm_config["constrained_gmm_settings"]
        self.trajectory_following_settings = temp_algorithm_config["trajectory_following_settings"]
        self.optimization_settings = temp_algorithm_config["optimization_settings"]
        self.activate_cluster_search = temp_algorithm_config["activate_cluster_search"]
        self.n_cluster_search_candidates = temp_algorithm_config["n_cluster_search_candidates"]
        self.debug_max_step = temp_algorithm_config["debug_max_step"]
        self.verbose = temp_algorithm_config["verbose"]

    def build(self):
        return {"use_constraints": self.use_constraints,
                "use_optimization": self.use_optimization,
                "use_constrained_gmm": self.use_constrained_gmm,
                "use_transition_model": self.use_transition_model,
                "use_global_optimization": self.use_global_optimization,
                "n_random_samples": self.n_random_samples,
                "activate_parameter_check": self.activate_parameter_check,
                "apply_smoothing": self.apply_smoothing,
                "smoothing_window": self.smoothing_window,
                "optimization_settings": self.optimization_settings,
                "constrained_gmm_settings": self.constrained_gmm_settings,
                "trajectory_following_settings": self.trajectory_following_settings,
                "activate_cluster_search": self.activate_cluster_search,
                "n_cluster_search_candidates": self.n_cluster_search_candidates,
                "verbose": self.verbose,
                "debug_max_step": self.debug_max_step
                }

