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
        self.global_spatial_optimization_mode = "none"
        self.n_random_samples = 300
        self.constrained_gmm_settings = dict()
        self.local_optimization_settings = dict()
        self.global_spatial_optimization_settings = dict()
        self.global_time_optimization_settings = dict()
        self.trajectory_following_settings = dict()
        self.inverse_kinematics_settings = dict()
        self.smoothing_settings = dict()
        self.constrained_sampling_mode = "random"
        self.n_cluster_search_candidates = 2
        self.debug_max_step = -1
        self.activate_inverse_kinematics = False
        self.verbose = False
        self.use_local_coordinates = False
        self.average_elementary_action_error_threshold = 500
        self.collision_avoidance_constraints_mode = "none"
        self.optimize_collision_avoidance_constraints_extra = False
        self.use_semantic_annotation_with_mgrd = False
        self.activate_time_variation = True
        self.set_default_constrained_gmm_settings()
        self.set_default_trajectory_following_settings()
        self.set_default_optimization_settings()
        self.set_default_smoothing_settings()
        self.set_default_inverse_kinematics_settings()

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
        self.global_spatial_optimization_settings["position_weight"] = 1000.0
        self.global_spatial_optimization_settings["orientation_weight"] = 500.0
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
        self.trajectory_following_settings["dir_constraint_factor"] = 1.0
        self.trajectory_following_settings["position_constraint_factor"] = 1.0
        self.trajectory_following_settings["transition_pose_constraint_factor"] = 1.0
        self.trajectory_following_settings["closest_point_search_accuracy"] = 0.001
        self.trajectory_following_settings["closest_point_search_max_iterations"] = 5000
        self.trajectory_following_settings["look_ahead_distance"] = 100
        self.trajectory_following_settings["end_step_length_factor"] = 1.0
        self.trajectory_following_settings["max_distance_to_path"] = 500
        self.trajectory_following_settings["arc_length_granularity"] = 1000
        self.trajectory_following_settings["use_transition_constraint"] = True

    def set_default_smoothing_settings(self):
        self.smoothing_settings = dict()
        self.smoothing_settings["spatial_smoothing"] = True
        self.smoothing_settings["time_smoothing"] = True
        self.smoothing_settings["spatial_smoothing_window"] = DEFAULT_SMOOTHING_WINDOW_SIZE
        self.smoothing_settings["time_smoothing_window"] = 15

    def set_default_inverse_kinematics_settings(self):
        self.inverse_kinematics_settings = dict()
        self.inverse_kinematics_settings["optimization_method"] = "L-BFGS-B"
        self.inverse_kinematics_settings["tolerance"] = 1e-10
        self.inverse_kinematics_settings["max_iterations"] = 500
        self.inverse_kinematics_settings["interpolation_window"] = 120
        self.inverse_kinematics_settings["use_euler_representation"] = False
        self.inverse_kinematics_settings["solving_method"] = "unconstrained"
        self.inverse_kinematics_settings["activate_look_at"] = True
        self.inverse_kinematics_settings["max_retries"] = 1
        self.inverse_kinematics_settings["success_threshold"] = 5.0
        self.inverse_kinematics_settings["transition_window"] = 60
        self.inverse_kinematics_settings["optimize_orientation"] = True
        self.inverse_kinematics_settings["elementary_action_max_iterations"] = 10
        self.inverse_kinematics_settings["elementary_action_optimization_eps"] = 1.0

    def from_json(self, filename):
        temp_algorithm_config = load_json_file(filename)
        for name in temp_algorithm_config.keys():
            if hasattr(self, name):
                setattr(self, name, temp_algorithm_config[name])

    def build(self):
        return {"use_constraints": self.use_constraints,
                "local_optimization_mode": self.local_optimization_mode,
                "use_constrained_gmm": self.use_constrained_gmm,
                "use_transition_model": self.use_transition_model,
                "use_global_time_optimization": self.use_global_time_optimization,
                "global_spatial_optimization_mode": self.global_spatial_optimization_mode,
                "n_random_samples": self.n_random_samples,
                "activate_parameter_check": self.activate_parameter_check,
                "smoothing_settings": self.smoothing_settings,
                "local_optimization_settings": self.local_optimization_settings,
                "global_spatial_optimization_settings": self.global_spatial_optimization_settings,
                "global_time_optimization_settings": self.global_time_optimization_settings,
                "constrained_gmm_settings": self.constrained_gmm_settings,
                "trajectory_following_settings": self.trajectory_following_settings,
                "inverse_kinematics_settings": self.inverse_kinematics_settings,
                "constrained_sampling_mode": self.constrained_sampling_mode,
                "n_cluster_search_candidates": self.n_cluster_search_candidates,
                "activate_inverse_kinematics": self.activate_inverse_kinematics,
                "verbose": self.verbose,
                "average_elementary_action_error_threshold": self.average_elementary_action_error_threshold,
                "collision_avoidance_constraints_mode": self.collision_avoidance_constraints_mode,
                "optimize_collision_avoidance_constraints_extra": self.optimize_collision_avoidance_constraints_extra,
                "debug_max_step": self.debug_max_step,
                "use_local_coordinates": self.use_local_coordinates,
                "use_semantic_annotation_with_mgrd": self.use_semantic_annotation_with_mgrd,
                "activate_time_variation": self.activate_time_variation
                }
