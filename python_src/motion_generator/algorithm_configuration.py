# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:19:46 2015

@author: erhe01
"""

from utilities.io_helper_functions import load_json_file


class AlgorithmConfigurationBuilder(object):

    """Can be used to generate a dict containing all settings for the algorithm    
    """

    def __init__(self):
        self.use_constraints = True
        self.use_optimization = True
        self.use_transition_model = False
        self.use_constrained_gmm = False
        self.activate_parameter_check = False
        self.apply_smoothing = True
        self.smoothing_window = 20
        self.n_random_samples = 100
        self.constrained_gmm_pos_precision = 5
        self.constrained_gmm_rot_precision = 0.15
        self.constrained_gmm_smooth_precision = 5
        self.strict_constrained_gmm = False
        self.constrained_gmm_max_bad_samples = 200
        self.optimization_method = "Nelder-Mead"
        self.max_optimization_iterations = 150
        self.optimization_quality_scale_factor = 0.001
        self.optimization_error_scale_factor = 0.01
        self.optimization_tolerance = 0.05
        self.optimization_kinematic_epsilon = 0
        self.trajectory_extraction_method = "arc_length"
        self.trajectory_step_length_factor = 0.8
        self.trajectory_use_position_constraints = True
        self.trajectory_use_dir_vector_constraints = True
        self.trajectory_use_frame_constraints = True
        self.activate_cluster_search = True
        self.n_cluster_search_candidates = 2
        self.debug_max_step = -1
        self.verbose = False
        self.build()
        return

    def from_json(self, filename):
        temp_algorithm_config = load_json_file(filename)
        self.use_constraints = temp_algorithm_config["use_constraints"]
        self.use_optimization = temp_algorithm_config["use_optimization"]
        self.use_transition_model = temp_algorithm_config[
            "use_transition_model"]
        self.use_constrained_gmm = temp_algorithm_config["use_constrained_gmm"]
        self.activate_parameter_check = temp_algorithm_config[
            "activate_parameter_check"]
        self.apply_smoothing = temp_algorithm_config["apply_smoothing"]
        self.smoothing_window = temp_algorithm_config["smoothing_window"]
        self.n_random_samples = temp_algorithm_config["n_random_samples"]
        self.constrained_gmm_pos_precision = temp_algorithm_config[
            "constrained_gmm_settings"]["precision"]["pos"]
        self.constrained_gmm_rot_precision = temp_algorithm_config[
            "constrained_gmm_settings"]["precision"]["rot"]
        self.constrained_gmm_smooth_precision = temp_algorithm_config[
            "constrained_gmm_settings"]["precision"]["smooth"]
        self.strict_constrained_gmm = temp_algorithm_config[
            "constrained_gmm_settings"]["strict"]
        self.constrained_gmm_max_bad_samples = temp_algorithm_config[
            "constrained_gmm_settings"]["max_bad_samples"]
        self.optimization_method = temp_algorithm_config[
            "optimization_settings"]["method"]
        self.max_optimization_iterations = temp_algorithm_config[
            "optimization_settings"]["max_iterations"]
        self.optimization_quality_scale_factor = temp_algorithm_config[
            "optimization_settings"]["quality_scale_factor"]
        self.optimization_error_scale_factor = temp_algorithm_config[
            "optimization_settings"]["error_scale_factor"]
        self.optimization_tolerance = temp_algorithm_config[
            "optimization_settings"]["tolerance"]
        self.optimization_kinematic_epsilon = temp_algorithm_config[
            "optimization_settings"]["kinematic_epsilon"]
        self.trajectory_extraction_method = temp_algorithm_config[
            "trajectory_following_settings"]["method"]
        self.trajectory_step_length_factor = temp_algorithm_config[
            "trajectory_following_settings"]["step_length_factor"]
        self.trajectory_use_position_constraints = temp_algorithm_config[
            "trajectory_following_settings"]["use_position_constraints"]
        self.trajectory_use_dir_vector_constraints = temp_algorithm_config[
            "trajectory_following_settings"]["use_dir_vector_constraints"]
        self.trajectory_use_frame_constraints = temp_algorithm_config[
            "trajectory_following_settings"]["use_frame_constraints"]
        self.activate_cluster_search = temp_algorithm_config[
            "activate_cluster_search"]
        self.n_cluster_search_candidates = temp_algorithm_config[
            "n_cluster_search_candidates"]
        self.debug_max_step = temp_algorithm_config["debug_max_step"]
        self.verbose = temp_algorithm_config["verbose"]
        return

    def _generate_optimization_configuration(self):
        """ Generates optimization_settings dict that needs to be passed to run_optimization in optimize_motion_parameters.py
        """

        optimization_settings = {"method": self.optimization_method,
                                 "max_iterations": self.max_optimization_iterations,
                                 "quality_scale_factor": self.optimization_quality_scale_factor,
                                 "error_scale_factor": self.optimization_error_scale_factor,
                                 "tolerance": self.optimization_tolerance,
                                 "kinematic_epsilon": self.optimization_kinematic_epsilon}
        return optimization_settings

    def build(self):
        optimization_settings = self._generate_optimization_configuration()

        constrained_gmm_settings = {"precision": {"pos": self.constrained_gmm_pos_precision, "rot": self.constrained_gmm_rot_precision, "smooth": self.constrained_gmm_smooth_precision},
                                    "strict": self.strict_constrained_gmm,
                                    "max_bad_samples": self.constrained_gmm_max_bad_samples}
        trajectory_following_settings = {"method": self.trajectory_extraction_method,
                                         "step_length_factor": self.trajectory_step_length_factor,
                                         "use_position_constraints": self.trajectory_use_position_constraints,
                                         "use_dir_vector_constraints": self.trajectory_use_dir_vector_constraints,
                                         "use_frame_constraints": self.trajectory_use_frame_constraints
                                         }

        self._algorithm_config = {"use_constraints": self.use_constraints,
                                  "use_optimization": self.use_optimization,
                                  "use_constrained_gmm": self.use_constrained_gmm,
                                  "use_transition_model": self.use_transition_model,
                                  "n_random_samples": self.n_random_samples,
                                  "activate_parameter_check": self.activate_parameter_check,
                                  "apply_smoothing": self.apply_smoothing,
                                  "smoothing_window": self.smoothing_window,
                                  "optimization_settings": optimization_settings,
                                  "constrained_gmm_settings": constrained_gmm_settings,
                                  "trajectory_following_settings": trajectory_following_settings,
                                  "activate_cluster_search": self.activate_cluster_search,
                                  "n_cluster_search_candidates": self.n_cluster_search_candidates,
                                  "verbose": self.verbose,
                                  "debug_max_step": self.debug_max_step
                                  }

        return self._algorithm_config

    def get_configuration(self):
        """
        Returns
        -------
        algorithm_config : dict
          Settings that can be passed as parameter to the MotionGenerator pipeline
         """
        return self._algorithm_settings
