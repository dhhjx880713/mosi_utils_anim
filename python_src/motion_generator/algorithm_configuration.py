# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:19:46 2015

@author: erhe01
"""

class AlgorithmConfigurationBuilder(object):
    """Can be used to generate a dict containing all settings for the algorithm
    Parameters
    ----------
    * max_step : integer
    Debug parameter for the maximum number of motion primitives to be converted before stopping.
    If set to -1 this parameter is ignored
    
    """
    def __init__(self,use_constraints=True,
                        use_optimization=True,
                        use_transition_model=False,
                        use_constrained_gmm=False,
                        activate_parameter_check=False,
                        apply_smoothing=True,
                        sample_size=100,
                        constrained_gmm_pos_precision=5,
                        constrained_gmm_rot_precision=0.15,
                        constrained_gmm_smooth_precision=5,
                        strict_constrained_gmm=False,
                        constrained_gmm_max_bad_samples=200,
                        optimization_method="BFGS", 
                        max_optimization_iterations=150,
                        optimization_quality_scale_factor=0.001,
                        optimization_error_scale_factor=0.01,
                        optimization_tolerance=0.05,
                        optimization_kinematic_epsilon=0,
                        trajectory_extraction_method="arc_length",
                        trajectory_step_length_factor=0.8,
                        trajectory_use_position_constraints=True,
                        trajectory_use_dir_vector_constraints=True,
                        trajectory_use_frame_constraints=True,
                        activate_cluster_search=True,
                        debug_max_step=-1,
                        verbose=False):
        self.use_constraints = use_constraints
        self.use_optimization = use_optimization
        self.use_transition_model = use_transition_model
        self.use_constrained_gmm = use_constrained_gmm
        self.activate_parameter_check = activate_parameter_check
        self.apply_smoothing = apply_smoothing
        self.sample_size = sample_size
        self.constrained_gmm_pos_precision = constrained_gmm_pos_precision
        self.constrained_gmm_rot_precision = constrained_gmm_rot_precision
        self.constrained_gmm_smooth_precision = constrained_gmm_smooth_precision
        self.strict_constrained_gmm = strict_constrained_gmm
        self.constrained_gmm_max_bad_samples = constrained_gmm_max_bad_samples
        self.optimization_method = optimization_method
        self.max_optimization_iterations = max_optimization_iterations
        self.optimization_quality_scale_factor = optimization_quality_scale_factor
        self.optimization_error_scale_factor = optimization_error_scale_factor
        self.optimization_tolerance = optimization_tolerance
        self.optimization_kinematic_epsilon = optimization_kinematic_epsilon
        self.trajectory_extraction_method = trajectory_extraction_method
        self.trajectory_step_length_factor = trajectory_step_length_factor
        self.trajectory_use_position_constraints = trajectory_use_position_constraints
        self.trajectory_use_dir_vector_constraints = trajectory_use_dir_vector_constraints
        self.trajectory_use_frame_constraints = trajectory_use_frame_constraints
        self.activate_cluster_search = activate_cluster_search
        self.debug_max_step = debug_max_step
        self.verbose = verbose
        self.build()
        return
        
        
    def _generate_optimization_configuration(self, method="BFGS",max_iterations=100,quality_scale_factor=1,
                                   error_scale_factor=0.1,tolerance=0.05,optimize_theta=False,kinematic_epsilon=5):
        """ Generates optimization_settings dict that needs to be passed to the run_optimization mbvh_readerethod
        """
    
        optimization_settings = {"method":method, 
                 "max_iterations"  : max_iterations,
                "quality_scale_factor":quality_scale_factor,
                "error_scale_factor": error_scale_factor,
                "optimize_theta":optimize_theta,
                "tolerance":tolerance,
                "kinematic_epsilon":kinematic_epsilon}
        return optimization_settings
        


    def build(self):
        optimization_settings = self._generate_optimization_configuration(method=self.optimization_method,max_iterations=self.max_optimization_iterations,
                                                              quality_scale_factor=self.optimization_quality_scale_factor ,
                                                              error_scale_factor=self.optimization_error_scale_factor,
                                                              tolerance=self.optimization_tolerance,
                                                              kinematic_epsilon=self.optimization_kinematic_epsilon)
        constrained_gmm_settings ={"sample_size" : self.sample_size,
                               "precision" : {"pos" : self.constrained_gmm_pos_precision,"rot" : self.constrained_gmm_rot_precision,"smooth":self.constrained_gmm_smooth_precision},
                               "strict" : self.strict_constrained_gmm,
                               "max_bad_samples":self.constrained_gmm_max_bad_samples}
        trajectory_following_settings = {"method" : self.trajectory_extraction_method,
                                     "step_length_factor" : self.trajectory_step_length_factor,
                                     "use_position_constraints":self.trajectory_use_position_constraints, 
                                     "use_dir_vector_constraints"  : self.trajectory_use_dir_vector_constraints,
                                     "use_frame_constraints":self.trajectory_use_frame_constraints
                                     }
    
        self._algorithm_config = {"use_constraints": self.use_constraints,
                                   "use_optimization": self.use_optimization,
                                   "use_constrained_gmm" : self.use_constrained_gmm,
                                   "use_transition_model": self.use_transition_model,
                                   "activate_parameter_check": self.activate_parameter_check,
                                   "apply_smoothing": self.apply_smoothing,
                                   "optimization_settings": optimization_settings,
                                   "constrained_gmm_settings": constrained_gmm_settings,
                                   "trajectory_following_settings" : trajectory_following_settings,
                                   "activate_cluster_search" : self.activate_cluster_search,
                                   "verbose": self.verbose, 
                                   "debug_max_step": self.debug_max_step
                                    }
        return
        
        
    def get_configuration(self):
        """
        Returns
        -------
        algorithm_config : dict
          Settings that can be passed as parameter to the MotionGenerator pipeline
         """
        return self._algorithm_settings
