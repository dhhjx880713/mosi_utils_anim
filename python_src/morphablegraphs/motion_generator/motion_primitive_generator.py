# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

@author: erhe01,mamauer,hadu01,FARUPP
"""

import numpy as np
import time
from ..animation_data.evaluation_methods import check_sample_validity
from statistics import ConstrainedGMMBuilder
from ..utilities.exceptions import ConstraintError, SynthesisError
from optimization import OptimizerBuilder
from objective_functions import obj_spatial_error_sum, obj_spatial_error_sum_and_naturalness


class MotionPrimitiveGenerator(object):
    """
    Parameters
    ----------
    * action_constraints : ElementaryActionConstraints
        contains reference to motion data structure
    * algorithm_config : dict
        Contains settings for the algorithm.
    """
    def __init__(self, action_constraints, algorithm_config):
        self._action_constraints = action_constraints
        self._algorithm_config = algorithm_config
        self.action_name = action_constraints.action_name
        self.prev_action_name = action_constraints.prev_action_name
        self._motion_primitive_graph = self._action_constraints.motion_primitive_graph
        self.skeleton = self._action_constraints.get_skeleton()
        self._constrained_gmm_config = self._algorithm_config["constrained_gmm_settings"]
        self.n_random_samples = self._algorithm_config["n_random_samples"]
        self.verbose = self._algorithm_config["verbose"]
        self.precision = self._constrained_gmm_config["precision"]        
        self.max_bad_samples = self._constrained_gmm_config["max_bad_samples"]
        self.use_constraints = self._algorithm_config["use_constraints"]
        self.local_optimization_mode = self._algorithm_config["local_optimization_mode"]
        self._optimization_settings = self._algorithm_config["local_optimization_settings"]
        self.optimization_start_error_threshold = self._optimization_settings["start_error_threshold"]
        self.use_constrained_gmm = self._algorithm_config["use_constrained_gmm"]
        self.use_transition_model = self._algorithm_config["use_transition_model"]
        self.activate_cluster_search = self._algorithm_config["activate_cluster_search"]
        self.activate_parameter_check = self._algorithm_config["activate_parameter_check"]
        self.n_cluster_search_candidates = self._algorithm_config["n_cluster_search_candidates"]
        if self.use_constrained_gmm:
            self._constrained_gmm_builder = ConstrainedGMMBuilder(self._motion_primitive_graph, self._algorithm_config,
                                                                  self._action_constraints.start_pose, self.skeleton)
        else:
            self._constrained_gmm_builder = None
        self.numerical_minimizer = OptimizerBuilder(self._algorithm_config).build_spatial_and_naturalness_error_minimizer()

    def generate_motion_primitive_sample(self, motion_primitive_constraints, graph_walk):
        """Calls get_optimal_parameters and backpojects the results.
        
        Parameters
        ----------
        *motion_primitive_constraints: MotionPrimitiveConstraints
            Constraints specific for the current motion primitive.
        *graph_walk: GraphWalk
            Annotated motion with information on the graph walk.
            
        Returns
        -------
        * motion_primitive_sample: MotionSpline
            back projected motion primitive sample
        """
        mp_name = motion_primitive_constraints.motion_primitive_name
        if len(graph_walk.steps) > 0:
            prev_mp_name = graph_walk.steps[-1].node_key[1]
            prev_parameters = graph_walk.steps[-1].parameters
        else:
            prev_mp_name = ""
            prev_parameters = None

        start = time.clock()
        if self.use_constraints and len(motion_primitive_constraints.constraints) > 0:
            try:
                low_dimensional_parameters = self.get_optimal_motion_primitive_parameters(mp_name,
                                                                                          motion_primitive_constraints,
                                                                                          prev_mp_name=prev_mp_name,
                                                                                          prev_frames=graph_walk.get_quat_frames(),
                                                                                          prev_parameters=prev_parameters)
            except ConstraintError as exception:
                print "Exception", exception.message
                raise SynthesisError(graph_walk.get_quat_frames(), exception.bad_samples)
        else:  # no constraints were given
                print "pick random sample for motion primitive", mp_name
                low_dimensional_parameters = self._get_random_parameters(mp_name, prev_mp_name, prev_parameters)
        time_in_seconds = time.clock() - start
        print "found best fit motion primitive sample in " + str(time_in_seconds) + " seconds"

        motion_primitive_sample = self._motion_primitive_graph.nodes[(self.action_name, mp_name)].back_project(low_dimensional_parameters, use_time_parameters=False)
        return motion_primitive_sample

    def get_optimal_motion_primitive_parameters(self, mp_name,
                                                motion_primitive_constraints,
                                                prev_mp_name="",
                                                prev_frames=None,
                                                prev_parameters=None):
        """Uses the constraints to find the optimal paramaters for a motion primitive.
        Parameters
        ----------
        * mp_name : string
            name of the motion primitive and node in the subgraph of the elemantary action
        * motion_primitive_constraints: MotionPrimitiveConstraints
            contains a list of dict with constraints for joints
        * prev_frames: np.ndarray
            quaternion frames
        Returns
        -------
        * parameters : np.ndarray
            Low dimensional parameters for the morphable model
        """
        graph_node = self._motion_primitive_graph.nodes[(self.action_name, mp_name)]
        close_to_optimum = False
        if self.activate_cluster_search and graph_node.cluster_tree is not None:
            parameters = self._search_for_best_sample_in_cluster_tree(graph_node,
                                                                      motion_primitive_constraints,
                                                                      prev_frames)


        else:
            parameters = self._get_best_random_sample_from_statistical_model(graph_node,
                                                                             mp_name,
                                                                             motion_primitive_constraints,
                                                                             prev_mp_name,
                                                                             prev_frames,
                                                                             prev_parameters)
        if motion_primitive_constraints.min_error <= self.optimization_start_error_threshold:
            close_to_optimum = True
        #print "condition", not self.use_transition_model, motion_primitive_constraints.use_local_optimization, not close_to_optimum, self.optimization_start_error_threshold
        if not self.use_transition_model and motion_primitive_constraints.use_local_optimization and not close_to_optimum:
            data = graph_node, motion_primitive_constraints, \
                   prev_frames, self._optimization_settings["error_scale_factor"], \
                   self._optimization_settings["quality_scale_factor"]

            self.numerical_minimizer.set_objective_function_parameters(data)
            parameters = self.numerical_minimizer.run(initial_guess=parameters)
        return parameters

    def _get_best_random_sample_from_statistical_model(self, graph_node, mp_name, motion_primitive_constraints, prev_mp_name, prev_frames, prev_parameters):
        #  1) get gaussian_mixture_model and modify it based on the current state and settings
        if self._constrained_gmm_builder is not None:
            gmm = self._constrained_gmm_builder.build(self.action_name, mp_name, motion_primitive_constraints,
                                                      self.prev_action_name, prev_mp_name,
                                                      prev_frames, prev_parameters)

        elif self.use_transition_model and prev_parameters is not None:
            gmm = self._predict_gmm(mp_name, prev_mp_name, prev_parameters)
        else:
            gmm = graph_node.gaussian_mixture_model
        #  2) sample parameters  from the Gaussian Mixture Model based on constraints and make sure
        #     the resulting motion is valid
        if not self.activate_parameter_check:
            parameters, min_error = self.sample_from_gaussian_mixture_model(graph_node, gmm,
                                                                        motion_primitive_constraints,
                                                                        prev_frames)
        else:
            parameters, min_error = self.sample_from_gaussian_mixture_model_with_validity_check(graph_node, gmm,
                                                                        motion_primitive_constraints,
                                                                        prev_frames)

        return parameters

    def _get_random_parameters(self, mp_name, prev_mp_name="", prev_parameters=None):
        if self.use_transition_model and prev_parameters is not None and self._motion_primitive_graph.nodes[(self.prev_action_name, prev_mp_name)].has_transition_model((self.action_name, mp_name)):
            parameters = self._motion_primitive_graph.nodes[(self.prev_action_name, prev_mp_name)].predict_parameters((self.action_name, mp_name), prev_parameters)
        else:
            parameters = self._motion_primitive_graph.nodes[(self.action_name, mp_name)].sample_low_dimensional_vector()
        return parameters

    def _predict_gmm(self, mp_name, prev_mp_name, prev_parameters):
        to_key = (self.action_name, mp_name)
        gmm = self._motion_primitive_graph.nodes[(self.prev_action_name,prev_mp_name)].predict_gmm(to_key, prev_parameters)
        return gmm
    
    def _search_for_best_sample_in_cluster_tree(self, graph_node, constraints, prev_frames):
        """ Directed search in precomputed hierarchical space partitioning data structure
        """
        data = graph_node, constraints, prev_frames
        distance, s = graph_node.search_best_sample(obj_spatial_error_sum, data, self.n_cluster_search_candidates)
        print "found best sample with distance:", distance
        constraints.min_error = distance
        return np.array(s)                                 

    def sample_from_gaussian_mixture_model(self, mp_node, gmm, constraints, prev_frames):
        """samples and picks the best samples out of a given set, quality measure
        is naturalness
    
        Parameters
        ----------
        * mp_node : MotionState
            contains a motion primitive and meta information
        * gmm : sklearn.mixture.gmm
            The gmm to sample
        * constraints: MotionPrimitiveConstraints
        contains a list of dict with constraints for joints
        * prev_frames: list
            A list of quaternion frames
        Returns
        -------
        * sample : numpy.ndarray
            The best sample out of those which have been created
        * error : bool
            the error of the best sample
        """
        best_sample = None
        min_error = 1000000.0
        count = 0
        while count < self.n_random_samples:
            parameter_sample = np.ravel(gmm.sample())
            object_function_params = mp_node, constraints, prev_frames
            error = obj_spatial_error_sum(parameter_sample, object_function_params)
            if min_error > error:
                min_error = error
                best_sample = parameter_sample
            count += 1

        print "found best sample with distance:", min_error
        constraints.min_error = min_error
        return best_sample, min_error

    def sample_from_gaussian_mixture_model_with_validity_check(self, mp_node, gmm, constraints, prev_frames):
        """samples and picks the best samples out of a given set, quality measure
        is naturalness

        Parameters
        ----------
        * mp_node : MotionState
            contains a motion primitive and meta information
        * gmm : sklearn.mixture.gmm
            The gmm to sample
        * constraints: MotionPrimitiveConstraints
        contains a list of dict with constraints for joints
        * prev_frames: list
            A list of quaternion frames
        Returns
        -------
        * sample : numpy.ndarray
            The best sample out of those which have been created
        * error : bool
            the error of the best sample
        """
        best_sample = None
        min_error = 1000000.0
        reached_max_bad_samples = False
        tmp_bad_samples = 0
        count = 0
        while count < self.n_random_samples:
            if tmp_bad_samples > self.max_bad_samples:
                    reached_max_bad_samples = True
            parameter_sample = np.ravel(gmm.sample())
            # using bounding box to check sample is good or bad
            valid = check_sample_validity(mp_node, parameter_sample, self.skeleton)

            if valid:
                object_function_params = mp_node, constraints, prev_frames
                error = obj_spatial_error_sum(parameter_sample, object_function_params)
                if min_error > error:
                    min_error = error
                    best_sample = parameter_sample
                count += 1
            else:
                if self.verbose:
                    print "sample failed validity check"
                tmp_bad_samples += 1

        if reached_max_bad_samples:
            print "Warning: Failed to pick good sample from GMM"
            return best_sample, min_error

        print "found best sample with distance:", min_error
        constraints.min_error = min_error
        return best_sample, min_error