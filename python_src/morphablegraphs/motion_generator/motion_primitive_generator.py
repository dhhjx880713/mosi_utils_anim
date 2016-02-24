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
from mgrd import motion_primitive_get_random_samples
from mgrd_filter import MGRDFilter
SAMPLING_MODE_RANDOM = "random_discrete"
SAMPLING_MODE_CLUSTER_SEARCH = "cluster_search"
SAMPLING_MODE_RANDOM_SPLINE = "random_spline"


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
        self._motion_state_graph = self._action_constraints.motion_state_graph
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
        self.constrained_sampling_mode = self._algorithm_config["constrained_sampling_mode"]
        self.activate_parameter_check = self._algorithm_config["activate_parameter_check"]
        self.n_cluster_search_candidates = self._algorithm_config["n_cluster_search_candidates"]
        if self.use_constrained_gmm:
            self._constrained_gmm_builder = ConstrainedGMMBuilder(self._motion_state_graph, self._algorithm_config,
                                                                  self._action_constraints.start_pose, self.skeleton)
        else:
            self._constrained_gmm_builder = None
        self.numerical_minimizer = OptimizerBuilder(self._algorithm_config).build_spatial_and_naturalness_error_minimizer()
        self.mgrd_filter = MGRDFilter()

    def generate_constrained_motion_spline(self, mp_constraints, prev_graph_walk):
        """Calls generate_constrained_sample and backpojects the result to a MotionSpline.

        Parameters
        ----------
        *mp_constraints: MotionPrimitiveConstraints
            Constraints specific for the current motion primitive.
        *prev_graph_walk: GraphWalk
            Annotated motion with information on the graph walk.

        Returns
        -------
        * motion_spline: MotionSpline
            back projected motion primitive sample
        * parameters:
            low dimensional motion primitive sample vector
        """
        mp_name = mp_constraints.motion_primitive_name
        if len(prev_graph_walk.steps) > 0:
            prev_mp_name = prev_graph_walk.steps[-1].node_key[1]
            prev_parameters = prev_graph_walk.steps[-1].parameters
        else:
            prev_mp_name = ""
            prev_parameters = None

        start = time.clock()
        if self.use_constraints and len(mp_constraints.constraints) > 0:
            try:
                parameters = self.generate_constrained_sample(mp_name, mp_constraints,prev_mp_name,
                                                              prev_graph_walk.get_quat_frames(), prev_parameters)
            except ConstraintError as exception:
                print "Exception", exception.message
                raise SynthesisError(prev_graph_walk.get_quat_frames(), exception.bad_samples)
        else:  # no constraints were given
            print "pick random sample for motion primitive", mp_name
            parameters = self.generate_random_sample(mp_name, prev_mp_name, prev_parameters)
        time_in_seconds = time.clock() - start
        print "found best fit motion primitive sample in " + str(time_in_seconds) + " seconds"

        motion_spline = self._motion_state_graph.nodes[(self.action_name, mp_name)].back_project(parameters, use_time_parameters=False)
        return motion_spline, parameters

    def generate_constrained_sample(self, mp_name, mp_constraints, prev_mp_name="", prev_frames=None, prev_parameters=None):
        """Uses the constraints to find the optimal parameters for a motion primitive.
        Parameters
        ----------
        * mp_name : string
            name of the motion primitive and node in the subgraph of the elementary action
        * mp_constraints: MotionPrimitiveConstraints
            contains a list of dict with constraints for joints
        * prev_frames: np.ndarray
            quaternion frames
        Returns
        -------
        * sample : np.ndarray
            Low dimensional parameters for the morphable model
        """
        graph_node = self._motion_state_graph.nodes[(self.action_name, mp_name)]
        #prev_frames = None
        #mp_constraints = mp_constraints.transform_constraints_to_local_cos()
        if self.constrained_sampling_mode == SAMPLING_MODE_RANDOM_SPLINE:
            sample = self._get_best_fit_random_sample_using_mgrd(graph_node, mp_constraints)
        elif self.constrained_sampling_mode == SAMPLING_MODE_CLUSTER_SEARCH and graph_node.cluster_tree is not None:
            sample = self._search_for_best_fit_sample_in_cluster_tree(graph_node, mp_constraints, prev_frames)
        else:
            sample = self._get_best_fit_random_sample(graph_node, mp_name, mp_constraints,
                                                          prev_mp_name, prev_frames, prev_parameters)
        if not self.use_transition_model and mp_constraints.use_local_optimization and mp_constraints.min_error <= self.optimization_start_error_threshold:
            sample = self._optimize_parameters_numerically(sample, graph_node, mp_constraints, prev_frames)
        return sample

    def _get_best_fit_random_sample_using_mgrd(self, graph_node, mp_constraints):
        #TODO handle transformation of motion primitive to global coordinate system or constraints to local coordinate system of motion primitive based on the previous motion
        samples = motion_primitive_get_random_samples(graph_node.motion_primitive, self.n_random_samples)
        scores = self.mgrd_filter.score_samples(graph_node.motion_primitive, samples, mp_constraints)
        best_idx = np.argmin(scores)
        mp_constraints.min_error = scores[best_idx]
        print "Found best sample with score", scores[best_idx]
        return samples[best_idx]

    def _optimize_parameters_numerically(self, inital_guess, graph_node, mp_constraints, prev_frames):
         #print "condition", not self.use_transition_model, mp_constraints.use_local_optimization, not close_to_optimum, self.optimization_start_error_threshold
        data = graph_node, mp_constraints, prev_frames, self._optimization_settings["error_scale_factor"], self._optimization_settings["quality_scale_factor"]
        self.numerical_minimizer.set_objective_function_parameters(data)
        return self.numerical_minimizer.run(initial_guess=inital_guess)

    def _get_best_fit_random_sample(self, graph_node, mp_name, mp_constraints, prev_mp_name, prev_frames, prev_parameters):
        #  1) get gaussian_mixture_model and modify it based on the current state and settings
        if self._constrained_gmm_builder is not None:
            gmm = self._constrained_gmm_builder.build(self.action_name, mp_name, mp_constraints,
                                                      self.prev_action_name, prev_mp_name,
                                                      prev_frames, prev_parameters)

        elif self.use_transition_model and prev_parameters is not None:
            gmm = self._predict_gmm(mp_name, prev_mp_name, prev_parameters)
        else:
            gmm = graph_node.get_gaussian_mixture_model()
        #  2) sample parameters  from the Gaussian Mixture Model based on constraints and make sure
        #     the resulting motion is valid
        if not self.activate_parameter_check:
            parameters, min_error = self._sample_from_gmm_using_constraints(graph_node, gmm, mp_constraints, prev_frames)
        else:
            parameters, min_error = self._sample_from_gmm_using_constraints_and_validity_check(graph_node, gmm,
                                                                                               mp_constraints,
                                                                                               prev_frames)

        print "Found best sample with distance:", min_error
        return parameters

    def generate_random_sample(self, mp_name, prev_mp_name="", prev_parameters=None):
        if self.use_transition_model and prev_parameters is not None and self._motion_state_graph.nodes[(self.prev_action_name, prev_mp_name)].has_transition_model((self.action_name, mp_name)):
            parameters = self._motion_state_graph.nodes[(self.prev_action_name, prev_mp_name)].predict_parameters((self.action_name, mp_name), prev_parameters)
        else:
            parameters = self._motion_state_graph.nodes[(self.action_name, mp_name)].sample_low_dimensional_vector()
        return parameters

    def _predict_gmm(self, mp_name, prev_mp_name, prev_parameters):
        to_key = (self.action_name, mp_name)
        gmm = self._motion_state_graph.nodes[(self.prev_action_name,prev_mp_name)].predict_gmm(to_key, prev_parameters)
        return gmm

    def _search_for_best_fit_sample_in_cluster_tree(self, graph_node, constraints, prev_frames):
        """ Directed search in precomputed hierarchical space partitioning data structure
        """
        data = graph_node, constraints, prev_frames
        distance, s = graph_node.search_best_sample(obj_spatial_error_sum, data, self.n_cluster_search_candidates)
        print "found best sample with distance:", distance
        constraints.min_error = distance
        return np.array(s)

    def _sample_from_gmm_using_constraints(self, mp_node, gmm, constraints, prev_frames):
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
        min_error = np.inf
        idx = 0
        samples = gmm.sample(self.n_random_samples)
        while idx < self.n_random_samples:
            object_function_params = mp_node, constraints, prev_frames
            error = obj_spatial_error_sum(samples[idx], object_function_params)
            if min_error > error:
                min_error = error
                best_sample = samples[idx]
            idx += 1
        constraints.min_error = min_error
        return best_sample, min_error

    def _sample_from_gmm_using_constraints_and_validity_check(self, mp_node, gmm, constraints, prev_frames):
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
        min_error = np.inf
        reached_max_bad_samples = False
        tmp_bad_samples = 0
        idx = 0
        samples = np.ravel(gmm.sample(self.n_random_samples))
        while idx < self.n_random_samples:
            if tmp_bad_samples > self.max_bad_samples:
                    reached_max_bad_samples = True
            # using bounding box to check sample is good or bad
            valid = check_sample_validity(mp_node, samples[idx], self.skeleton)

            if valid:
                object_function_params = mp_node, constraints, prev_frames
                error = obj_spatial_error_sum(samples[idx], object_function_params)
                if min_error > error:
                    min_error = error
                    best_sample = samples[idx]
            else:
                if self.verbose:
                    print "sample failed validity check"
                tmp_bad_samples += 1
            idx += 1
        if reached_max_bad_samples:
            print "Warning: Failed to pick good sample from GMM"
            return best_sample, min_error
        constraints.min_error = min_error
        return best_sample, min_error
