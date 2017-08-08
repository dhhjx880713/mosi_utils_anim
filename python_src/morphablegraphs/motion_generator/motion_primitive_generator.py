# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

@author: erhe01,mamauer,hadu01,FARUPP
"""

import time
import numpy as np
from .optimization import OptimizerBuilder
from ..constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE
from .optimization.objective_functions import obj_spatial_error_sum
from .statistics import ConstrainedGMMBuilder
from ..utilities.exceptions import ConstraintError, SynthesisError
try:
    from mgrd import motion_primitive_get_random_samples
except ImportError:
    pass
from .mgrd_motion_primitive_sample_filter import MGRDMotionPrimitiveSampleFilter as MGRDFilter
from ..utilities import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_ERROR
SAMPLING_MODE_RANDOM = "random_discrete"
SAMPLING_MODE_CLUSTER_TREE_SEARCH = "cluster_tree_search"
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
        self.set_algorithm_config(algorithm_config)
        self.action_name = action_constraints.action_name
        self.prev_action_name = action_constraints.prev_action_name
        self._motion_state_graph = self._action_constraints.motion_state_graph
        self.skeleton = self._action_constraints.get_skeleton()
        if self.use_constrained_gmm:
            self._constrained_gmm_builder = ConstrainedGMMBuilder(self._motion_state_graph, self._algorithm_config,
                                                                  self._action_constraints.start_pose, self.skeleton)
        else:
            self._constrained_gmm_builder = None
        self.numerical_minimizer = OptimizerBuilder(self._algorithm_config).build_spatial_and_naturalness_error_minimizer()
        self.mgrd_filter = MGRDFilter()

    def set_algorithm_config(self, algorithm_config):
        self._algorithm_config = algorithm_config
        self._constrained_gmm_config = self._algorithm_config["constrained_gmm_settings"]
        self.n_random_samples = self._algorithm_config["n_random_samples"]
        self.verbose = self._algorithm_config["verbose"]
        self.precision = self._constrained_gmm_config["precision"]
        self.max_bad_samples = self._constrained_gmm_config["max_bad_samples"]
        self.use_constraints = self._algorithm_config["use_constraints"]
        self.local_optimization_mode = self._algorithm_config["local_optimization_mode"]
        self._settings = self._algorithm_config["local_optimization_settings"]
        self.optimization_start_error_threshold = self._settings["start_error_threshold"]
        self.use_constrained_gmm = self._algorithm_config["use_constrained_gmm"]
        self.use_transition_model = self._algorithm_config["use_transition_model"]
        self.constrained_sampling_mode = self._algorithm_config["constrained_sampling_mode"]
        self.activate_parameter_check = self._algorithm_config["activate_parameter_check"]
        self.n_cluster_search_candidates = self._algorithm_config["n_cluster_search_candidates"]
        self.use_local_coordinates = self._algorithm_config["use_local_coordinates"]
        self.use_semantic_annotation_with_mgrd = self._algorithm_config["use_semantic_annotation_with_mgrd"]

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
        node_key = (self.action_name, mp_constraints.motion_primitive_name)
        if len(prev_graph_walk.steps) > 0:
            prev_mp_name = prev_graph_walk.steps[-1].node_key[1]
            prev_parameters = prev_graph_walk.steps[-1].parameters
        else:
            prev_mp_name = ""
            prev_parameters = None

        start = time.clock()
        if self.use_constraints and len(mp_constraints.constraints) > 0:
            try:
                graph_node = self._motion_state_graph.nodes[node_key]
                parameters = self.generate_constrained_sample(graph_node, mp_constraints, prev_mp_name,
                                                              prev_graph_walk.get_quat_frames(), prev_parameters)
            except ConstraintError as exception:
                write_message_to_log("Exception " + exception.message, mode=LOG_MODE_ERROR)
                raise SynthesisError(prev_graph_walk.get_quat_frames(), exception.bad_samples)
        else:  # no constraints were given
            write_message_to_log("No constraints specified pick random sample instead", mode=LOG_MODE_DEBUG)
            parameters = self.generate_random_sample(node_key, prev_mp_name, prev_parameters)
        mp_constraints.time = time.clock() - start
        write_message_to_log("Found best fit motion primitive sample in " + str(mp_constraints.time) + " seconds", mode=LOG_MODE_DEBUG)

        motion_spline = self._motion_state_graph.nodes[node_key].back_project(parameters, use_time_parameters=False)
        return motion_spline, parameters

    def generate_constrained_sample(self, graph_node, in_mp_constraints, prev_mp_name="", prev_frames=None, prev_parameters=None):
        """Uses the constraints to find the optimal parameters for a motion primitive.
        Parameters
        ----------
        * mp_name : string
            name of the motion primitive and node in the subgraph of the elementary action
        * in_mp_constraints: MotionPrimitiveConstraints
            contains a list of dict with constraints for joints
        * prev_frames: np.ndarray
            quaternion frames
        Returns
        -------
        * sample : np.ndarray
            Low dimensional parameters for the morphable model
        """

        if self.use_local_coordinates:
            prev_frames_copy = None
            mp_constraints = in_mp_constraints.transform_constraints_to_local_cos()
        else:
            mp_constraints = in_mp_constraints
            prev_frames_copy = prev_frames

        if self.constrained_sampling_mode == SAMPLING_MODE_RANDOM_SPLINE:
            sample = self._get_best_fit_sample_using_mgrd(graph_node, mp_constraints)
        elif self.constrained_sampling_mode == SAMPLING_MODE_CLUSTER_TREE_SEARCH and graph_node.cluster_tree is not None:
            sample = self._get_best_fit_sample_using_cluster_tree(graph_node, mp_constraints, prev_frames_copy)
        else:
            sample = self._get_best_fit_sample_using_gmm(graph_node, mp_constraints, prev_mp_name,
                                                         prev_frames_copy, prev_parameters)
        #write_log("start optimization", self._is_optimization_required(mp_constraints),mp_constraints.use_local_optimization,mp_constraints.min_error,self.optimization_start_error_threshold)
        if self._is_optimization_required(mp_constraints):
            sample = self._optimize_parameters_numerically(sample, graph_node, mp_constraints, prev_frames_copy)
        in_mp_constraints.min_error = mp_constraints.min_error
        in_mp_constraints.evaluations = mp_constraints.evaluations
        return sample

    def _is_optimization_required(self, mp_constraints):
        return mp_constraints.use_local_optimization and not self.use_transition_model and \
               mp_constraints.min_error >= self.optimization_start_error_threshold

    def _get_best_fit_sample_using_mgrd(self, graph_node, mp_constraints):
        samples = motion_primitive_get_random_samples(graph_node.motion_primitive, self.n_random_samples)
        scores = MGRDFilter.score_samples(graph_node.motion_primitive, samples, *mp_constraints.convert_to_mgrd_constraints(self.use_semantic_annotation_with_mgrd), weights=(1,1))
        if scores is not None:
            best_idx = np.argmin(scores)
            mp_constraints.min_error = scores[best_idx]
            write_message_to_log("Found best sample with score "+ str(scores[best_idx]), LOG_MODE_DEBUG)
            return samples[best_idx]
        else:
            write_message_to_log("Error: MGRD returned None. Use random sample instead.", LOG_MODE_ERROR)
            return graph_node.sample_low_dimensional_vector()

    def _optimize_parameters_numerically(self, initial_guess, graph_node, mp_constraints, prev_frames):
        #print "condition", not self.use_transition_model, mp_constraints.use_local_optimization#, not close_to_optimum, self.optimization_start_error_threshold
        #mp_constraints.constraints = [c for c in mp_constraints.constraints if c.constraint_type != SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSE]
        if len(mp_constraints.constraints) > 0:
            data = (graph_node, mp_constraints, prev_frames, self._settings["error_scale_factor"], self._settings["quality_scale_factor"], 1.0)
            error_sum = max(abs(np.sum(self.numerical_minimizer._objective_function(initial_guess, data))), 1.0)
            data = graph_node, mp_constraints, prev_frames, self._settings["error_scale_factor"], self._settings["quality_scale_factor"], error_sum
            self.numerical_minimizer.set_objective_function_parameters(data)
            return self.numerical_minimizer.run(initial_guess=initial_guess)
        else:
            return initial_guess

    def _get_best_fit_sample_using_gmm(self, graph_node, mp_constraints, prev_mp_name, prev_frames, prev_parameters):
        #  1) get gaussian_mixture_model and modify it based on the current state and settings
        if self._constrained_gmm_builder is not None:
            gmm = self._constrained_gmm_builder.build(self.action_name, mp_constraints.motion_primitive_name, mp_constraints,
                                                      self.prev_action_name, prev_mp_name,
                                                      prev_frames, prev_parameters)

        elif self.use_transition_model and prev_parameters is not None:
            gmm = self._predict_gmm(mp_constraints.motion_primitive_name, prev_mp_name, prev_parameters)
        else:
            gmm = graph_node.get_gaussian_mixture_model()
        #  2) sample parameters  from the Gaussian Mixture Model based on constraints and make sure
        #     the resulting motion is valid
        parameters, min_error = self._sample_from_gmm_using_constraints(graph_node, gmm, mp_constraints, prev_frames, self.n_random_samples)

        write_message_to_log("Found best sample with distance: "+ str(min_error), LOG_MODE_DEBUG)
        return parameters

    def generate_random_sample(self, node_key, prev_mp_name="", prev_parameters=None):
        if self.use_transition_model and prev_parameters is not None and self._motion_state_graph.nodes[(self.prev_action_name, prev_mp_name)].has_transition_model(node_key):
            parameters = self._motion_state_graph.nodes[(self.prev_action_name, prev_mp_name)].predict_parameters(node_key, prev_parameters)
        else:
            parameters = self._motion_state_graph.nodes[node_key].sample_low_dimensional_vector()
        return parameters

    def _predict_gmm(self, mp_name, prev_mp_name, prev_parameters):
        to_key = (self.action_name, mp_name)
        gmm = self._motion_state_graph.nodes[(self.prev_action_name,prev_mp_name)].predict_gmm(to_key, prev_parameters)
        return gmm

    def _get_best_fit_sample_using_cluster_tree(self, graph_node, constraints, prev_frames, n_candidates=-1):
        """ Directed search in precomputed hierarchical space partitioning data structure
        """
        n_candidates = self.n_cluster_search_candidates if n_candidates < 1 else n_candidates
        data = graph_node, constraints, prev_frames
        distance, s = graph_node.search_best_sample(obj_spatial_error_sum, data, n_candidates)
        write_message_to_log("Found best sample with distance: " + str(distance), LOG_MODE_DEBUG)
        constraints.min_error = distance
        return np.array(s)

    def _sample_from_gmm_using_constraints(self, mp_node, gmm, constraints, prev_frames, n_samples):
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
        samples = gmm.sample(n_samples)[0]
        for idx, s_vector in enumerate(samples):
            object_function_params = mp_node, constraints, prev_frames
            error = obj_spatial_error_sum(s_vector, object_function_params)
            print("evaluated sample", idx, error)
            if min_error > error:
                min_error = error
                best_sample = s_vector
        constraints.min_error = min_error
        return best_sample, min_error

