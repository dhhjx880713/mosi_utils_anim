# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

@author: erhe01,mamauer,hadu01,FARUPP
"""

import numpy as np
from animation_data.evaluation_methods import check_sample_validity
from statistics.constrained_gmm_builder import ConstrainedGMMBuilder
from utilities.exceptions import ConstraintError, SynthesisError
from numerical_minimizer import NumericalMinimizer
from . import global_counter_dict
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
        self._morphable_graph = self._action_constraints.morphable_graph
        self.skeleton = self._action_constraints.get_skeleton()
        self._constrained_gmm_config = self._algorithm_config["constrained_gmm_settings"]
        self._optimization_settings = self._algorithm_config["optimization_settings"]
        self.n_random_samples = self._algorithm_config["n_random_samples"]
        self.verbose = self._algorithm_config["verbose"]
        self.precision = self._constrained_gmm_config["precision"]        
        self.max_bad_samples = self._constrained_gmm_config["max_bad_samples"]
        self.use_constraints = self._algorithm_config["use_constraints"]
        self.use_optimization = self._algorithm_config["use_optimization"]
        self.use_constrained_gmm = self._algorithm_config["use_constrained_gmm"]
        self.use_transition_model = self._algorithm_config["use_transition_model"]
        self.activate_cluster_search = self._algorithm_config["activate_cluster_search"]
        self.activate_parameter_check = self._algorithm_config["activate_parameter_check"]
        if self.use_constrained_gmm:
            self._constrained_gmm_builder = ConstrainedGMMBuilder(self._morphable_graph, self._algorithm_config, 
                                                                    self._action_constraints.start_pose, self.skeleton)
        else:
            self._constrained_gmm_builder = None
        self.numerical_minimizer = NumericalMinimizer(self._algorithm_config, self.skeleton, action_constraints.start_pose)
        self.numerical_minimizer.set_objective_function(obj_spatial_error_sum_and_naturalness)
        
        
    def generate_motion_primitive_from_constraints(self, motion_primitive_constraints, prev_motion):
        """Calls get_optimal_parameters and backpojects the results.
        
        Parameters
        ----------
        *motion_primitive_constraints: MotionPrimitiveConstraints
            Constraints specific for the current motion primitive.
        *prev_motion: AnnotatedMotion
            Annotated motion with information on the graph walk.
            
        Returns
        -------
        * quat_frames : list of np.ndarray
            list of skeleton pose parameters.
        * parameters : np.ndarray
            low dimensional motion parameters used to generate the frames
        """
    
 
        mp_name = motion_primitive_constraints.motion_primitive_name
  
        if len(prev_motion.graph_walk)> 0:
            prev_mp_name =  prev_motion.graph_walk[-1].motion_primitive_name
            prev_parameters =  prev_motion.graph_walk[-1].parameters

        else:
            prev_mp_name =  ""
            prev_parameters =  None
 
        use_optimization= self.use_optimization or motion_primitive_constraints.use_optimization
        motion_primitive_constraints.use_optimization   
        
        try:
            parameters = self.get_optimal_motion_primitive_parameters(mp_name,
                                                motion_primitive_constraints,
                                                prev_mp_name=prev_mp_name,
                                                prev_frames=prev_motion.quat_frames,
                                                prev_parameters=prev_parameters, use_optimization=use_optimization)
        except  ConstraintError as e:
            print "Exception",e.message
            raise SynthesisError(prev_motion.quat_frames,e.bad_samples)
            
        tmp_quat_frames = self._morphable_graph.nodes[(self.action_name,mp_name)].motion_primitive.back_project(parameters, use_time_parameters=True).get_motion_vector()
    
        return tmp_quat_frames, parameters


    def get_optimal_motion_primitive_parameters(self, mp_name,
                                 motion_primitive_constraints,
                                 prev_mp_name="", 
                                 prev_frames=None, 
                                 prev_parameters=None,
                                 use_optimization=False):
        """Uses the constraints to find the optimal paramaters for a motion primitive.
        Parameters
        ----------
        * mp_name : string
            name of the motion primitive and node in the subgraph of the elemantary action
        * motion_primitive_constraints: MotionPrimitiveConstraints
            contains a list of dict with constraints for joints
        * prev_frames: np.ndarry
            quaternion frames               
        * use_optimization: boolean
            Activates or deactivates optimization (Not all motion primitives need optimization).
        Returns
        -------
        * parameters : np.ndarray
            Low dimensional parameters for the morphable model
        """



        if self.use_constraints and len(motion_primitive_constraints.constraints) > 0:

            graph_node = self._morphable_graph.nodes[(self.action_name, mp_name)]
            gmm = graph_node.motion_primitive.gmm

            if self.activate_cluster_search and graph_node.cluster_tree is not None:
                #  find best sample using a directed search in a 
                #  space partitioning data structure
                parameters = self._search_for_best_sample_in_cluster_tree(graph_node, motion_primitive_constraints, prev_frames, self._algorithm_config["n_cluster_search_candidates"])
                close_to_optimum = True
            else: 
                #  1) get gmm and modify it based on the current state and settings
                # Get prior gaussian mixture model from node
                
                if self._constrained_gmm_builder is not None:
                    gmm = self._constrained_gmm_builder.build(self.action_name, mp_name, motion_primitive_constraints,
                                                              self.prev_action_name, prev_mp_name, 
                                                              prev_frames, prev_parameters)
                                                
                elif self.use_transition_model and prev_parameters is not None:
                    gmm = self._predict_gmm(mp_name, prev_mp_name, 
                                                     prev_frames, 
                                                     prev_parameters)
    
                #  2) sample parameters  from the Gaussian Mixture Model based on constraints and make sure
                #     the resulting motion is valid                
                parameters, min_error = self._pick_best_random_sample(graph_node,gmm,
                                                                    motion_primitive_constraints,prev_frames)
                close_to_optimum = True
                

            if  not self.use_transition_model and use_optimization and not close_to_optimum:

                data =  graph_node.motion_primitive, motion_primitive_constraints, \
                       prev_frames, self._optimization_settings["error_scale_factor"], \
                       self._optimization_settings["quality_scale_factor"]

                self.numerical_minimizer.set_objective_function_parameters(data)
                parameters = self.numerical_minimizer.run(initial_guess=parameters)


        else: # no constraints were given
            print "motion primitive", mp_name
            parameters = self._get_random_parameters(mp_name, prev_mp_name, 
                                                     prev_frames, 
                                                     prev_parameters)
         
        return parameters
            
            

    
    def _get_random_parameters(self, mp_name, prev_mp_name="", 
                                 prev_frames=None, 
                                 prev_parameters=None):
        
        to_key = self.action_name+"_"+mp_name
        if self.use_transition_model and prev_parameters is not None and self._morphable_graph.nodes[(self.prev_action_name,prev_mp_name)].has_transition_model(to_key):
             
            parameters = self._morphable_graph.nodes[(self.prev_action_name,prev_mp_name)].predict_parameters(to_key,prev_parameters)
        else:
            parameters = self._morphable_graph.nodes[(self.action_name,mp_name)].sample_parameters()
        return parameters
        


    def _predict_gmm(self, mp_name, prev_mp_name, prev_frames, prev_parameters):
        to_key = (self.action_name, mp_name)
        gmm = self._morphable_graph.nodes[(self.prev_action_name,prev_mp_name)].predict_gmm(to_key,prev_parameters)
        return gmm
    
    def _search_for_best_sample_in_cluster_tree(self, graph_node, constraints, prev_frames, n_candidates=2):
        """ Directed search in precomputed hierarchical space partitioning data structure
        """
        data = graph_node.motion_primitive, constraints, prev_frames
        distance, s = graph_node.search_best_sample(obj_spatial_error_sum, data, n_candidates)
        print "found best sample with distance:",distance
        global_counter_dict["motionPrimitveErrors"].append(distance)
        return np.array(s)                                 

    def _pick_best_random_sample(self, mp_node, gmm, constraints, prev_frames):
        """samples and picks the best samples out of a given set, quality measure
        is naturalness
    
        Parameters
        ----------
        * mp_node : MotionPrimitiveNode
            contains a motion primitive and meta information
        * gmm : sklearn.mixture.gmm
            The gmm to sample
        * motion_primitive_constraints: MotionPrimitiveConstraints
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
    
            if tmp_bad_samples>self.max_bad_samples:
                    reached_max_bad_samples = True
                   
                 
            s = np.ravel(gmm.sample())
            if self.activate_parameter_check:
                # using bounding box to check sample is good or bad
                valid = check_sample_validity(mp_node, s, self.skeleton) 
            else:
                valid = True
            if valid:                 
                object_function_params = mp_node.motion_primitive, constraints, prev_frames
                error = obj_spatial_error_sum(s,object_function_params)
                if min_error > error:
                    min_error = error
                    best_sample = s
                count+=1
            else:
               if self.verbose:
                   print "sample failed validity check"
               tmp_bad_samples+=1
           
        if reached_max_bad_samples:
            print "Warning: Failed to pick good sample from GMM"
            return best_sample, min_error
        
        print "found best sample with distance:",min_error
        global_counter_dict["motionPrimitveErrors"].append(min_error)
        return best_sample, min_error
    
            