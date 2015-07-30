# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

@author: erhe01,mamauer,hadu01,FARUPP
"""

import numpy as np
from animation_data.evaluation_methods import check_sample_validity
from statistics.constrained_gmm_builder import ConstrainedGMMBuilder
from utilities.exceptions import ConstraintError, SynthesisError
from optimize_motion_parameters import run_optimization
from constraint.constraint_check import obj_error_sum,evaluate_list_of_constraints,\
                                        global_counter_dict


class MotionPrimitiveGenerator(object):
    """
    Parameters
    ----------
    * action_constraints : ElementaryActionConstraints
        contains reference to motion data structure
    * algorithm_config : dict
        Contains settings for the algorithm.
    * prev_action_name : string
        name of the action before the current action in the elementary action list
    """
    def __init__(self, action_constraints, algorithm_config, prev_action_name=""):
        self._action_constraints = action_constraints
        self._algorithm_config = algorithm_config
        self.action_name = action_constraints.action_name
        self.prev_action_name = prev_action_name
        self._morphable_graph = self._action_constraints.parent_constraint.morphable_graph
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
                                                motion_primitive_constraints.constraints,
                                                prev_mp_name=prev_mp_name,
                                                prev_frames=prev_motion.quat_frames,
                                                prev_parameters=prev_parameters, use_optimization=use_optimization)
        except  ConstraintError as e:
            print "Exception",e.message
            raise SynthesisError(prev_motion.quat_frames,e.bad_samples)
            
        tmp_quat_frames = self._morphable_graph.subgraphs[self.action_name].nodes[mp_name].motion_primitive.back_project(parameters, use_time_parameters=True).get_motion_vector()
    
        return tmp_quat_frames, parameters


    def get_optimal_motion_primitive_parameters(self, mp_name,
                                 constraints,
                                 prev_mp_name="", 
                                 prev_frames=None, 
                                 prev_parameters=None,
                                 use_optimization=False):
        """Uses the constraints to find the optimal paramaters for a motion primitive.
        Parameters
        ----------
        * mp_name : string
            name of the motion primitive and node in the subgraph of the elemantary action
        * constraints: list
            list of dict with position constraints for joints
        * prev_frames: np.ndarry
            quaternion frames               
        * use_optimization: boolean
            Activates or deactivates optimization (Not all motion primitives need optimization).
        Returns
        -------
        * parameters : np.ndarray
            Low dimensional parameters for the morphable model
        """



        if self.use_constraints and len(constraints) > 0: # estimate parameters fitting constraints
            graph_node = self._morphable_graph.subgraphs[self.action_name].nodes[mp_name]
            
            # A) find best sample from model
            if self.activate_cluster_search and graph_node.cluster_tree is not None:
                #  find best sample using a directed search in a 
                #  space partitioning data structure
                parameters = self._search_for_best_sample(graph_node, constraints, prev_frames, self._algorithm_config["n_cluster_search_candidates"])
                close_to_optimum = True
            else: 
                #  1) get gmm and modify it based on the current state and settings
                # Get prior gaussian mixture model from node
                
                gmm = graph_node.motion_primitive.gmm
                if self._constrained_gmm_builder is not None:
                    gmm = self._constrained_gmm_builder.build(self.action_name, mp_name, constraints,\
                                                self.prev_action_name, prev_mp_name, prev_frames, prev_parameters)
                                                
                elif self.use_transition_model and prev_parameters is not None:
                    gmm = self._predict_gmm(mp_name, prev_mp_name, 
                                                     prev_frames, 
                                                     prev_parameters)
    
                #  2) sample parameters  from the Gaussian Mixture Model based on constraints and make sure
                #     the resulting motion is valid                
                parameters,close_to_optimum = self._sample_and_pick_best(graph_node,gmm,\
                                                                    constraints,prev_frames)
                
            # B) optimize sampled parameters as initial guess if the constraints were not reached
            if  not self.use_transition_model and use_optimization and not close_to_optimum:
                bounding_boxes = (graph_node.parameter_bb, graph_node.cartesian_bb)
               
                initial_guess = parameters
                parameters = run_optimization(graph_node.motion_primitive, gmm, constraints,
                                                initial_guess, self.skeleton,
                                                self._optimization_settings, bounding_boxes=bounding_boxes,
                                                prev_frames=prev_frames, start_pose=self._action_constraints.start_pose, 
                                                verbose=self.verbose)


        else: # generate random parameters
            parameters = self._get_random_parameters(mp_name, prev_mp_name, 
                                                     prev_frames, 
                                                     prev_parameters)
         
        return parameters
            
            
    def _search_for_best_sample(self, graph_node, constraints, prev_frames, n_candidates=2):
    
        """ Directed search in precomputed hierarchical space partitioning data structure
        """
        data = graph_node.motion_primitive, constraints, prev_frames,self._action_constraints.start_pose, self.skeleton, self.precision
        distance, s = graph_node.search_best_sample(obj_error_sum, data, n_candidates)
        print "found best sample with distance:",distance
        global_counter_dict["motionPrimitveErrors"].append(distance)
        return np.array(s)
    
    def _get_random_parameters(self, mp_name, prev_mp_name="", 
                                 prev_frames=None, 
                                 prev_parameters=None):
        
        to_key = self.action_name+"_"+mp_name
        if self.use_transition_model and prev_parameters is not None and self._morphable_graph.subgraphs[self.prev_action_name].nodes[prev_mp_name].has_transition_model(to_key):
             
             parameters = self._morphable_graph.subgraphs[self.prev_action_name].nodes[prev_mp_name].predict_parameters(to_key,prev_parameters)
        else:
            parameters = self._morphable_graph.subgraphs[self.action_name].nodes[mp_name].sample_parameters()
        return parameters
        


    def _predict_gmm(self, mp_name, prev_mp_name, prev_frames, prev_parameters):
        to_key = self.action_name + "_" + mp_name
        gmm = self._morphable_graph.subgraphs[self.prev_action_name].nodes[prev_mp_name].predict_gmm(to_key,prev_parameters)
        return gmm
    
                                          

    def _sample_and_pick_best(self, mp_node, gmm, constraints, prev_frames):
        """samples and picks the best sample out of a given set, quality measures
        is naturalness
    
        Parameters
        ----------
        * mp_node : MotionPrimitiveNode
        \t contains a motion primitive and meta information
        * gmm : sklearn.mixture.gmm
        \tThe gmm to sample
    	* constraints : list of dict
    	\t Each entry should contain joint position orientation and semanticAnnotation
        * num_samples : (optional) int
        \tThe number of samples to check
        * prev_frames: quaternion frames
        Returns
        -------
        * sample : numpy.ndarray
        \tThe best sample out of those which have been created
        * success : bool
        \t the constraints were reached exactly
        """
        samples, distances, successes =  self._sample_from_gmm(mp_node, gmm, constraints,prev_frames)
       
        best_index = distances.index(min(distances))
        best_sample = samples[best_index]
        if np.any(successes):
            success = True
        else: ######################################################################################################
            success = False
            
        if self.verbose:
            if success:
                print "reached constraint exactly",distances[best_index]
            else:
                print "failed to reach constraint exactly return closest sample",distances[best_index]
        print "found best sample with distance:",distances[best_index]
        return best_sample,success
        
        

    
    def _sample_from_gmm(self, mp_node, gmm, constraints, prev_frames):
    
        """samples and picks the best samples out of a given set, quality measure
        is naturalness
    
        Parameters
        ----------
        * mp_node : MotionPrimitiveNode
        \t contains a motion primitive and meta information
        * gmm : sklearn.mixture.gmm
        \tThe gmm to sample
    	* constraints : list of dict
    	\t Each entry should contain joint position orientation and semanticAnnotation
        * prev_frames: list
        \tA list of quaternion frames
        Returns
        -------
        * samples : list of numpy.ndarray
        \tThe samples out of those which have been created
        * distances : list of float
        \t distance of the samples to the constraint
        * successes : list of bool 
         \t wether or not  the samples is meeting the desired precision
       
        """
        reached_max_bad_samples = False
        samples = []
        distances = []
        successes = []
        tmp_bad_samples = 0
        count = 0
        while count < self.n_random_samples:
    
            if tmp_bad_samples>self.max_bad_samples:
                    reached_max_bad_samples = True
                   
                 
            s = np.ravel(gmm.sample())
            if self.activate_parameter_check:
                # using bounding box to check sample is good or bad
                valid = check_sample_validity(mp_node,s,self.skeleton) 
            else:
                valid = True
            if valid: 
    #            tmp_bad_samples = 0
                samples.append(s)
                min_distance,successes = evaluate_list_of_constraints(mp_node.motion_primitive,s,constraints,prev_frames,self._action_constraints.start_pose,self.skeleton,
                                                            precision=self.precision,verbose=self.verbose)
                # check the root path for each sample, punish the curve walking
                acr_length = mp_node.get_step_length_for_sample(s, method="arc_length")
                absolute_length = mp_node.get_step_length_for_sample(s, method="distance")
                factor = acr_length/absolute_length   
                min_distance = factor * min_distance                              
                if self.verbose:
                    print "sample no",count,"min distance",min_distance
                distances.append(min_distance)
                count+=1
            else:
               if self.verbose:
                   print "sample failed validity check"
               tmp_bad_samples+=1
           
        if reached_max_bad_samples:
            print "Warning: Failed to pick good sample from GMM"

        return samples, distances, successes
    
            