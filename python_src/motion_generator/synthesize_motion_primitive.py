# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

@author: FARUPP,erhe01,mamauer,hadu01
"""
import time
import numpy as np
from utilities.gmm_math import mul
from utilities.evaluation_methods import check_sample_validity
from constrain_gmm import ConstrainedGMM
from optimize_motion_parameters import run_optimization
from constraint.constraint_extraction import get_step_length_for_sample
from constraint.constraint_check import obj_error_sum,evaluate_list_of_constraints,\
                            global_counter_dict

import copy
from constrain_gmm import ConstraintError
from utilities.exceptions import SynthesisError



def sample_from_gmm(graph_node,gmm, constraints, prev_frames, start_pose, skeleton,\
                        precision = {"pos":1,"rot":1,"smooth":1},num_samples=300,activate_parameter_check=False,verbose = False):

    """samples and picks the best samples out of a given set, quality measure
    is naturalness

    Parameters
    ----------
    * graph_node : GraphNode
    \t contains a motion primitive and meta information
    * gmm : sklearn.mixture.gmm
    \tThe gmm to sample
	* constraints : list of dict
	\t Each entry should contain joint position orientation and semanticAnnotation
     * transformation : dict
    \t Contains position as cartesian coordinates and orientation 
    \t as euler angles in degrees
    * num_samples : (optional) int
    \tThe number of samples to check
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
    max_bad_samples = 200
    samples = []
    distances = []
    successes = []
    tmp_bad_samples = 0
    count = 0
    while count < num_samples:

        if tmp_bad_samples>max_bad_samples:
                reached_max_bad_samples = True
                break
             
        s = np.ravel(gmm.sample())
        if activate_parameter_check:
            # using bounding box to check sample is good or bad
            valid = check_sample_validity(graph_node,s,skeleton) 
        else:
            valid = True
        if valid: 
#            tmp_bad_samples = 0
            samples.append(s)
            min_distance,successes = evaluate_list_of_constraints(graph_node.motion_primitive,s,constraints,prev_frames,start_pose,skeleton,
                                                        precision=precision,verbose=verbose)
            # check the root path for each sample, punish the curve walking
            acr_length = get_step_length_for_sample(graph_node.motion_primitive, 
                                                    s, 
                                                    method = "arc_length")
            absolute_length = get_step_length_for_sample(graph_node.motion_primitive,
                                                         s,
                                                         method="distance") 
            factor = acr_length/absolute_length   
            min_distance = factor * min_distance                              
            if verbose:
                print "sample no",count,"min distance",min_distance
            distances.append(min_distance)
            count+=1
        else:
           if verbose:
               print "sample failed validity check"
           tmp_bad_samples+=1
       
    if reached_max_bad_samples:
        raise RuntimeError("Failed to pick good sample from GMM")
        
    return samples, distances, successes

    


def constrain_primitive(mp_node,constraint, prev_frames,start_pose, skeleton,
                        firstFrame=None, lastFrame=None,
                        constrained_gmm_settings=None,verbose = False):
    """constrains a primitive with a given constraint

    Parameters
    ----------
    * mp_node : MotionPrimitiveNode
    \t\b

    * constraint : tuple
    \tof the shape (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
    * prev_frames : dict
    \t Used to estimate transformation of new samples 
    Returns
    -------
    * cgmm : ConstrainedGMM
    \tThe gmm of the motion_primitive constrained by the constraint
    """

    cgmm = ConstrainedGMM(mp_node,mp_node.motion_primitive.gmm, constraint=None, skeleton=skeleton, settings=constrained_gmm_settings, verbose=verbose)
    cgmm.set_constraint(constraint, prev_frames, start_pose,firstFrame=firstFrame,
                        lastFrame=lastFrame)
    return cgmm


def multiple_constrain_primitive(mp_node, constraints,prev_frames, start_pose, skeleton,
                                 constrained_gmm_settings=None, verbose = False):

    """constrains a primitive with all given constraints and yields one gmm
    Parameters
    ----------
    * mp_node : MotionPrimitiveNode
    \t\b

    * constraints : list of tuples
    \tof the shape (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
    * prev_frames : dict
    \t Used to estimate transformation of new samples 

    Returns
    -------
    * cgmm : ConstrainedGMM
    \tThe gmm of the motion_primitive constrained by the constraints
    """
    if verbose:
        print "generating gmm using",len(constraints),"constraints"
        start = time.clock()
    cgmms = []
    

    for i, constraint in enumerate(constraints):
        print "\t checking constraint %d" % i
        print constraint
        #constraint = (c['joint'], c['position'], c['orientation'])
        firstFrame = constraint['semanticAnnotation']['firstFrame']
        lastFrame = constraint['semanticAnnotation']['lastFrame']
        cgmms.append(constrain_primitive(mp_node, constraint, prev_frames, start_pose,
                                         skeleton,
                                         constrained_gmm_settings=constrained_gmm_settings,
                                         firstFrame=firstFrame,
                                         lastFrame=lastFrame,verbose=verbose))
    cgmm = cgmms[0]
    for k in xrange(1, len(cgmms)):
        cgmm = mul(cgmm, cgmms[k])
    if verbose:
        print "generated gmm in ",time.clock()-start,"seconds"
    return cgmm
    


def create_next_motion_distribution(prev_parameters, prev_primitive, mp_node,
                                    gpm, prev_frames,start_pose,skeleton=None, constraints=None,
                                    constrained_gmm_settings=None,verbose=False):
    """ creates the motion following the first_motion fulfilling the given
    constraints and multiplied by the output_gmm

    Parameters
    ----------
    * first_motion : numpy.ndarray
    \tThe s-vector of the first motion
    * first_primitive : MotionPrimitive object
    \tThe first primitive
    * second_primitive : MotionPrimitive object
    \tThe second primitive
    * second_gmm : sklearn.mixture.gmm
    * constraints : list of numpy.dicts
    \tThe constraints for the second motion
    * prev_frames : dict
    \t Used to estimate transformation of new samples 
    * gpm : GPMixture object
    \tThe GPM from the transition model for the transition\
    first_primitive_to_second_primitive

    Returns
    -------
    * predict_gmm : sklearn.mixture.gmm
    \tThe predicted and constrained new gmm multiplied with the output gmm

    """

    predict_gmm = gpm.predict(prev_parameters)
    if constraints:
        cgmm = multiple_constrain_primitive(mp_node,constraints,
                                            prev_frames,start_pose, skeleton, constrained_gmm_settings,verbose=verbose)

        constrained_predict_gmm = mul(predict_gmm, cgmm)
        return mul(constrained_predict_gmm, mp_node.motion_primitive.gmm)
    else:
        return mul(predict_gmm, mp_node.motion_primitive.gmm)



def manipulate_gmm(graph_node, pipeline_parameters):
    """ Restrict the gmm to samples that roughly fit the constraints and 
        multiply with a predicted GMM from the transition model.
    """
    
    morphable_graph,action_name,mp_name,constraints,\
    algorithm_config, prev_action_name, prev_mp_name, prev_frames, prev_parameters, \
    skeleton, \
    start_pose,verbose = pipeline_parameters
     
    # Perform manipulation based on settings and the current state.
    if algorithm_config["use_transition_model"] and prev_parameters is not None:

        transition_key = action_name +"_"+mp_name
        
        #only proceed the GMM prediction if the transition model was loaded
        if morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].has_transition_model(transition_key):
            gpm = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].outgoing_edges[transition_key].transition_model 
            prev_primitve = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].motion_primitive

            gmm = create_next_motion_distribution(prev_parameters, prev_primitve,\
                                                graph_node,\
                                                gpm, prev_frames, start_pose,\
                                                skeleton,\
                                                constraints,
                                                 algorithm_config["constrained_gmm_settings"],\
                                                verbose=verbose)

    else:
        gmm = multiple_constrain_primitive(graph_node,\
                                        constraints,\
                                        prev_frames,start_pose, \
                                        skeleton,\
                                        algorithm_config["constrained_gmm_settings"],verbose=verbose)   
    
    return gmm


class MotionPrimitiveGenerator(object):
    def __init__(self, action_constraints, algorithm_config, prev_action_name=""):
        self._action_constraints = action_constraints
        self._algorithm_config = algorithm_config
        self.action_name = action_constraints.action_name
        self.prev_action_name = prev_action_name
        self._morphable_graph = self._action_constraints.parent_constraint.morphable_graph
        self.skeleton = self._action_constraints.get_skeleton()
        self.verbose = self._algorithm_config["verbose"]
        self.precision = self._algorithm_config["constrained_gmm_settings"]["precision"]
        self.sample_size = self._algorithm_config["constrained_gmm_settings"]["sample_size"]
        self.use_constraints = self._algorithm_config["use_constraints"]
        self.use_optimization = self._algorithm_config["use_optimization"]
        self.use_constrained_gmm = self._algorithm_config["use_constrained_gmm"]
        self.use_transition_model = self._algorithm_config["use_transition_model"]
        self.activate_cluster_search = self._algorithm_config["activate_cluster_search"]
        self.activate_parameter_check = self._algorithm_config["activate_parameter_check"]
        
    def generate_motion_primitive_from_constraints(self, motion_primitive_constraints, prev_motion):
        """Calls get_optimal_parameters and backpoject the results.
        
        Parameters
        ----------
        *action_constraints: ActionConstraints
            Constraints specific for the elementary action.
        *motion_primitive_constraints: MotionPrimitiveConstraints
            Constraints specific for the current motion primitive.
        * algorithm_config : dict
            Contains parameters for the algorithm.
        *prev_motion: AnnotatedMotion
            Annotated motion with information on the graph walk.
            
        Returns
        -------
        * quat_frames : list of np.ndarray
            list of skeleton pose parameters.
        * parameters : np.ndarray
            low dimensional motion parameters used to generate the frames
        """
    
        try:
            mp_name = motion_primitive_constraints.motion_primitive_name
            
            algorithm_config_copy = copy.copy(self._algorithm_config)
            algorithm_config_copy["use_optimization"] = motion_primitive_constraints.use_optimization
    
            if len(prev_motion.graph_walk)> 0:
                prev_mp_name =  prev_motion.graph_walk[-1].motion_primitive_name
                prev_parameters =  prev_motion.graph_walk[-1].parameters
    
            else:
                prev_mp_name =  ""
                prev_parameters =  None
    
            parameters = self.get_optimal_parameters(mp_name,
                                                motion_primitive_constraints.constraints,
                                                prev_mp_name=prev_mp_name,
                                                prev_frames=prev_motion.quat_frames,
                                                prev_parameters=prev_parameters)
        except  ConstraintError as e:
            print "Exception",e.message
            raise SynthesisError(prev_motion.quat_frames,e.bad_samples)
            
        tmp_quat_frames = self._morphable_graph.subgraphs[self.action_name].nodes[mp_name].motion_primitive.back_project(parameters, use_time_parameters=True).get_motion_vector()
    
        return tmp_quat_frames, parameters


    def get_optimal_parameters(self, mp_name,
                                 constraints,
                                 prev_mp_name="", 
                                 prev_frames=None, 
                                 prev_parameters=None):
            """Uses the constraints to find the optimal paramaters for a motion primitive.
            Parameters
            ----------
            * morphable_graph : MorphableGraph
                Data structure containing the morphable models
            * action_name : string
                name of the elementary action and subgraph in morphable_graph
            * mp_name : string
                name of the motion primitive and node in the subgraph of the elemantary action
            * algorithm_config : dict
                Contains settings for the algorithm.
            * prev_frames: quaternion frames               
            Returns
            -------
            * parameters : np.ndarray
                Low dimensional parameters for the morphable model
            """
    

    
            if self.use_constraints and len(constraints) > 0: # estimate parameters fitting constraints
                graph_node = self._morphable_graph.subgraphs[self.action_name].nodes[mp_name]
                
                #  1) get gmm and modify it based on the current state and settings
                
                # Get prior gaussian mixture model from node
                graph_node = self._morphable_graph.subgraphs[self.action_name].nodes[mp_name]
                gmm = graph_node.motion_primitive.gmm
                if self.use_constrained_gmm:
                    pipeline_parameters = self._morphable_graph,self.action_name, mp_name,constraints,\
                                    self._algorithm_config, self.prev_action_name, \
                                 prev_mp_name, prev_frames, prev_parameters, \
                                 self.skeleton, \
                                 self._action_constraints.start_pose,self.verbose
      
                    gmm = manipulate_gmm(graph_node, pipeline_parameters)
                elif self.use_transition_model and prev_parameters is not None:
                    gmm = self._predict_gmm(mp_name, prev_mp_name, 
                                                     prev_frames, 
                                                     prev_parameters)
    
                #  2) sample parameters based on constraints and make sure
                #     the resulting motion is valid
                if self.activate_cluster_search:
                    #  find best sample using a directed search in a 
                    #  space partitioning data structure
                    parameters = self._search_for_best_sample(graph_node, constraints, prev_frames)
                    close_to_optimum = True
                else: 
                    # pick new random samples from the Gaussian Mixture Model
                    parameters,close_to_optimum = self.sample_and_pick_best(graph_node,gmm,\
                                                                        constraints,prev_frames)
                    
                #3) optimize sampled parameters as initial guess if the constraints were not reached
                if  not self.use_transition_model and self.use_optimization and not close_to_optimum:
                    bounding_boxes = (graph_node.parameter_bb, graph_node.cartesian_bb)
                    try:
                        initial_guess = parameters
                        parameters = run_optimization(graph_node.motion_primitive, gmm, constraints,
                                                        initial_guess, self.skeleton,
                                                        optimization_settings=self.optimization_settings, bounding_boxes=bounding_boxes,
                                                        prev_frames=prev_frames, start_pose=self._action_constraints.start_pose, 
                                                        verbose=self.verbose)
                    except ValueError as e:
                        print e.message
                        parameters = initial_guess
     
    
            else: # generate random parameters
                parameters = self._get_random_parameters(mp_name, prev_mp_name, 
                                                         prev_frames, 
                                                         prev_parameters)
             
            return parameters
            
            
    def _search_for_best_sample(self, graph_node,constraints,prev_frames):
    
        """ Directed search in precomputed hierarchical space partitioning data structure
        """
        data = graph_node.motion_primitive, constraints, prev_frames,self._action_constraints.start_pose, self.skeleton, self.precision
        distance, s = graph_node.search_best_sample(obj_error_sum,data)
        print "found best sample with distance:",distance
        global_counter_dict["motionPrimitveErrors"].append(distance)
        return np.array(s)
                                         

    def sample_and_pick_best(self, graph_node,gmm, constraints, prev_frames):
        """samples and picks the best sample out of a given set, quality measures
        is naturalness
    
        Parameters
        ----------
        * graph_node : GraphNode
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
        samples, distances, successes =  sample_from_gmm(graph_node,gmm, constraints,prev_frames,self._action_constraints.start_pose, self.skeleton,\
                                precision=self.precision,num_samples=self.sample_size,activate_parameter_check=self.activate_parameter_check,verbose=self.verbose)
       
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
        
        

    def _predict_gmm(self, mp_name, prev_mp_name, prev_frames, prev_parameters):
        to_key = self.action_name + "_" + mp_name
        gmm = self._morphable_graph.subgraphs[self.prev_action_name].nodes[prev_mp_name].predict_gmm(to_key,prev_parameters)
        return gmm
    
    
    def _get_random_parameters(self, mp_name, prev_mp_name="", 
                                 prev_frames=None, 
                                 prev_parameters=None):
      
        if self._algorithm_config["use_transition_model"] and prev_parameters is not None and self._morphable_graph.subgraphs[self.prev_action_name].nodes[prev_mp_name].has_transition_model(to_key):
             to_key = self.action_name+"_"+mp_name
             parameters = self._morphable_graph.subgraphs[self.prev_action_name].nodes[prev_mp_name].predict_parameters(to_key,prev_parameters)
        else:
            parameters = self._morphable_graph.subgraphs[self.action_name].nodes[mp_name].sample_parameters()
        return parameters
        

        