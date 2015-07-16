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
from optimize_motion_parameters import run_optimization,\
                    generate_optimization_settings
from constraint.constraint_extraction import get_step_length_for_sample
from constraint.constraint_check import obj_error_sum,evaluate_list_of_constraints,\
                            global_counter_dict

import copy
from constrain_gmm import ConstraintError
from utilities.exceptions import SynthesisError


          
def generate_algorithm_settings(use_constraints=True,
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
                            verbose=False):               
    """Should be used to generate a dict containing all settings for the algorithm
    Returns
    ------
    algorithm_config : dict
      Settings that can be passed as parameter to the pipeline used in get_optimal_parameters
    
    """
    optimization_settings = generate_optimization_settings(method=optimization_method,max_iterations=max_optimization_iterations,
                                                              quality_scale_factor=optimization_quality_scale_factor ,
                                                              error_scale_factor = optimization_error_scale_factor,
                                                              tolerance=optimization_tolerance,
                                                              kinematic_epsilon=optimization_kinematic_epsilon)
    constrained_gmm_settings ={"sample_size" : sample_size,
                               "precision" : {"pos" : constrained_gmm_pos_precision,"rot" : constrained_gmm_rot_precision,"smooth":constrained_gmm_smooth_precision},
                               "strict" : strict_constrained_gmm,
                               "max_bad_samples":constrained_gmm_max_bad_samples}
    trajectory_following_settings = {"method" : trajectory_extraction_method,
                                     "step_length_factor" : trajectory_step_length_factor,
                                     "use_position_constraints":trajectory_use_position_constraints, 
                                     "use_dir_vector_constraints"  : trajectory_use_dir_vector_constraints,
                                     "use_frame_constraints":trajectory_use_frame_constraints
                                     
                                     }
    
    algorithm_config = {"use_constraints":use_constraints,
           "use_optimization":use_optimization,
           "use_constrained_gmm" : use_constrained_gmm,
           "use_transition_model":use_transition_model,
           "activate_parameter_check":activate_parameter_check,
           "apply_smoothing":apply_smoothing,
           "optimization_settings": optimization_settings,
           "constrained_gmm_settings":constrained_gmm_settings,
           "trajectory_following_settings" : trajectory_following_settings,
           "activate_cluster_search" : activate_cluster_search,
           "verbose": verbose
            }
    return algorithm_config
                
                



def constrain_primitive(motion_primitive, gmm,constraint, prev_frames,start_pose, skeleton,
                        firstFrame=None, lastFrame=None, size=300,
                        constrained_gmm_settings=None,verbose = False):
    """constrains a primitive with a given constraint

    Parameters
    ----------
    * motion_primitive : MotionPrimitive
    \t\b
     * gmm : sklearn.mixture.gmm
    \t the GMM to constrain
    * constraint : tuple
    \tof the shape (joint, [pos_x, pos_y, pos_z], [rot_x, rot_y, rot_z])
    * prev_frames : dict
    \t Used to estimate transformation of new samples 
    Returns
    -------
    * cgmm : ConstrainedGMM
    \tThe gmm of the motion_primitive constrained by the constraint
    """

    cgmm = ConstrainedGMM(motion_primitive,gmm, constraint=None, skeleton=skeleton, settings=constrained_gmm_settings, verbose=verbose)
    cgmm.set_constraint(constraint, prev_frames, start_pose,size=size,firstFrame=firstFrame,
                        lastFrame=lastFrame)
    return cgmm


def multiple_constrain_primitive(motion_primitive,gmm, constraints,prev_frames, start_pose, skeleton,
                                 size=300, constrained_gmm_settings=None, verbose = False):

    """constrains a primitive with all given constraints and yields one gmm
    Parameters
    ----------
    * motion_primitive : MotionPrimitive
    \t\b
    * gmm : sklearn.mixture.gmm
    \t the GMM to constrain
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
    

    for i, c in enumerate(constraints):
        print "\t checking constraint %d" % i
        print c
        #constraint = (c['joint'], c['position'], c['orientation'])
        firstFrame = c['semanticAnnotation']['firstFrame']
        lastFrame = c['semanticAnnotation']['lastFrame']
        cgmms.append(constrain_primitive(motion_primitive, gmm,c, prev_frames, start_pose,
                                         skeleton,size=size,
                                         constrained_gmm_settings=constrained_gmm_settings,
                                         firstFrame=firstFrame,
                                         lastFrame=lastFrame,verbose=verbose))
    cgmm = cgmms[0]
    for k in xrange(1, len(cgmms)):
        cgmm = mul(cgmm, cgmms[k])
    if verbose:
        print "generated gmm in ",time.clock()-start,"seconds"
    return cgmm
    


def create_next_motion_distribution(first_s, first_primitive, second_primitive,second_gmm,
                                    gpm, prev_frames,start_pose,skeleton=None, constraints=None,
                                    size=300, precision={"pos":1,"rot":1,"smooth":1},verbose=False):
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

    predict_gmm = gpm.predict(first_s)
    if constraints:
        cgmm = multiple_constrain_primitive(second_primitive, second_gmm,constraints,
                                            prev_frames,start_pose, skeleton, size, precision,verbose=verbose)

        constrained_predict_gmm = mul(predict_gmm, cgmm)
        return mul(constrained_predict_gmm, second_gmm)
    else:
        return mul(predict_gmm, second_gmm)




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

def sample_and_pick_best(graph_node,gmm, constraints, prev_frames, start_pose, skeleton,\
                        precision = {"pos":1,"rot":1,"smooth":1},num_samples=300,activate_parameter_check=False,verbose = False):
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
     * transformation : dict
    \t Contains position as cartesian coordinates and orientation 
    \t as euler angles in degrees
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
    samples, distances, successes =  sample_from_gmm(graph_node,gmm, constraints,prev_frames,start_pose, skeleton,\
                            precision=precision,num_samples=num_samples,activate_parameter_check=activate_parameter_check,verbose = verbose)
   
    best_index = distances.index(min(distances))
    best_sample = samples[best_index]
    if np.any(successes):
        success = True
    else: ######################################################################################################
        success = False
        
    if verbose:
        if success:
            print "reached constraint exactly",distances[best_index]
        else:
            print "failed to reach constraint exactly return closest sample",distances[best_index]
    print "found best sample with distance:",distances[best_index]
    return best_sample,success
    
    

def search_for_best_sample(graph_node,constraints,prev_frames,start_pose, skeleton,\
                        precision = {"pos":1,"rot":1,"smooth":1},verbose=False):

    """ Directed search in precomputed hierarchical space partitioning data structure
    """
    data = graph_node.motion_primitive, constraints, prev_frames,start_pose, skeleton, precision
    distance, s = graph_node.search_best_sample(obj_error_sum,data)
    print "found best sample with distance:",distance
    global_counter_dict["motionPrimitveErrors"].append(distance)
    return np.array(s)
                                     


def extract_gmm_from_motion_primitive(pipeline_parameters):
    """ Restrict the gmm to samples that roughly fit the constraints and 
        multiply with a predicted GMM from the transition model.
    """
    
    morphable_graph,action_name,mp_name,constraints,\
    algorithm_config, prev_action_name, prev_mp_name, prev_frames, prev_parameters, \
    skeleton, \
    start_pose,verbose = pipeline_parameters
     
    sample_size = algorithm_config["constrained_gmm_settings"]["sample_size"]
    # Get prior gaussian mixture model from node
    graph_node = morphable_graph.subgraphs[action_name].nodes[mp_name]
    gmm = graph_node.motion_primitive.gmm
    # Perform manipulation based on settings and the current state.
    if algorithm_config["use_transition_model"] and prev_parameters is not None:
        if algorithm_config["use_constrained_gmm"]:
            transition_key = action_name +"_"+mp_name
            
            #only proceed the GMM prediction if the transition model was loaded
            if morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].has_transition_model(transition_key):
                gpm = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].outgoing_edges[transition_key].transition_model 
                prev_primitve = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].motion_primitive
    
                gmm = create_next_motion_distribution(prev_parameters, prev_primitve,\
                                                    graph_node,gmm,\
                                                    gpm, prev_frames, start_pose,\
                                                    skeleton,\
                                                    constraints,sample_size,\
                                                     algorithm_config["constrained_gmm_settings"],\
                                                    verbose=verbose)
                                                    
        else:
            to_key = action_name+"_"+mp_name
            gmm = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].predict_gmm(to_key,prev_parameters)
    elif algorithm_config["use_constrained_gmm"]:
        gmm = multiple_constrain_primitive(graph_node,gmm,\
                                        constraints,\
                                        prev_frames,start_pose, \
                                        skeleton,\
                                        sample_size, algorithm_config["constrained_gmm_settings"],verbose=verbose)   

    return gmm

def get_random_parameters(pipeline_parameters):
    morphable_graph,action_name,mp_name,constraints,\
    algorithm_config, prev_action_name, prev_mp_name, prev_frames, prev_parameters, \
    bvh_reader, node_name_map, \
    start_pose,verbose = pipeline_parameters
    if algorithm_config["use_transition_model"] and prev_parameters is not None and morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].has_transition_model(to_key):
         to_key = action_name+"_"+mp_name
         parameters = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].predict_parameters(to_key,prev_parameters)
    else:
        parameters = morphable_graph.subgraphs[action_name].nodes[mp_name].sample_parameters()
    return parameters
    

def get_optimal_parameters(morphable_graph,action_name,mp_name,constraints,\
                         algorithm_config, prev_action_name="", prev_mp_name="", prev_frames=None, prev_parameters=None, \
                         skeleton=None, \
                         start_pose=None):
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
            Contains algorithm_config for the algorithm.
            When set to None generate_algorithm_settings() is called with default settings
            use_constraints: Sets whether or not to use constraints 
            use_optimization : Sets whether to activate optimization or use only sampling
            use_constrained_gmm : Sets whether or not to constrain the GMM
            use_transition_model : Sets whether or not to predict parameters using the transition model
            apply_smoothing : Sets whether or not smoothing is applied on transitions
            optimization_settings : parameters for the optimization algorithm: method, max_iterations 
            constrained_gmm_settings : position and orientation precision + sample size 
        * prev_frames: quaternion frames               
        Returns
        -------
        * parameters : np.ndarray
            Low dimensional parameters for the morphable model
        """

        pipeline_parameters = morphable_graph,action_name,mp_name,constraints,\
                         algorithm_config, prev_action_name, prev_mp_name, prev_frames, prev_parameters, \
                         skeleton, \
                         start_pose,algorithm_config["verbose"]
        sample_size = algorithm_config["constrained_gmm_settings"]["sample_size"]
        precision = algorithm_config["constrained_gmm_settings"]["precision"]

        if algorithm_config["use_constraints"] and len(constraints) > 0: # estimate parameters fitting constraints
            graph_node = morphable_graph.subgraphs[action_name].nodes[mp_name]
            
            #  1) get gmm and modify it based on the current state and settings
           
            gmm = extract_gmm_from_motion_primitive(pipeline_parameters)

            #  2) sample parameters based on constraints and make sure
            #     the resulting motion is valid
            if algorithm_config["activate_cluster_search"]:
                #  find best sample using a directed search in a 
                #  space partitioning data structure
                parameters = search_for_best_sample(graph_node,constraints,prev_frames,start_pose,\
                                        skeleton,\
                                         verbose=algorithm_config["verbose"])
                close_to_optimum = True
            else: 
                # pick new random samples from the Gaussian Mixture Model
                parameters,close_to_optimum = sample_and_pick_best(graph_node,gmm,\
                                        constraints,prev_frames,start_pose,\
                                        skeleton,\
                                        precision= precision,\
                                        num_samples=sample_size,\
                                        activate_parameter_check=algorithm_config["activate_parameter_check"],verbose=algorithm_config["verbose"])
                
            #3) optimize sampled parameters as initial guess if the constraints were not reached
            if  not algorithm_config["use_transition_model"] and algorithm_config["use_optimization"] and not close_to_optimum:
                bounding_boxes = (graph_node.parameter_bb, graph_node.cartesian_bb)
                try:
                    initial_guess = parameters
                    parameters = run_optimization(graph_node.motion_primitive, gmm, constraints,
                                                    initial_guess, skeleton,
                                                    optimization_settings=algorithm_config["optimization_settings"], bounding_boxes=bounding_boxes,
                                                    prev_frames=prev_frames, start_pose=start_pose, verbose=algorithm_config["verbose"])
                except ValueError as e:
                    print e.message
                    parameters = initial_guess
 

        else: # generate random parameters
            parameters = get_random_parameters(pipeline_parameters)
         
        return parameters
        
def get_optimal_motion(action_constraints, motion_primitive_constraints,
                       algorithm_config, prev_motion):
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
        action_name = action_constraints.action_name
        skeleton = action_constraints.get_skeleton()
        algorithm_config_copy = copy.copy(algorithm_config)
        algorithm_config_copy["use_optimization"] = motion_primitive_constraints.use_optimization

        if len(prev_motion.graph_walk)> 0:
            prev_action_name = prev_motion.graph_walk[-1].action_name
            prev_mp_name =  prev_motion.graph_walk[-1].motion_primitive_name
            prev_parameters =  prev_motion.graph_walk[-1].parameters

        else:
            prev_action_name = ""
            prev_mp_name =  ""
            prev_parameters =  None

        parameters = get_optimal_parameters(action_constraints.parent_constraint.morphable_graph,
                                            action_name,
                                            mp_name,
                                            motion_primitive_constraints.constraints,
                                            algorithm_config=algorithm_config_copy,
                                            prev_action_name=prev_action_name,
                                            prev_mp_name=prev_mp_name,
                                            prev_frames=prev_motion.quat_frames,
                                            prev_parameters=prev_parameters,
                                            skeleton=skeleton,
                                            start_pose=action_constraints.start_pose)
    except  ConstraintError as e:
        print "Exception",e.message
        raise SynthesisError(prev_motion.quat_frames,e.bad_samples)
        
    tmp_quat_frames = action_constraints.parent_constraint.morphable_graph.subgraphs[action_name].nodes[mp_name].motion_primitive.back_project(parameters, use_time_parameters=True).get_motion_vector()

    return tmp_quat_frames, parameters

