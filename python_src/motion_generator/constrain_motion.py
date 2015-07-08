# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:15:22 2015

@author: FARUPP,erhe01,mamauer,hadu01
"""
import time
import numpy as np
import sklearn.mixture as mixture
from lib.gmm_math import mul
from lib.evaluation_methods import check_sample_validity
from constrain_gmm import ConstrainedGMM
from lib import GP_predict
from optimize_motion import run_optimization,\
                    generate_optimization_settings
from lib.graph_walk_extraction import get_step_length_for_sample
from lib.constraint import obj_error_sum,evaluate_list_of_constraints,\
                            global_counter_dict

def check_constrained_hands(constraints):
    """ Returns a string describing which hands are used
    """
    constrained_hands = "None"
    for c in constraints:
        if c["joint"] in ["leftHand", "righHand"]:
            if constrained_hands == "None":
                constrained_hands = c["joint"]
            else:
                constrained_hands = "two_hands"
    return constrained_hands
    
def extract_clusters_from_gmm(gmm,clusters):
    """Returns a GMM with a reduced set of clusters
    """

    means_ = [gmm.means_[i] for i in clusters]
    covars_= [gmm.covars_[i] for i in clusters]
    weights_ = [gmm.weights_[i] for i in clusters]

    new_gmm = mixture.GMM(len(weights_), covariance_type='full')
    new_gmm.weights_ = np.array(weights_)
    new_gmm.means_ = np.array(means_)
    new_gmm.covars_ = np.array(covars_)
    new_gmm.converged_ = True
    return new_gmm


def select_clusters_based_on_constraints(graph_node,constraints):
    """In case a GMM contains multiple variations. This function decides which 
        motion variation to choose and creates a new GMM only with clusters that 
        fit the identified motion type. This is necessary for carry, pick and place
        which has variations with one hand and two hands.
    Returns
    -------
    new_gmm : sklearn.mixture.gmm
        A GMM with a reduced set of clusters 
    """
    gmm = graph_node.mp.gmm
    cluster_annotation = graph_node.cluster_annotation
    if len(constraints) > 0 and cluster_annotation is not None:
        hand_annotation = check_constrained_hands(constraints)
        if hand_annotation in cluster_annotation.keys():
            clusters = cluster_annotation[hand_annotation]
            gmm = extract_clusters_from_gmm(gmm, clusters)
    return gmm
    
    




def print_options(options):
    for key in options.keys():
        print key,options[key]
    return


        
          
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
                            activate_cluster_search=True
                            ):
                                
    """Should be used to generate a dict containing all settings for the algorithm
    Returns
    ------
    options : dict
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
    
    options = {"use_constraints":use_constraints,
           "use_optimization":use_optimization,
           "use_constrained_gmm" : use_constrained_gmm,
           "use_transition_model":use_transition_model,
           "activate_parameter_check":activate_parameter_check,
           "apply_smoothing":apply_smoothing,
           "optimization_settings": optimization_settings,
           "constrained_gmm_settings":constrained_gmm_settings,
           "trajectory_following_settings" : trajectory_following_settings,
           "activate_cluster_search" : activate_cluster_search
            }
    return options
                
                



def constrain_primitive(motion_primitive, gmm,constraint, prev_frames,start_pose,bvh_reader,node_name_map = None,
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

    cgmm = ConstrainedGMM(motion_primitive,gmm,constraint = None,bvh_reader=bvh_reader,node_name_map=node_name_map,settings=constrained_gmm_settings, verbose=verbose)
    cgmm.set_constraint(constraint, prev_frames, start_pose,size=size,firstFrame=firstFrame,
                        lastFrame=lastFrame)
    return cgmm


def multiple_constrain_primitive(motion_primitive,gmm, constraints,prev_frames, start_pose,bvh_reader,node_name_map=None,
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
                                         bvh_reader,
                                         node_name_map=node_name_map,size=size,
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
                                    gpm, prev_frames,start_pose,bvh_reader,node_name_map=None, constraints=None,
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

    predict_gmm = GP_predict.predict(gpm, first_s[None, :])
    if constraints:
        cgmm = multiple_constrain_primitive(second_primitive, second_gmm,constraints,
                                            prev_frames,start_pose, bvh_reader,node_name_map, size, precision,verbose=verbose)

        constrained_predict_gmm = mul(predict_gmm, cgmm)
        return mul(constrained_predict_gmm, second_gmm)
    else:
        return mul(predict_gmm, second_gmm)




def sample_from_gmm(graph_node,gmm, constraints,prev_frames,start_pose,bvh_reader,node_name_map=None,\
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
            valid = check_sample_validity(graph_node,s,bvh_reader,node_name_map) 
        else:
            valid = True
        if valid: 
#            tmp_bad_samples = 0
            samples.append(s)
            min_distance,successes = evaluate_list_of_constraints(graph_node.mp,s,constraints,prev_frames,start_pose,bvh_reader,node_name_map,
                                                        precision=precision,verbose=verbose)
            # check the root path for each sample, punish the curve walking
            acr_length = get_step_length_for_sample(graph_node.mp, 
                                                    s, 
                                                    method = "arc_length")
            absolute_length = get_step_length_for_sample(graph_node.mp,
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

def sample_and_pick_best(graph_node,gmm, constraints,prev_frames,start_pose,bvh_reader,node_name_map=None,\
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
    samples, distances, successes =  sample_from_gmm(graph_node,gmm, constraints,prev_frames,start_pose,bvh_reader,node_name_map=node_name_map,\
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
    
    

def search_for_best_sample(graph_node,constraints,prev_frames,start_pose,bvh_reader,node_name_map=None,\
                        precision = {"pos":1,"rot":1,"smooth":1},verbose=False):

    """ Directed search in precomputed hierarchical space partitioning data structure
    """
    data = graph_node.mp, constraints, prev_frames,start_pose, bvh_reader, node_name_map,precision
    distance, s = graph_node.search_best_sample(obj_error_sum,data)
    print "found best sample with distance:",distance
    global_counter_dict["motionPrimitveErrors"].append(distance)
    return np.array(s)
                                     


def extract_gmm_from_motion_primitive(pipeline_parameters):
    """ Restrict the gmm to samples that roughly fit the constraints and 
        multiply with a predicted GMM from the transition model.
    """
    
    morphable_graph,action_name,mp_name,constraints,\
    options, prev_action_name, prev_mp_name, prev_frames, prev_parameters, \
    bvh_reader, node_name_map, \
    start_pose,verbose = pipeline_parameters
     
    sample_size = options["constrained_gmm_settings"]["sample_size"]
    # Get prior gaussian mixture model from node
    graph_node = morphable_graph.subgraphs[action_name].nodes[mp_name]
    gmm = graph_node.mp.gmm

    # Perform manipulation based on settings and the current state.
    if options["use_transition_model"] and prev_parameters is not None:
        if options["use_constrained_gmm"]:
            transition_key = action_name +"_"+mp_name
            
            gpm = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].outgoing_edges[transition_key].transition_model 
            prev_primitve = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].mp

            gmm = create_next_motion_distribution(prev_parameters, prev_primitve,\
                                                graph_node,gmm,\
                                                gpm, prev_frames, start_pose,\
                                                bvh_reader, node_name_map,\
                                                constraints,sample_size,\
                                                 options["constrained_gmm_settings"],\
                                                verbose=verbose)
        else:
            to_key = action_name+"_"+mp_name
            gmm = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].predict_gmm(to_key,prev_parameters)
    elif options["use_constrained_gmm"]:
        gmm = multiple_constrain_primitive(graph_node,gmm,\
                                        constraints,\
                                        prev_frames,start_pose, \
                                        bvh_reader, node_name_map,\
                                        sample_size, options["constrained_gmm_settings"],verbose=verbose)   

    return gmm


def get_optimal_parameters(morphable_graph,action_name,mp_name,constraints,\
                         options, prev_action_name="", prev_mp_name="", prev_frames=None, prev_parameters=None, \
                         bvh_reader=None, node_name_map=None, \
                         start_pose=None,verbose=False):
        """Uses the constraints to find the optimal paramaters for a motion primitive.
        Parameters
        ----------
        * morphable_graph : MorphableGraph
            Data structure containing the morphable models
        * action_name : string
            name of the elementary action and subgraph in morphable_graph
        * mp_name : string
            name of the motion primitive and node in the subgraph of the elemantary action
        * options : dict
            Contains options for the algorithm.
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

        sample_size = options["constrained_gmm_settings"]["sample_size"]
        precision = options["constrained_gmm_settings"]["precision"]

        if options["use_constraints"] and len(constraints) > 0: # estimate parameters fitting constraints
            graph_node = morphable_graph.subgraphs[action_name].nodes[mp_name]
            
            #  1) get gmm and modify it based on the current state and settings
            pipeline_parameters = morphable_graph,action_name,mp_name,constraints,\
                         options, prev_action_name, prev_mp_name, prev_frames, prev_parameters, \
                         bvh_reader, node_name_map, \
                         start_pose,verbose
            gmm = extract_gmm_from_motion_primitive(pipeline_parameters)

            #  2) sample parameters based on constraints and make sure
            #     the resulting motion is valid
            if options["activate_cluster_search"]:
                #  find best sample using a directed search in a 
                #  space partitioning data structure
                parameters = search_for_best_sample(graph_node,constraints,prev_frames,start_pose,\
                                        bvh_reader, node_name_map,\
                                         verbose=verbose)
                optimization_needed = True
                #optimization_needed = False
            else: 
                # pick new random samples from the Gaussian Mixture Model
                parameters,optimization_needed = sample_and_pick_best(graph_node,gmm,\
                                        constraints,prev_frames,start_pose,\
                                        bvh_reader, node_name_map,\
                                        precision= precision,\
                                        num_samples = sample_size,\
                                        activate_parameter_check=options["activate_parameter_check"],verbose=verbose)
                
            #3) optimize sampled parameters as initial guess if the constraints were not reached
            if  options["use_optimization"] and not optimization_needed:
                verbose = True
                bounding_boxes = (graph_node.parameter_bb, graph_node.cartesian_bb)
                parameters = run_optimization(graph_node.mp, gmm, constraints,
                                                parameters, bvh_reader, node_name_map,
                                                optimization_settings=options["optimization_settings"], bounding_boxes=bounding_boxes,
                                                prev_frames=prev_frames, start_pose=start_pose, verbose=verbose)

        else: # generate random parameters
            
            if options["use_transition_model"] and prev_parameters is not None:
                to_key = action_name+"_"+mp_name
                parameters = morphable_graph.subgraphs[prev_action_name].nodes[prev_mp_name].predict_parameters(to_key,prev_parameters)
            else:
                parameters = morphable_graph.subgraphs[action_name].nodes[mp_name].sample_parameters()
        return parameters