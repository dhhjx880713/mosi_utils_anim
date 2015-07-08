# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

Runs the complete Morhable Graphs Pipeline to generate a motion based on an
json input file. Runs the optimization sequentially and creates constraints based on
previous steps.

@author: Erik Herrmann, Han Du, Fabian Rupp, Markus Mauer
"""

import os
import copy
import numpy as np
from lib.bvh2 import BVHReader,create_filtered_node_name_map
from lib.morphable_graph import MorphableGraph
from lib.helper_functions import get_morphable_model_directory, \
                                 get_transition_model_directory, \
                                 load_json_file, \
                                 export_euler_frames_to_bvh,\
                                 merge_two_dicts
from lib.motion_editing import convert_quaternion_to_euler,align_frames, \
                                transform_euler_frames, \
                                get_cartesian_coordinates2
from lib.graph_walk_extraction import create_trajectory_from_constraint,\
                                    extract_all_keyframe_constraints,\
                                    extract_trajectory_constraint,\
                                    get_step_length,\
                                    extract_root_positions,\
                                    get_arc_length_from_points,\
                                    extract_key_frame_constraint,\
                                    extract_keyframe_annotations
from constrain_motion import get_optimal_parameters,\
                             generate_algorithm_settings,\
                             print_options
from constrain_gmm import ConstraintError
                             
class SynthesisError(Exception):
    def __init__(self,  euler_frames,bad_samples):
        message = "Could not process input file"
        super(SynthesisError, self).__init__(message)
        self.bad_samples = bad_samples
        self.euler_frames = euler_frames
        
class PathSearchError(Exception):
    def __init__(self,parameters):
        self.search_parameters = parameters
        message = "Error in the navigation goal generation"
        super(PathSearchError, self).__init__(message)
        
        


def get_action_list(euler_frames,time_information,constraints,keyframe_annotations,offset = 0):
    """Associates annotations to frames 
    Parameters
    ----------
    *euler_frames : np.ndarray
      motion
    * time_information : dict
      maps keyframes to frame numbers
    * constraints: list of dict
      list of constraints for one motion primitive generated 
      based on the mg input file
    * keyframe_annotations : dict of dicts
      Contains a list of events/actions associated with certain keyframes
      
    Returns
    -------
    *  action_list : dict of lists of dicts
       A dict that contains a list of actions for certain keyframes
    """
    action_list = {}
    for c in constraints:
        if "semanticAnnotation" in c.keys():
            for key in c["semanticAnnotation"]:#can also contain lastFrame and firstFrame
                 if key in keyframe_annotations.keys() and key in time_information.keys():
                      if time_information[key] == "lastFrame":
                          key_frame = len(euler_frames)-1+offset
                      elif time_information[key] == "firstFrame":
                          key_frame = offset
                      if "annotations" in keyframe_annotations[key].keys():
                          action_list[key_frame] = keyframe_annotations[key]["annotations"]
    return action_list
    
def get_optimal_motion(morphable_graph,
                       action_name,
                       mp_name,
                       constraints,
                       options = None,
                       prev_action_name="", 
                       prev_mp_name="", 
                       prev_frames=None,
                       prev_parameters=None, 
                       bvh_reader=None, 
                       node_name_map=None, 
                       start_pose=None,
                       keyframe_annotations={},
                       verbose=False):
    """Calls get_optimal_parameters and backpoject the results.
    Parameters
    ----------
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
    Returns
    -------
    * euler_frames : list of np.ndarray
        list of skeleton pose parameters.
    * parameters : np.ndarray
        low dimensional motion parameters used to generate the frames
    * step_length : float
       length of the generated motion
    * action_list :  
    """                          
    apply_smoothing = options["apply_smoothing"]   
#    if action_name == 'pick':
#        apply_smoothing = False
    try:
        parameters = get_optimal_parameters(morphable_graph,
                                            action_name,
                                            mp_name,
                                            constraints,
                                            options=options,
                                            prev_action_name=prev_action_name,
                                            prev_mp_name=prev_mp_name,
                                            prev_frames=prev_frames, 
                                            prev_parameters=prev_parameters,
                                            bvh_reader=bvh_reader, 
                                            node_name_map=node_name_map,
                                            start_pose=start_pose,
                                            verbose=verbose)
    except  ConstraintError as e: 
        print "Exception",e.message
        raise SynthesisError(prev_frames,e.bad_samples)
       
   
    #back project to euler frames#TODO implement FK based on quaternions 
    quaternion_frames = morphable_graph.subgraphs[action_name].nodes[mp_name].mp.back_project(parameters).get_motion_vector()
    tmp_euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())
    
 
    #concatenate with frames from previous steps
    if prev_frames != None:
        frame_offset = len(prev_frames)
    else:
        frame_offset = 0
    if prev_parameters != None:
        euler_frames = align_frames(bvh_reader,prev_frames,tmp_euler_frames, node_name_map,apply_smoothing) 
    else:
        if start_pose != None:
            #rotate euler frames so the transformation can be found using alignment
            print "transform euler frames using start pose",start_pose
            euler_frames = transform_euler_frames(tmp_euler_frames,start_pose["orientation"],start_pose["position"])  
            print "resulting start position",euler_frames[0][:3]
        else:

            euler_frames = tmp_euler_frames
#            print 'length of euler frames in get optimal motion from no previous frames: ' + str(len(euler_frames))
    #associate keyframe annotation to euler_frames
    if mp_name in  morphable_graph.subgraphs[action_name].mp_annotations.keys():
        time_information = morphable_graph.subgraphs[action_name].mp_annotations[mp_name]
    else:
        time_information = {}
    action_list = get_action_list(tmp_euler_frames,time_information,constraints,keyframe_annotations,offset =frame_offset)
    return euler_frames,parameters,action_list
    

#def create_constraints_for_trajectory_primitive(joint_name,position=[None,None,None],\
#                            orientation=[None,None,None],semantic_annotation=None):
#    """ Wrapper around a dict object creation
#    Returns 
#    -------
#    * constraints : list of dicts
#      Each dict contains joint, position,orientation and semanticAnnotation describing a constraint
#    """
#    constraints = []
#    constraint = {"joint":joint_name,"position":position,"orientation":[None,None,None],
#    "semanticAnnotation":semantic_annotation} # (joint, [pos_x, pos_y, pos_z],[rot_x, rot_y, rot_z])
#    constraints.append(constraint)
#
#    return constraints

def get_point_from_step_length(morphable_subgraph,current_state,trajectory,step_length,unconstrained_indices = [],step_length_factor= 1.0,method = "arc_length"):
    """ Returns a point on the trajectory for the step length of a random sample
    """    
    step_length += get_step_length(morphable_subgraph,current_state,method) * step_length_factor
    goal = trajectory.query_point_by_absolute_arc_length(step_length).tolist()
    for i in unconstrained_indices:
        goal[i] = None
    return goal
    
    
def get_point_and_orientation_from_arc_length(trajectory,arc_length,unconstrained_indices):
    """ Returns a point, an orientation and a directoion vector on the trajectory 
    """   
    point = trajectory.query_point_by_absolute_arc_length(arc_length).tolist()
    
    reference_vector = np.array([0,1])# in z direction
    start,dir_vector,angle = trajectory.get_angle_at_arc_length_2d(arc_length,reference_vector)
    orientation = [None,angle,None]
    for i in unconstrained_indices:
        point[i] = None
        orientation[i] = None
    return point,orientation,dir_vector
    
    
def make_guess_for_goal_arc_length(morphable_subgraph,current_state,trajectory,
                                   last_arc_length,last_pos,unconstrained_indices = [],
                                    step_length_factor= 1.0,method = "arc_length"):
                                                    
    """ Makes a guess for a reachable arc length based on the current position.
        It searches for the closest point on the trajectory, retrieves the absolute arc length
        and its the arc length of a random sample of the next motion primitive
    Returns
    -------
    * arc_length : float
      The absolute arc length of the new goal on the trajectory. 
      The goal should then be extracted using get_point_and_orientation_from_arc_length
    """  
  
    step_length = get_step_length(morphable_subgraph,current_state,method) * step_length_factor
    max_arc_length = last_arc_length + 4.0* step_length
    #find closest point in the range of the last_arc_length and max_arc_length
    closest_point,distance = trajectory.find_closest_point(last_pos,min_arc_length = last_arc_length,max_arc_length=max_arc_length)
    if closest_point ==None:
        parameters = {"last":last_arc_length,"max":max_arc_length,"full":trajectory.full_arc_length}
        print "did not find closest point",closest_point,str(parameters)
        raise PathSearchError(parameters)
    # approximate arc length of the point closest to the current position
    arc_length,eval_point = trajectory.get_absolute_arc_length_of_point(closest_point,min_arc_length = last_arc_length)
    #update arc length based on the step length of the next motion primitive
    arc_length += step_length
    return arc_length


def create_frame_constraint(bvh_reader,node_name_map,prev_frames):
    """
    create frame a constraint from the preceding motion.

    """

    last_frame = prev_frames[-1]
    position_dict = {}
    for node_name in node_name_map.keys():
        
        joint_position = get_cartesian_coordinates2(bvh_reader,
                                                        node_name,
                                                        last_frame,
                                                        node_name_map)
        print "add joint position to constraints",node_name,joint_position
        position_dict[node_name] = joint_position
    frame_constraint = {"frame_constraint":position_dict, "semanticAnnotation":{"firstFrame":True,"lastFrame":None}} 
  
    return frame_constraint
 
 

def create_constraints_for_trajectory_primitive(morphable_subgraph,current_state,trajectory,
                                                 last_arc_length,last_pos,unconstrained_indices,
                                                 settings,joint_name,prev_frames = None,
                                                 bvh_reader = None,node_name_map = None,
                                                 keyframe_constraints={},semantic_annotation=None,
                                                 is_last_step = False):
    """ Creates a list of constraints for a motion primitive based on the current state and position
    Returns 
    -------
    * constraints : list of dicts
      Each dict contains joint, position,orientation and semanticAnnotation describing a constraint
    """
    
    
    step_length_factor = settings["step_length_factor"] #0.8 #used to increase the number of samples that can reach the point
    method =  settings["method"]#"arc_length"
    last_pos = copy.copy(last_pos)
    last_pos[1] = 0.0
    print "search for new goal from",last_pos
    # if it is the last step we need to reach the point exactly otherwise
    # make a guess for a reachable point on the path that we have not visited yet
    if not is_last_step:
        arc_length = make_guess_for_goal_arc_length(morphable_subgraph,
                                               current_state,trajectory,
                                               last_arc_length,last_pos,
                                               unconstrained_indices,
                                               step_length_factor,method)
    else:
        arc_length = trajectory.full_arc_length 

    goal,orientation,dir_vector = get_point_and_orientation_from_arc_length(trajectory,arc_length,unconstrained_indices)
    print "new goal for",current_state,goal,last_arc_length
        
    constraints = []
    if settings["use_frame_constraints"] and  prev_frames != None and bvh_reader != None and node_name_map != None:
        frame_constraint= create_frame_constraint(bvh_reader,node_name_map,prev_frames)
        constraints.append(frame_constraint)
            
#    else:
#        print "did not create first frame constraint #####################################"

    if settings["use_position_constraints"] :
        if not is_last_step:
            pos_semantic_annotation={"firstFrame":None,"lastFrame":None}
        else:
            pos_semantic_annotation={"firstFrame":None,"lastFrame":True}
        pos_constraint = {"joint":joint_name,"position":goal, 
                  "semanticAnnotation":pos_semantic_annotation} 
        constraints.append(pos_constraint)

    if settings["use_dir_vector_constraints"] :  
        rot_semantic_annotation={"firstFrame":None,"lastFrame":True}
        rot_constraint = {"joint":joint_name, "dir_vector":dir_vector,
                      "semanticAnnotation":rot_semantic_annotation} 
        constraints.append(rot_constraint)
        
    use_optimization = len(keyframe_constraints.keys()) > 0
    
    if current_state in keyframe_constraints.keys():
        constraints+= keyframe_constraints[current_state]
        
    return constraints, arc_length, use_optimization 
    
    
def prepare_keyframe_constraints_for_motion_primitves(morphable_subgraph,keyframe_constraints):
     """ Order constraints extracted by extract_all_keyframe_constraints for each state  
     """
     constraints = {}#dict of lists
     for label in keyframe_constraints.keys():#iterate over labels
        state = morphable_subgraph.annotation_map[label]#
        constraints[state] = []
        for joint_name in keyframe_constraints[label].keys():
            time_information = morphable_subgraph.mp_annotations[state][label]
            for c in keyframe_constraints[label][joint_name]:
                constraint_desc = extract_key_frame_constraint(joint_name,c,\
                                            morphable_subgraph,time_information)
                constraints[state].append(constraint_desc)
     return constraints
    

def generate_navigation_motion(morphable_graph,elementary_action,joint_name, trajectory,keyframe_constraints=None,\
                        unconstrained_indices = [], options = None,\
                        prev_action_name="", prev_mp_name="", prev_frames=None, prev_parameters=None,\
                        bvh_reader =None,node_name_map = None,\
                        start_pose = None, step_count = 0, max_step = -1,\
                        keyframe_annotations={},verbose = False):
    """Divides a trajectory into a list of segments and generates a motion
    by sequentially specifying constraints for each motion primitive.
    
    Parameters
    ---------
    *morphable_graph: MorphableGraph
    \tRepresents an elementary action.   
    *elementary_action: string
    \tIdentifier of the subgraph of the elementary action.
    *trajectory: CatmullRomSpline
    \tThe trajectory that should be followed. It needs to start at the origin.
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
    * start_pose : dict
    \tContains orientation and position as lists with three elements
    * keyframe_annotations : list of dicts
    \tContains a list of events/actions associated with certain keyframes
    
    Returns
    -------
    * euler_frames : np.ndarray
    * last_action_name : string
    * last_mp_name : string
    * last_parameters : np.ndarray
    * step_count : integer
    * action_list : dict of lists of dict
    \t euler_frames + information for the transiton to another elementary action
    """
    
    ######################################################################################################
    #initialize algorithm
    if options == None:
        options = generate_algorithm_settings()
        
    original_use_optimization = options["use_optimization"]
    action_list = {}
    trajectory_following_settings = options["trajectory_following_settings"]
    morphable_subgraph = morphable_graph.subgraphs[elementary_action]
    keyframe_constraints = prepare_keyframe_constraints_for_motion_primitves(morphable_subgraph,keyframe_constraints)
    step_length_at_end = get_step_length(morphable_subgraph,morphable_subgraph.get_random_end_state())   
    
    euler_frames = prev_frames
    travelled_arc_length = 0.0 
    full_length = trajectory.full_arc_length#get_full_arc_length()
    print "end last step",step_length_at_end,full_length,"#################################"
   
    ######################################################################################################
    #generate constraint and motion for the first step
    start_state = morphable_subgraph.get_random_start_state()
    if verbose: 
        print "start at",start_state
    current_state = start_state
    
    if euler_frames != None:
        last_pos = euler_frames[-1][:3]
    else:
        last_pos = start_pose["position"]
    constraints, temp_step_length, use_optimization = create_constraints_for_trajectory_primitive(morphable_subgraph,\
                  current_state,trajectory,travelled_arc_length,last_pos,unconstrained_indices,trajectory_following_settings,\
                  joint_name,euler_frames,bvh_reader,node_name_map,keyframe_constraints,is_last_step = False)
#    print constraints
    options["use_optimization"] = use_optimization
    euler_frames,parameters,tmp_action_list = get_optimal_motion(morphable_graph,elementary_action,current_state,constraints,\
                                        prev_action_name=prev_action_name, prev_mp_name=prev_mp_name, prev_frames=euler_frames, prev_parameters=prev_parameters,\
                                        options=options, bvh_reader=bvh_reader,node_name_map=node_name_map,\
                                        start_pose=start_pose,keyframe_annotations=keyframe_annotations,verbose=verbose)
    action_list = merge_two_dicts(action_list,tmp_action_list )                            
    #update with step length of optimized motion
   
    travelled_arc_length = temp_step_length           
   
    prev_parameters = parameters
    prev_action_name = elementary_action
    prev_mp_name = current_state
    step_count+=1
    
    ######################################################################################################
    #loop until stop condition is met
    #arbort when a maximum step length has been reached
    #make sure that the step lenght is increasing
    #if not it indicates the end could not be reached exactly
    distance_to_end = full_length
    while distance_to_end > step_length_at_end and travelled_arc_length <= full_length-step_length_at_end:# 

        if max_step > -1 and step_count > max_step:
            print "reached max step"
            break
        if  morphable_subgraph.nodes[current_state].n_standard_transitions > 0 :
            to_key = morphable_subgraph.nodes[current_state].generate_random_transition("standard") 
            current_state = to_key.split("_")[1]
            if verbose: 
                print "transition to ",current_state,"at step",step_count
            
            last_pos = euler_frames[-1][:3]
            try :
                constraints, temp_step_length, use_optimization = create_constraints_for_trajectory_primitive(morphable_subgraph,\
                      current_state,trajectory,travelled_arc_length,last_pos,unconstrained_indices,trajectory_following_settings,\
                      joint_name,euler_frames,bvh_reader,node_name_map,keyframe_constraints,is_last_step = False)
                options["use_optimization"] = use_optimization
            except PathSearchError as e:
                print "moved beyond end point using parameters",
                str(e.search_parameters)
                break
            euler_frames,parameters,tmp_action_list  = get_optimal_motion(morphable_graph,elementary_action,current_state,constraints,\
                                                prev_action_name=prev_action_name, prev_mp_name=prev_mp_name, prev_frames=euler_frames, prev_parameters=prev_parameters,\
                                                options=options, bvh_reader=bvh_reader,node_name_map=node_name_map,\
                                                start_pose=start_pose,keyframe_annotations=keyframe_annotations,verbose=verbose)                                   
            action_list = merge_two_dicts(action_list,tmp_action_list )  
            #update with step length of optimized motion
            travelled_arc_length = temp_step_length     
            #print "##################### update step length",step_count,temp_step_length
            prev_parameters = parameters
            prev_action_name = elementary_action
            prev_mp_name = current_state   
        
           
            distance_to_end = np.linalg.norm(trajectory.get_last_control_point() - euler_frames[-1][:3])
            print "new distance ",distance_to_end ,"################################",step_count
            step_count+=1
                     
        else:
            print "did not find suitable transitions"
            break
    stop_condition = (distance_to_end > step_length_at_end and travelled_arc_length <= full_length-step_length_at_end)
    print "stopped loop at arc length", \
            step_count,\
            travelled_arc_length,\
            full_length,\
            step_length_at_end,\
            distance_to_end,\
            stop_condition,"##############################"
    
    
    
    ######################################################################################################
    #generate constraint and motion for the last step
    if max_step <= -1 or step_count < max_step:
        #add end state
        to_key = morphable_subgraph.nodes[current_state].generate_random_transition("end")
        current_state = to_key.split("_")[1]
        if verbose:
            print "end at",current_state
        travelled_arc_length = trajectory.full_arc_length
        last_pos = euler_frames[-1][:3]
        constraints, temp_step_length, use_optimization = create_constraints_for_trajectory_primitive(morphable_subgraph,\
                  current_state,trajectory,travelled_arc_length,last_pos,unconstrained_indices,trajectory_following_settings,\
                  joint_name,euler_frames,bvh_reader,node_name_map,keyframe_constraints,is_last_step = True)
        
        options["use_optimization"] = use_optimization
        #constraint = {"joint":joint_name,"position":goal,"orientation":[None,None,None],"firstFrame":None,"lastFrame":True}# (joint, [pos_x, pos_y, pos_z],[rot_x, rot_y, rot_z])
        euler_frames,parameters,tmp_action_list = get_optimal_motion(morphable_graph,elementary_action,current_state,constraints,\
                                                    prev_action_name=prev_action_name, prev_mp_name=prev_mp_name, prev_frames=euler_frames, prev_parameters=prev_parameters,\
                                                    options=options, bvh_reader=bvh_reader,node_name_map=node_name_map,\
                                                    start_pose=start_pose,keyframe_annotations=keyframe_annotations,verbose=verbose)
        action_list = merge_two_dicts(action_list,tmp_action_list )      
        prev_parameters = parameters
        prev_action_name = elementary_action
        prev_mp_name = current_state     
        step_count+=1
    options["use_optimization"] = original_use_optimization
    
    return euler_frames,prev_action_name,prev_mp_name,prev_parameters,step_count,action_list
       
        
def convert_trajectory_and_keyframe_constraints_to_motion(morphable_graph, trajectory_constraint,keyframe_constraints, \
                                            elementary_action, joint_name, options = None,\
                                              prev_action_name="", prev_mp_name="", prev_frames=None,\
                                              prev_parameters=None,bvh_reader =None,node_name_map = None,\
                                            start_pose = None, first_action = False, step_count = 0, max_step = -1,\
                                            keyframe_annotations={},verbose=False):     
    """Converts one trajectory constraint and a list of keyframe constraints
       for one elementary action into a list of euler frames.
    
    Parameters
    ----------
     * morphable_subgraph : MorphableGraph
    \tRepresents an elementary action.
    *trajectory_constraint : dict
    \tEntry in an trajectory constraint array of the Morphable Graphs interface
    *keyframe_constraints : list
    \tThe keyframe constraints array of an elementary action from the Morphable Graphs input file
    * elementary_action : string
    \tName of an elementary action.
    * joint_name : string
    \tJoint identifier
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
    * start_pose : dict
    \tContains orientation and position as lists with three elements
    * first_action : bool
    \t When this is true the origin is added to the control points
    
    Returns
    -------
    * euler_frames : np.ndarray
    * last_action_name : string
    * last_mp_name : string
    * last_parameters : np.ndarray
    * step_count : integer
    * action_list : dict of lists of dict
     \t euler_frames + information for the transiton to another elementary action
    """
    if options == None: 
        options = generate_algorithm_settings()

    trajectory,unconstrained_indices = create_trajectory_from_constraint(trajectory_constraint)
 
    result = generate_navigation_motion(morphable_graph,elementary_action,joint_name,\
                                                                trajectory,keyframe_constraints,unconstrained_indices=unconstrained_indices,\
                                                                 options = options,\
                                                                 prev_action_name=prev_action_name, prev_mp_name=prev_mp_name, prev_frames=prev_frames,\
                                                                prev_parameters=prev_parameters,\
                                                                bvh_reader =bvh_reader,node_name_map = node_name_map,\
                                                                start_pose=start_pose,step_count=step_count,max_step=max_step,\
                                                                keyframe_annotations=keyframe_annotations,verbose=verbose)
    return result
    

    
def convert_keyframe_constraints_to_motion(morphable_graph, 
                                           keyframe_constraints,
                                           elementary_action,
                                           options = None,
                                           prev_action_name="", 
                                           prev_mp_name="", 
                                           prev_frames=None,
                                           prev_parameters=None,
                                           bvh_reader =None,
                                           node_name_map = None,
                                           start_pose = None,
                                           first_action = False,
                                           step_count = 0,
                                           max_step = -1,
                                           keyframe_annotations={},
                                           verbose=False):
    """Converts keyframe constraints for one elementary action into a list of
    euler frames.
    
    Parameters
    ----------
     * morphable_subgraph : MorphableGraph
    \tRepresents an elementary action.
    *keyframe_constraints : list
    \tThe keyframe constraints array of an elementary action from the Morphable Graphs input file
    * elementary_action : string
    \tName of an elementary action.
    * joint_name : string
    \tJoint identifier
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
    * start_pose : dict
    \tContains orientation and position as lists with three elements
    * first_action : bool
    \t When this is true the origin is added to the control points
    
    Returns
    -------
    * euler_frames : np.ndarray
    * last_action_name : string
    * last_mp_name : string
    * last_parameters : np.ndarray
    * step_count : integer
    * action_list : dict of lists of dict
     \t euler_frames + information for the transiton to another elementary action
    """
#    print "parameters passed to action", elementary_action,prev_frames,"#################"
    if options == None:
        options = generate_algorithm_settings()
    settings = options["trajectory_following_settings"]#TODO move options to different key
    
    action_list = {}
    euler_frames = prev_frames
    step_count = 0
    morphable_subgraph = morphable_graph.subgraphs[elementary_action]
    nodes = morphable_subgraph.nodes
    number_of_standard_transitions = len([n for n in \
                                 nodes.keys() if nodes[n].node_type == "standard"])
                                 
    ordered_keyframe_constraints = prepare_keyframe_constraints_for_motion_primitves(morphable_subgraph,keyframe_constraints)

    ######################################################################################################
    # add start state
    current_state = morphable_subgraph.get_random_start_state()
    if current_state in ordered_keyframe_constraints.keys():
        constraints = ordered_keyframe_constraints[current_state]
    else:
        constraints = []
    if settings["use_frame_constraints"] and  prev_frames != None and  bvh_reader != None and node_name_map != None:
        frame_constraint= create_frame_constraint(bvh_reader,node_name_map,prev_frames)
        constraints.append(frame_constraint)
    euler_frames,parameters,tmp_action_list = \
    get_optimal_motion(morphable_graph,
                       elementary_action,
                       current_state,
                       constraints,
                       prev_action_name=prev_action_name, 
                       prev_mp_name=prev_mp_name, 
                       prev_frames=euler_frames, 
                       prev_parameters=prev_parameters,
                       options=options, 
                       bvh_reader=bvh_reader,
                       node_name_map=node_name_map,
                       start_pose=start_pose,
                       keyframe_annotations=keyframe_annotations,
                       verbose=verbose)
                                        
    action_list = merge_two_dicts(action_list,tmp_action_list )          
    prev_parameters = parameters
    prev_action_name = elementary_action
    prev_mp_name = current_state     
    step_count+=1
    
    ######################################################################################################
    #add in between states
    standard_count = 0
    while  standard_count < number_of_standard_transitions:
            
        if nodes[current_state].n_standard_transitions > 0:
            to_key = nodes[current_state].generate_random_transition("standard") 
            current_state = to_key.split("_")[1]
            if current_state in ordered_keyframe_constraints.keys():
                constraints = ordered_keyframe_constraints[current_state]
            else:
                constraints = []
            
            if settings["use_frame_constraints"] and  prev_frames != None and  bvh_reader != None and node_name_map != None:
                frame_constraint= create_frame_constraint(bvh_reader,node_name_map,prev_frames)
                constraints.append(frame_constraint)
                
            euler_frames,parameters,tmp_action_list = get_optimal_motion(morphable_graph,elementary_action,current_state,constraints,\
                                            prev_action_name=prev_action_name, prev_mp_name=prev_mp_name, prev_frames=euler_frames, prev_parameters=prev_parameters,\
                                            options=options, bvh_reader=bvh_reader,node_name_map=node_name_map,\
                                            start_pose=start_pose,keyframe_annotations=keyframe_annotations,verbose=verbose)
            action_list = merge_two_dicts(action_list,tmp_action_list )    
            prev_parameters = parameters
            prev_action_name = elementary_action
            prev_mp_name = current_state     
            step_count += 1
            standard_count += 1
        else:
            break
    ######################################################################################################
    #add end state
    to_key = nodes[current_state].generate_random_transition("end")
    previous_state = current_state
    current_state = to_key.split("_")[1]
    if current_state in ordered_keyframe_constraints.keys():
        constraints = ordered_keyframe_constraints[current_state]
    else:
#        constraints = []
        constraints = ordered_keyframe_constraints[previous_state]
        # change the last frame annotation to the first frame
        for constraint in constraints:
            constraint["semanticAnnotation"]["firstFrame"] = True
            constraint["semanticAnnotation"]["lastFrame"] = None
            
    if settings["use_frame_constraints"] and  prev_frames != None and bvh_reader != None and node_name_map != None:
        frame_constraint= create_frame_constraint(bvh_reader,node_name_map,prev_frames)
        constraints.append(frame_constraint)
        
    euler_frames,parameters,tmp_action_list = \
                                get_optimal_motion(morphable_graph,
                                                   elementary_action,
                                                   current_state,
                                                   constraints,
                                                   prev_action_name=prev_action_name, 
                                                   prev_mp_name=prev_mp_name,
                                                   prev_frames=euler_frames, 
                                                   prev_parameters=prev_parameters,
                                                   options=options, 
                                                   bvh_reader=bvh_reader,
                                                   node_name_map=node_name_map,
                                                   start_pose=start_pose,
                                                   keyframe_annotations=keyframe_annotations,
                                                   verbose=verbose)
                       
    action_list = merge_two_dicts(action_list,tmp_action_list )    
    prev_parameters = parameters
    prev_action_name = elementary_action
    prev_mp_name = current_state     
    step_count+=1

 
    return euler_frames,prev_action_name,prev_mp_name,prev_parameters,step_count,action_list   

def convert_elementary_action_to_motion(action_name,
                                        constraint_list,
                                        morphable_graph,
                                        options = None,
                                        prev_action_name="", 
                                        prev_mp_name="", 
                                        prev_frames=None,
                                        prev_parameters=None,   
                                        bvh_reader =None,
                                        node_name_map = None,
                                        start_pose = None,
                                        first_action = False, 
                                        step_count = 0, 
                                        max_step = -1,
                                        keyframe_annotations={},
                                        verbose=False):
    """Convert an entry in the elementary action list to a list of euler frames. 
    Note only one trajectory constraint per elementary action is currently supported
    and it should be for the Hip joint.
    
    If there is a trajectory constraint it is used otherwise a random graph walk is used 
    if there is a keyframe constraint it is assigned to the motion primitves
    in the graph walk
    
    Paramaters
    ---------
    * action_name : string
      the identifier of the elementary action
    
    * constraint_list : list of dict
     the constraints element from the elementary action list entry
    
    * morphable_graph : MorphableGraph
    \t An instance of the MorphableGraph.
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
    * start_pose : dict
     Contains orientation and position as lists with three elements
    
    * first_action : bool
      When this is true the origin is added to the control points of trajectory constraints
    * keyframe_annotations : dict of dicts
      Contains a dict of events/actions associated with certain keyframes
      
    Returns
    -------
   * euler_frames : np.ndarray
    * last_action_name : string
    * last_mp_name : string
    * last_parameters : np.ndarray
    * step_count : integer
    * action_list : dict of lists of dict
     \t euler_frames + information for the transiton to another elementary action
    """
    action_list = {}
    euler_frames = prev_frames
    morphable_subgraph = morphable_graph.subgraphs[action_name]

    #currently we only look for trajectories on the Hips joint
    joint_name = "Hips"
    trajectory_constraint = extract_trajectory_constraint(constraint_list,joint_name)
    keyframe_constraints = extract_all_keyframe_constraints(constraint_list,morphable_subgraph)
#    print "#######################################################"
#    print "trajectory_constraint: ", trajectory_constraint
#    print "keyframe_constraints: ", keyframe_constraints
    #create a motion based on a trajectory
    if trajectory_constraint != None:
#        print "start trajectory constraint: "
#        print "trajectory_constraint is: "
#        print trajectory_constraint
        euler_frames,prev_action_name,prev_mp_name,prev_parameters,step_count,tmp_action_list = \
                        convert_trajectory_and_keyframe_constraints_to_motion(morphable_graph,\
                                trajectory_constraint,keyframe_constraints,action_name,joint_name,\
                                options = options,\
                                 prev_action_name=prev_action_name, prev_mp_name=prev_mp_name, \
                                 prev_frames=euler_frames,prev_parameters=prev_parameters,\
                                 bvh_reader =bvh_reader,node_name_map = node_name_map,\
                                 start_pose = start_pose, first_action = first_action,\
                                 step_count = step_count, max_step = max_step,\
                                 keyframe_annotations=keyframe_annotations,verbose=verbose)     
        action_list = merge_two_dicts(action_list,tmp_action_list)
    else:
        #create a motion randomly with a fixed maximum number of transitions derived from 
        #the number and types of motion primitives
        euler_frames,prev_action_name,prev_mp_name,prev_parameters, step_count,tmp_action_list = \
                    convert_keyframe_constraints_to_motion(morphable_graph,keyframe_constraints,
                                                   action_name,options = options,
                                                   prev_action_name=prev_action_name, 
                                                   prev_mp_name=prev_mp_name,                                                      
                                                   prev_frames=euler_frames,prev_parameters=prev_parameters,\
                                                   bvh_reader =bvh_reader,node_name_map = node_name_map,\
                                                   start_pose = start_pose, first_action = first_action,\
                                                   step_count = step_count, max_step = max_step,\
                                                   keyframe_annotations=keyframe_annotations,verbose=verbose)
        action_list = merge_two_dicts(action_list,tmp_action_list)                                       
    return euler_frames,prev_action_name,prev_mp_name,prev_parameters,step_count,action_list



def convert_elementary_action_list_to_motion(morphable_graph,elementary_action_list,\
    options = None, bvh_reader=None,node_name_map = None, \
     max_step =  -1, start_pose=None, keyframe_annotations={},verbose = False):
    """ Converts a constrained graph walk to euler frames
     Parameters
    ----------
    * morphable_graph : MorphableGraph
        Data structure containing the morphable models
    * elementary_action_list : list of dict
        Contains a list of dictionaries with the entries for "subgraph","state" and "parameters"
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
    * bvh_reader : BVHReader
        Used for to extract the skeleton hierarchy information.
    * node_name_map : dict
        Maps joint names to indices in their original loading sequence ignoring the "Bip" joints
    * max_step : integer
        Sets the maximum number of graph walk steps to be performed. If less than 0
        then it is unconstrained
    * start_pose : dict
        Contains keys position and orientation. "position" contains Cartesian coordinates 
        and orientation contains Euler angles in degrees)
    * keyframe_annotations : list of dicts of dicts
      Contains for every elementary action a dict that associates of events/actions with certain keyframes
      
    Returns
    -------
    * concatenated_frames : np.ndarray
    \tA list of euler frames representing a motion.
    * frame_annotation : dict
    \tAssociates the euler frames with the elementary actions
    * action_list : dict of lists of dict
    \tContains actions/events for some frames based on the keyframe_annotations 
    """
    if options == None:
        options = generate_algorithm_settings()
    if verbose:
        print_options(options)
        print "max_step",max_step
        
    action_list  ={} 
    frame_annotation = {}
    frame_annotation['elementaryActionSequence'] = []
    euler_frames = None
    prev_parameters = None
    prev_action_name = ""
    prev_mp_name = ""

    step_count = 0
    for action_index in range(len(elementary_action_list)) :
        action = elementary_action_list[action_index]["action"]
        if verbose:
            print "convert",action,"to graph walk"
        first_action = euler_frames == None
        action_annotations = keyframe_annotations[action_index]
        constraints = elementary_action_list[action_index]["constraints"]

        if euler_frames != None:
            start_frame=len(euler_frames)
        else:
            start_frame=0
  
        try :
            euler_frames,prev_action_name,prev_mp_name,prev_parameters,step_count,tmp_action_list = \
                        convert_elementary_action_to_motion(action,constraints,
                                                        morphable_graph,options = options,
                                                         prev_action_name=prev_action_name, prev_mp_name=prev_mp_name, prev_frames=euler_frames,\
                                                         prev_parameters=prev_parameters,\
                                                         bvh_reader =bvh_reader,node_name_map = node_name_map,\
                                                         start_pose=start_pose,first_action=first_action, step_count = step_count, max_step = max_step,keyframe_annotations=action_annotations,verbose=verbose)
            action_list = merge_two_dicts(action_list,tmp_action_list)


        except SynthesisError as e:
            print "Arborting conversion",e.message
            return e.euler_frames,frame_annotation,action_list
            
            #update frame annotation
            action_frame_annotation = {}
            action_frame_annotation["startFrame"]=start_frame
            action_frame_annotation["elementaryAction"]=action
            action_frame_annotation["endFrame"]=len(euler_frames)-1
            frame_annotation['elementaryActionSequence'].append(action_frame_annotation)
    return euler_frames,frame_annotation, action_list


    
def run_pipeline(mg_input_filename,output_dir="output",max_step = -1, options=None,verbose = False):
    """Converts a file with a list of elementary actions to a bvh file
    """
    
    mm_directory = get_morphable_model_directory()
    transition_directory = get_transition_model_directory()
    if "use_transition_model" in options.keys():
        use_transition_model = options[ "use_transition_model"]
    else:
        use_transition_model = False
    morphable_graph = MorphableGraph(mm_directory,transition_directory,use_transition_model)
    skeleton_path = "lib"+os.sep + "skeleton.bvh"
    bvh_reader = BVHReader(skeleton_path)    
    node_name_map = create_filtered_node_name_map(bvh_reader)

    mg_input = load_json_file(mg_input_filename)

    elementary_action_list = mg_input["elementaryActions"]
#    print "##################################################################"
#    print morphable_graph
#    print elementary_action_list[1]["constraints"]
    start_pose = mg_input["startPose"]
#    print 'start pose is: ' 
#    print start_pose
    
    keyframe_annotations = extract_keyframe_annotations(elementary_action_list)
#    print 'key frame annotations: '
#    print keyframe_annotations
    
    euler_frames, frame_annotation, action_list = convert_elementary_action_list_to_motion(morphable_graph,\
                    elementary_action_list,options,bvh_reader,node_name_map,\
                     max_step= max_step,start_pose = start_pose,keyframe_annotations=keyframe_annotations,verbose = verbose) 
    
#    convert_elementary_action_list_to_motion(morphable_graph,
#                                             elementary_action_list,
#                                             options,
#                                             bvh_reader,
#                                             node_name_map,
#                                             max_step= max_step,
#                                             start_pose = start_pose,
#                                             keyframe_annotations=keyframe_annotations,
#                                             verbose = verbose)     
    
    
    if "session" in mg_input.keys():    
        session = mg_input["session"]
    else:
        session = ""
    if euler_frames != None:
        export_euler_frames_to_bvh(output_dir,bvh_reader,euler_frames,prefix=session,start_pose= None)
    else:
        print "failed to generate motion data" 
    
def main():

    input_file = "mg_input_test001.json"
    verbose = True
    options = generate_algorithm_settings(use_constraints = True,\
                            use_optimization = True,\
                            use_transition_model = True,\
                            use_constrained_gmm = True,\
                            activate_parameter_check = True,\
                            apply_smoothing = True,
                            sample_size = 300,\
                             constrained_gmm_pos_precision = 5,
                            constrained_gmm_rot_precision = 0.15,
                            optimization_method= "BFGS", \
                            max_optimization_iterations = 50,\
                            optimization_quality_scale_factor = 0.001,\
                            optimization_error_scale_factor = 0.01,\
                            optimization_tolerance = 0.05)
    run_pipeline(input_file,max_step= 5, options=options,verbose=verbose)

    return

def test():

    input_file = r"lib\walk_pick_test.json"
    verbose = True
    options = generate_algorithm_settings(use_constraints = True,\
                            use_optimization = False,\
                            use_transition_model = False,\
                            use_constrained_gmm = True,\
                            activate_parameter_check = False,\
                            apply_smoothing = True,
                            sample_size = 50,\
                            constrained_gmm_pos_precision = 15,
                            constrained_gmm_rot_precision = 0.15,
                            constrained_gmm_smooth_precision = 4,
                            optimization_method= "BFGS", \
                            max_optimization_iterations = 25,\
                            optimization_quality_scale_factor = 1,\
                            optimization_error_scale_factor = 0.1,\
                            optimization_tolerance = 0.05)
    run_pipeline(input_file,max_step= 20, options=options,verbose=verbose)

    return


if __name__ == "__main__":
    test()
#    main()