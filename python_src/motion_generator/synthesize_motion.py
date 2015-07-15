# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

Runs the complete Morphable Graphs Pipeline to generate a motion based on an
json input file. Runs the optimization sequentially and creates constraints
 based on previous steps.

@author: Erik Herrmann, Han Du, Fabian Rupp, Markus Mauer
"""

import copy
import numpy as np
from utilities.exceptions import SynthesisError, PathSearchError
from motion_model.morphable_graph import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from utilities.motion_editing import transform_quaternion_frames, \
                                fast_quat_frames_alignment
from constrain_motion import get_optimal_parameters,\
                             generate_algorithm_settings
from constrain_gmm import ConstraintError
from constraint.motion_constraints import MotionPrimitiveConstraints
from annotated_motion import AnnotatedMotion, GraphWalkEntry


def get_action_list(quat_frames, time_information, constraints, keyframe_annotations, start_frame, last_frame):
    """Associates annotations to frames
    Parameters
    ----------
    *quat_frames : np.ndarray
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
    key_frame_label_pairs = set()
    #extract the set of keyframes and their annotations referred to by the constraints
    for c in constraints:
        if "semanticAnnotation" in c.keys():
            for key_label in c["semanticAnnotation"]:  # can also contain lastFrame and firstFrame
                if key_label in keyframe_annotations.keys() and key_label in time_information.keys():
                    if time_information[key_label] == "lastFrame":
                        key_frame = last_frame
                    elif time_information[key_label] == "firstFrame":
                        key_frame = start_frame
                        
                    if "annotations" in keyframe_annotations[key_label].keys():
                        key_frame_label_pairs.add((key_frame,key_label))
                        
        
    #extract the annotations for the referred keyframes
    for key_frame, key_label in key_frame_label_pairs:
        annotations = keyframe_annotations[key_label]["annotations"]

        num_events = len(annotations)
        if num_events > 1:
            #check if an event is mentioned multiple times
            event_list = [(annotations[i]["event"],annotations[i]) for i in xrange(num_events)]
            temp_event_dict = dict()
            for name, event in event_list:#merge parameters to one event if it is found multiple times
                if name not in temp_event_dict.keys():
                   temp_event_dict[name]= event
                else:
                    if "joint" in temp_event_dict[name]["parameters"].keys():
                        existing_entry = copy.copy(temp_event_dict[name]["parameters"]["joint"])
                        if isinstance(existing_entry, basestring):
                            temp_event_dict[name]["parameters"]["joint"] = [existing_entry,event["parameters"]["joint"]]
                        else:
                            temp_event_dict[name]["parameters"]["joint"].append(event["parameters"]["joint"])
                        print "event dict merged",temp_event_dict[name]
                    else:
                        print "event dict merge did not happen",temp_event_dict[name]   
            action_list[key_frame] = copy.copy(temp_event_dict.values())

        else:
            action_list[key_frame] = annotations          
    return action_list
    

def get_aligned_frames(morphable_graph, action_name, mp_name, parameters, prev_frames, start_pose=None, use_time_parameters=True, apply_smoothing=True):
    
    #back project to quaternion frames
    tmp_quat_frames = morphable_graph.subgraphs[action_name].nodes[mp_name].mp.back_project(parameters, use_time_parameters).get_motion_vector()

    #concatenate with frames from previous steps

    if prev_frames is not None:
        quat_frames = fast_quat_frames_alignment(prev_frames,
                                              tmp_quat_frames,
                                              apply_smoothing)
    else:
        if start_pose is not None:
            #rotate quat frames so the transformation can be found using alignment
           quat_frames = transform_quaternion_frames(tmp_quat_frames, 
                                                     start_pose["orientation"], 
                                                     start_pose["position"])
           #print "resulting start position",quat_frames[0][:3]
        else:
            quat_frames = tmp_quat_frames
    return quat_frames


def get_optimal_motion(action_constraints, motion_primitive_constraints,
                       algorithm_config, prev_motion):
    """Calls get_optimal_parameters and backpoject the results.
    Parameters
    ----------
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
    Returns
    -------
    * quat_frames : list of np.ndarray
        list of skeleton pose parameters.
    * parameters : np.ndarray
        low dimensional motion parameters used to generate the frames
    * step_length : float
       length of the generated motion
    * action_list :
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
        if prev_motion.quat_frames is not None:
            start_frame = len(prev_motion.quat_frames)
        else:
            start_frame = 0
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
        
        


    use_time_parameters = True
    quat_frames = get_aligned_frames(action_constraints.parent_constraint.morphable_graph, action_name, mp_name,
                                     parameters, prev_motion.quat_frames, action_constraints.start_pose,
                                     use_time_parameters, algorithm_config["apply_smoothing"])

#            print 'length of quat frames in get optimal motion from no previous frames: ' + str(len(quat_frames))
    #associate keyframe annotation to quat_frames
    
    last_frame = len(quat_frames)-1
    if mp_name in action_constraints.parent_constraint.morphable_graph.subgraphs[action_constraints.action_name].mp_annotations.keys():
        time_information = action_constraints.parent_constraint.morphable_graph.subgraphs[action_constraints.action_name].mp_annotations[mp_name]
    else:
        time_information = {}
    action_list = get_action_list(quat_frames, time_information, motion_primitive_constraints.constraints, action_constraints.keyframe_annotations, start_frame, last_frame)
    return quat_frames, parameters, action_list



def check_end_condition(morphable_subgraph,prev_frames,trajectory,travelled_arc_length,arc_length_offset):
    """
    Checks wether or not a threshold distance to the end has been reached.
    Returns
    -------
    True if yes and False if not
    """
    distance_to_end = np.linalg.norm(trajectory.get_last_control_point() - prev_frames[-1][:3])
#    print "current distance to end: " + str(distance_to_end)
#    print "travelled arc length is: " + str(travelled_arc_length)
#    print "full arc length is; " + str(trajectory.full_arc_length)
#    raw_input("go on...")

    continue_with_the_loop = distance_to_end > arc_length_offset/2 and \
                        travelled_arc_length < trajectory.full_arc_length - arc_length_offset
    return not continue_with_the_loop

    
    
def get_random_start_state(motion, morphable_graph,action_name):
    """ Get random start state based on edge from previous elementary action if possible
    """
    next_state = ""
    if motion.step_count > 0:
        prev_action_name = motion.graph_walk[-1].action_name
        prev_mp_name = motion.graph_walk[-1].motion_primitive_name
  
        if prev_action_name in morphable_graph.subgraphs.keys() and \
               prev_mp_name in morphable_graph.subgraphs[prev_action_name].nodes.keys():
                                   
           to_key = morphable_graph.subgraphs[prev_action_name]\
                           .nodes[prev_mp_name].generate_random_action_transition(action_name)
           if to_key is not None:
               next_state = to_key.split("_")[1]
               return next_state
           else:
               return None
           print "generate start from transition of last action", prev_action_name, prev_mp_name, to_key
       
    # if there is no previous elementary action or no action transition
    #  use transition to random start state
    if next_state == "" or next_state not in morphable_graph.subgraphs[action_name].nodes.keys():
        print next_state,"not in", action_name#,prev_action_name,prev_mp_name
        next_state = morphable_graph.subgraphs[action_name].get_random_start_state()
        print "generate random start",next_state
    return next_state
    
def get_random_transition_state(motion, morphable_subgraph, trajectory, travelled_arc_length, arc_length_of_end):
    """ Get next state of the elementary action based on previous iteration.
    """
    prev_mp_name = motion.graph_walk[-1].motion_primitive_name
        
    if trajectory is not None :
            
         #test end condition for trajectory constraints
        if not check_end_condition(morphable_subgraph,motion.quat_frames,trajectory,\
                                travelled_arc_length,arc_length_of_end) :            

            #make standard transition to go on with trajectory following
            next_mp_type = NODE_TYPE_STANDARD
        else:
            # threshold was overstepped. remove previous step before 
            # trying to reach the goal using a last step
            #TODO replace with more efficient solution or optimization

            next_mp_type = NODE_TYPE_END
            
        print "generate",next_mp_type,"transition from trajectory"
    else:
        n_standard_transitions = len([e for e in morphable_subgraph.nodes[prev_mp_name].outgoing_edges.keys() if morphable_subgraph.nodes[prev_mp_name].outgoing_edges[e].transition_type == "standard"])
        if n_standard_transitions > 0:
            next_mp_type = NODE_TYPE_STANDARD
        else:
            next_mp_type = NODE_TYPE_END
        print "generate",next_mp_type,"transition without trajectory",n_standard_transitions

    to_key = morphable_subgraph.nodes[prev_mp_name].generate_random_transition(next_mp_type)
    
    if to_key is not None:
        current_motion_primitive = to_key.split("_")[1]
        return current_motion_primitive, next_mp_type
    else:
        return None, next_mp_type
       

def append_elementary_action_to_motion(action_constraints,
                                        algorithm_config,
                                        motion):
    """Convert an entry in the elementary action list to a list of euler frames.
    Note only one trajectory constraint per elementary action is currently supported
    and it should be for the Hip joint.

    If there is a trajectory constraint it is used otherwise a random graph walk is used
    if there is a keyframe constraint it is assigned to the motion primitves
    in the graph walk

    Paramaters
    ---------
    * elementary_action : string
      the identifier of the elementary action

    * constraint_list : list of dict
     the constraints element from the elementary action list entry

    * morphable_graph : MorphableGraph
    \t An instance of the MorphableGraph.
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
    * start_pose : dict
     Contains orientation and position as lists with three elements

    * keyframe_annotations : dict of dicts
      Contains a dict of events/actions associated with certain keyframes

    Returns
    -------
    * motion: AnnotatedMotion
    """
    
    start_frame = motion.n_frames    
    #skeleton = action_constraints.get_skeleton()
    morphable_subgraph = action_constraints.get_subgraph()
    
    trajectory_following_settings = algorithm_config["trajectory_following_settings"]#  TODO move trajectory_following_settings to different key of the algorithm_config

    arc_length_of_end = morphable_subgraph.nodes[morphable_subgraph.get_random_end_state()].average_step_length
    
#    number_of_standard_transitions = len([n for n in \
#                                 morphable_subgraph.nodes.keys() if morphable_subgraph.nodes[n].node_type == "standard"])
#
    #create sequence of list motion primitives,arc length and number of frames for backstepping 
    current_motion_primitive = None
    current_motion_primitive_type = ""
    temp_step = 0
    travelled_arc_length = 0.0
    print "start converting elementary action",action_constraints.action_name
    while current_motion_primitive_type != NODE_TYPE_END:

        if action_constraints.max_step > -1 and motion.step_count + temp_step > action_constraints.max_step:
            print "reached max step"
            break
        #######################################################################
        # Get motion primitive = extract from graph based on previous last step + heuristic
        if temp_step == 0:  
             current_motion_primitive = get_random_start_state(motion, action_constraints.parent_constraint.morphable_graph, action_constraints.action_name)
             current_motion_primitive_type = NODE_TYPE_START
             if current_motion_primitive is None:
                 if motion.step_count >0:
                     prev_action_name = motion.graph_walk[-1]
                     prev_mp_name = motion.graph_walk[-1]
                 else:
                     prev_action_name = None
                     prev_mp_name = None
                 print "Error: Could not find a transition of type action_transition from ",prev_action_name,prev_mp_name ," to state",current_motion_primitive
                 break
        elif len(morphable_subgraph.nodes[current_motion_primitive].outgoing_edges) > 0:
            prev_motion_primitive = current_motion_primitive
            current_motion_primitive, current_motion_primitive_type = get_random_transition_state(motion, morphable_subgraph, action_constraints.trajectory, travelled_arc_length, arc_length_of_end)
            if current_motion_primitive is None:
                 print "Error: Could not find a transition of type",current_motion_primitive_type,"from state",prev_motion_primitive
                 break
        else:
            print "Error: Could not find a transition from state",current_motion_primitive
            break

        print "transitioned to state",current_motion_primitive
        #######################################################################
        #Generate constraints from action_constraints
        if motion.quat_frames is None:
            last_pos = action_constraints.start_pose["position"]  
        else:
            last_pos = motion.quat_frames[-1][:3]

        try: 
            is_last_step = (current_motion_primitive_type == NODE_TYPE_END) 
            motion_primitive_constraints = MotionPrimitiveConstraints( current_motion_primitive, action_constraints,travelled_arc_length,last_pos, trajectory_following_settings, motion.quat_frames, is_last_step)
#            constraints, temp_arc_length, use_optimization = create_constraints_for_motion_primitive(action_constraints,current_motion_primitive,\
#                                                                                                     travelled_arc_length,last_pos, trajectory_following_settings,\
#                                                                                                     motion.quat_frames,is_last_step=is_last_step)
        except PathSearchError as e:
                print "moved beyond end point using parameters",
                str(e.search_parameters)
                break
        # get optimal parameters, Back-project to frames in joint angle space,
        # Concatenate frames to motion and apply smoothing

       
        motion.quat_frames, parameters, tmp_action_list = get_optimal_motion(action_constraints, motion_primitive_constraints,prev_motion=motion, algorithm_config=algorithm_config)                                            

        
        #update arc length based on new closest point
        if action_constraints.trajectory is not None:
            if len(motion.graph_walk) > 0:
                min_arc_length = motion.graph_walk[-1].arc_length
            else:
                min_arc_length = 0.0
            closest_point,distance = action_constraints.trajectory.find_closest_point(motion.quat_frames[-1][:3],min_arc_length=min_arc_length)
            travelled_arc_length,eval_point = action_constraints.trajectory.get_absolute_arc_length_of_point(closest_point,min_arc_length=travelled_arc_length)
            if travelled_arc_length == -1 :
                travelled_arc_length = action_constraints.trajectory.full_arc_length

        motion.update_action_list(tmp_action_list)
        graph_walk_entry = GraphWalkEntry(action_constraints.action_name,current_motion_primitive, parameters, travelled_arc_length)
        motion.graph_walk.append(graph_walk_entry)

        temp_step += 1

    motion.step_count += temp_step
    motion.n_frames = len(motion.quat_frames)
    motion.update_frame_annotation(action_constraints.action_name, start_frame, motion.n_frames)
    
    print "reached end of elementary action", action_constraints.action_name

#    if trajectory is not None:
#        print "info", trajectory.full_arc_length, \
#               travelled_arc_length,arc_length_of_end, \
#               np.linalg.norm(trajectory.get_last_control_point() - quat_frames[-1][:3]), \
#               check_end_condition(morphable_subgraph,quat_frames,trajectory,\
#                                        travelled_arc_length,arc_length_of_end)
        
  
    return motion# quat_frames, prev_action_name, mp_sequence[-1][0], prev_parameters, step_count, action_list




def generate_motion_from_constraints(motion_constraints, algorithm_config=None):
    """ Converts a constrained graph walk to euler frames
     Parameters
    ----------
    * morphable_graph : MorphableGraph
        Data structure containing the morphable models
    * motion_constraints : list of dict
        Contains a list of dictionaries with the entries for "subgraph","state" and "parameters"
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
    * skeleton : Skeleton
        Used for to extract the skeleton hierarchy information.
        
    Returns
    -------
    * motion: AnnotatedMotion
    
    * concatenated_frames : np.ndarray
    \tA list of euler frames representing a motion.
    * frame_annotation : dict
    \tAssociates the quat frames with the elementary actions
    * action_list : dict of lists of dict
    \tContains actions/events for some frames based on the keyframe_annotations
    """
    if algorithm_config is None:
        algorithm_config = generate_algorithm_settings()
    if algorithm_config["verbose"]:
        for key in algorithm_config.keys():
            print key,algorithm_config[key]


    motion = AnnotatedMotion()
    action_constraints = motion_constraints.get_next_elementary_action_constraints()
    while action_constraints != None:
   
        if motion_constraints.max_step > -1 and motion.step_count > motion_constraints.max_step:
            print "reached max step"
            break
          
        if algorithm_config["verbose"]:
           print "convert",action_constraints.action_name,"to graph walk"

        try:
            motion = append_elementary_action_to_motion(action_constraints, algorithm_config, motion)
            
        except SynthesisError as e:
            print "Arborting conversion",e.message
            return motion
        action_constraints = motion_constraints.get_next_elementary_action_constraints() 
        
    return motion


