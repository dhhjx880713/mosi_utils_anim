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
from constraint.constraint_extraction import associate_actions_to_frames


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
        
        

    tmp_quat_frames = action_constraints.parent_constraint.morphable_graph.subgraphs[action_name].nodes[mp_name].mp.back_project(parameters, use_time_parameters=True).get_motion_vector()

    return tmp_quat_frames, parameters



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
    """Convert an entry in the elementary action list to a list of quaternion frames.
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
        Contains parameters for the algorithm.
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
            motion_primitive_constraints = MotionPrimitiveConstraints(current_motion_primitive, action_constraints,travelled_arc_length,last_pos, trajectory_following_settings, motion.quat_frames, is_last_step)
#            constraints, temp_arc_length, use_optimization = create_constraints_for_motion_primitive(action_constraints,current_motion_primitive,\
#                                                                                                     travelled_arc_length,last_pos, trajectory_following_settings,\
#                                                                                                     motion.quat_frames,is_last_step=is_last_step)
        except PathSearchError as e:
                print "moved beyond end point using parameters",
                str(e.search_parameters)
                break
        # get optimal parameters, Back-project to frames in joint angle space,
        # Concatenate frames to motion and apply smoothing

       
        tmp_quat_frames, parameters = get_optimal_motion(action_constraints, motion_primitive_constraints, prev_motion=motion, algorithm_config=algorithm_config)                                            
        
        #update annotated motion
        canonical_keyframe_labels = morphable_subgraph.get_canonical_keyframe_labels(current_motion_primitive)
        start_frame = motion.n_frames
        motion.append_quat_frames(tmp_quat_frames, action_constraints.start_pose, algorithm_config["apply_smoothing"])
        last_frame = motion.n_frames-1
        motion.update_action_list(motion_primitive_constraints.constraints, action_constraints.keyframe_annotations, canonical_keyframe_labels, start_frame, last_frame)
        
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

        #update graph walk of motion
        graph_walk_entry = GraphWalkEntry(action_constraints.action_name,current_motion_primitive, parameters, travelled_arc_length)
        motion.graph_walk.append(graph_walk_entry)

        temp_step += 1

    motion.step_count += temp_step
    motion.update_frame_annotation(action_constraints.action_name, start_frame, motion.n_frames)
    
    print "reached end of elementary action", action_constraints.action_name

#    if trajectory is not None:
#        print "info", trajectory.full_arc_length, \
#               travelled_arc_length,arc_length_of_end, \
#               np.linalg.norm(trajectory.get_last_control_point() - quat_frames[-1][:3]), \
#               check_end_condition(morphable_subgraph,quat_frames,trajectory,\
#                                        travelled_arc_length,arc_length_of_end)
        
  
    return motion




def generate_motion_from_constraints(motion_constraints, algorithm_config=None):
    """ Converts a constrained graph walk to quaternion frames
     Parameters
    ----------
    * morphable_graph : MorphableGraph
        Data structure containing the morphable models
    * motion_constraints : list of dict
        Contains a list of dictionaries with the entries for "subgraph","state" and "parameters"
    * algorithm_config : dict
        Contains parameters for the algorithm.
    * skeleton : Skeleton
        Used for to extract the skeleton hierarchy information.
        
    Returns
    -------
    * motion: AnnotatedMotion
        Contains the quaternion frames and annotations of the frames based on actions.
    """
    if algorithm_config is None:
        algorithm_config = generate_algorithm_settings()
    if algorithm_config["verbose"]:
        for key in algorithm_config.keys():
            print key,algorithm_config[key]

    motion = AnnotatedMotion()
    action_constraints = motion_constraints.get_next_elementary_action_constraints()
    while action_constraints is not None:
   
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


