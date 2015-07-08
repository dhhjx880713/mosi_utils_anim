# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015

Runs the complete Morhable Graphs Pipeline to generate a motion based on an
json input file. Runs the optimization sequentially and creates constraints
 based on previous steps.

@author: Erik Herrmann, Han Du, Fabian Rupp, Markus Mauer
"""

import copy
import numpy as np
from lib.morphable_graph import NODE_TYPE_STANDARD,NODE_TYPE_END
from lib.helper_functions import merge_two_dicts
from lib.motion_editing import convert_quaternion_to_euler, \
                                get_cartesian_coordinates2, \
                                transform_quaternion_frames, \
                                fast_quat_frames_alignment
from lib.graph_walk_extraction import create_trajectory_from_constraint,\
                                    extract_all_keyframe_constraints,\
                                    extract_trajectory_constraint,\
                                    extract_key_frame_constraint,\
                                    transform_point_from_cad_to_opengl_cs
from constrain_motion import get_optimal_parameters,\
                             generate_algorithm_settings,\
                             print_options
from constrain_gmm import ConstraintError



class SynthesisError(Exception):
    def __init__(self,  euler_frames, bad_samples):
        message = "Could not process input file"
        super(SynthesisError, self).__init__(message)
        self.bad_samples = bad_samples
        self.euler_frames = euler_frames


class PathSearchError(Exception):
    def __init__(self, parameters):
        self.search_parameters = parameters
        message = "Error in the navigation goal generation"
        super(PathSearchError, self).__init__(message)


def get_action_list(quat_frames, time_information, constraints, keyframe_annotations, offset=0):
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
    key_frame_label_pairs = set()
    #extract the set of keyframes and their annotations referred to by the constraints
    for c in constraints:
        if "semanticAnnotation" in c.keys():
            for key_label in c["semanticAnnotation"]:  # can also contain lastFrame and firstFrame
                if key_label in keyframe_annotations.keys() and key_label in time_information.keys():
                    if time_information[key_label] == "lastFrame":
                        key_frame = len(quat_frames)-1+offset
                    elif time_information[key_label] == "firstFrame":
                        key_frame = offset
                        
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
    

def get_optimal_motion(morphable_graph,
                       action_name,
                       mp_name,
                       constraints,
                       options=None,
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
    tmp_quat_frames = morphable_graph.subgraphs[action_name].nodes[mp_name].mp.back_project(parameters, use_time_parameters=True).get_motion_vector()
#    tmp_euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())

    #concatenate with frames from previous steps
    if prev_frames is not None:
        frame_offset = len(prev_frames)
    else:
        frame_offset = 0
    if prev_parameters is not None:
#        euler_frames = align_frames(bvh_reader,prev_frames,tmp_euler_frames, node_name_map,apply_smoothing)
#        quat_frames = align_quaternion_frames(bvh_reader, 
#                                              prev_frames,
#                                              tmp_quat_frames,
#                                              node_name_map,
#                                              apply_smoothing)
        quat_frames = fast_quat_frames_alignment(prev_frames,
                                              tmp_quat_frames,
                                              apply_smoothing)
    else:
        if start_pose is not None:
            #rotate euler frames so the transformation can be found using alignment
#            print "transform euler frames using start pose",start_pose
#            euler_frames = transform_euler_frames(tmp_euler_frames,start_pose["orientation"],start_pose["position"])
           #print "transform quaternion frames using start pose", start_pose
           quat_frames = transform_quaternion_frames(tmp_quat_frames, 
                                                     start_pose["orientation"], 
                                                     start_pose["position"])
           #print "resulting start position",quat_frames[0][:3]
        else:

#            euler_frames = tmp_euler_frames
            quat_frames = tmp_quat_frames
#            print 'length of euler frames in get optimal motion from no previous frames: ' + str(len(euler_frames))
    #associate keyframe annotation to euler_frames
    if mp_name in  morphable_graph.subgraphs[action_name].mp_annotations.keys():
        time_information = morphable_graph.subgraphs[action_name].mp_annotations[mp_name]
    else:
        time_information = {}
#    action_list = get_action_list(tmp_euler_frames,time_information,constraints,keyframe_annotations,offset =frame_offset)
    action_list = get_action_list(tmp_quat_frames,time_information,constraints,keyframe_annotations,offset =frame_offset)        
#    print "#########################################################################"
#    target_dir = np.array([constraints[1]["dir_vector"][0], constraints[1]["dir_vector"][2]])
#    target_dir = target_dir/np.linalg.norm(target_dir)
#    print "constraint direction: "
#    print target_dir
#    last_transformation = create_transformation(euler_frames[-1][3:6],[0, 0, 0])
#    motion_dir = transform_point(last_transformation,[0,0,1])
#    motion_dir = np.array([motion_dir[0], motion_dir[2]])
#    motion_dir = motion_dir/np.linalg.norm(motion_dir)    
#    print "current orientation: "
#    print motion_dir
#    return euler_frames,parameters,action_list
    return quat_frames, parameters, action_list

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
    step_length = morphable_subgraph.nodes[current_state].average_step_length * step_length_factor
    max_arc_length = last_arc_length + 4.0* step_length
    #find closest point in the range of the last_arc_length and max_arc_length
    closest_point,distance = trajectory.find_closest_point(last_pos,min_arc_length = last_arc_length,max_arc_length=max_arc_length)
    if closest_point is None:
        parameters = {"last":last_arc_length,"max":max_arc_length,"full":trajectory.full_arc_length}
        print "did not find closest point",closest_point,str(parameters)
        raise PathSearchError(parameters)
    # approximate arc length of the point closest to the current position
    arc_length,eval_point = trajectory.get_absolute_arc_length_of_point(closest_point,min_arc_length = last_arc_length)
    #update arc length based on the step length of the next motion primitive
    if arc_length == -1 :
        arc_length = trajectory.full_arc_length
    else:
        arc_length +=step_length# max(step_length-distance,0)
    return arc_length


def create_frame_constraint(bvh_reader,node_name_map,prev_frames):
    """
    create frame a constraint from the preceding motion.

    """

#    last_frame = prev_frames[-1]
    last_euler_frame = np.ravel(convert_quaternion_to_euler([prev_frames[-1]]))
    position_dict = {}
    for node_name in node_name_map.keys():

        joint_position = get_cartesian_coordinates2(bvh_reader,
                                                        node_name,
                                                        last_euler_frame,
                                                        node_name_map)
#        print "add joint position to constraints",node_name,joint_position
        position_dict[node_name] = joint_position
    frame_constraint = {"frame_constraint":position_dict, "semanticAnnotation":{"firstFrame":True,"lastFrame":None}}

    return frame_constraint

def prepare_keyframe_constraints_for_motion_primitves(morphable_subgraph,keyframe_constraints):
     """ Order constraints extracted by extract_all_keyframe_constraints for each state
     """
     constraints = {}#dict of lists
     #iterate over keyframe labels
     for label in keyframe_constraints.keys():
        state = morphable_subgraph.annotation_map[label]
        time_information = morphable_subgraph.mp_annotations[state][label]
        constraints[state] = []
        # iterate over joints constrained at that keyframe
        for joint_name in keyframe_constraints[label].keys():
            # iterate over constraints for that joint
            for c in keyframe_constraints[label][joint_name]:
                # create constraint definition usable by the algorithm
                # and add it to the list of constraints for that state
                constraint_desc = extract_key_frame_constraint(joint_name,c,\
                                            morphable_subgraph,time_information)
                constraints[state].append(constraint_desc)
     return constraints



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

def create_constraints_for_motion_primitive(morphable_subgraph,current_state,trajectory,
                                                 last_arc_length,last_pos,unconstrained_indices,
                                                 settings,root_joint_name,prev_frames = None,
                                                 bvh_reader = None,node_name_map = None,
                                                 keyframe_constraints={},semantic_annotation=None,
                                                 is_last_step = False):
    """ Creates a list of constraints for a motion primitive based on the current state and position
    Returns
    -------
    * constraints : list of dicts
      Each dict contains joint, position,orientation and semanticAnnotation describing a constraint
    """
    constraints = []
    arc_length = 0
    pose_constraint_set = False
    if trajectory is not None:
        step_length_factor = settings["step_length_factor"] #0.8 #used to increase the number of samples that can reach the point
        method =  settings["method"]#"arc_length"
        last_pos = copy.copy(last_pos)
        last_pos[1] = 0.0
        print "search for new goal"
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

#        print  "starting from",last_pos,last_arc_length,"the new goal for", \
#                current_state,"is",goal,"at arc length",arc_length
        print "starting from: "
        print last_pos
        print "the new goal for " + current_state
        print goal
        print "arc length is: " + str(arc_length)



        if settings["use_frame_constraints"] and  prev_frames != None and bvh_reader != None and node_name_map != None:
            frame_constraint= create_frame_constraint(bvh_reader,node_name_map,prev_frames)
            constraints.append(frame_constraint)
            pose_constraint_set = True

    #    else:
    #        print "did not create first frame constraint #####################################"

        if settings["use_position_constraints"] :
            if not is_last_step:
                pos_semantic_annotation={"firstFrame":None,"lastFrame":True}
            else:
                pos_semantic_annotation={"firstFrame":None,"lastFrame":True}
            pos_constraint = {"joint":root_joint_name,"position":goal,
                      "semanticAnnotation":pos_semantic_annotation}
            constraints.append(pos_constraint)

        if settings["use_dir_vector_constraints"] :
            rot_semantic_annotation={"firstFrame":None,"lastFrame":True}
            rot_constraint = {"joint":root_joint_name, "dir_vector":dir_vector,
                          "semanticAnnotation":rot_semantic_annotation}
            constraints.append(rot_constraint)

    if len(keyframe_constraints.keys()) > 0:
        # extract keyframe constraints of the current state
        if current_state in keyframe_constraints.keys():
            constraints+= keyframe_constraints[current_state]
            
        # generate frame constraints for the last step basd on the previous state
        # if not already done for the trajectory following
        if not pose_constraint_set and is_last_step and prev_frames != None:
            frame_constraint= create_frame_constraint(bvh_reader,node_name_map,prev_frames)
            constraints.append(frame_constraint)
  
        
    use_optimization = len(keyframe_constraints.keys()) > 0 or is_last_step
    return constraints, arc_length, use_optimization 



def extract_trajectory_from_constraint_list(constraint_list,joint_name):
    """ Extract the trajectory information from the constraints and constructs
        a trajectory as an CatmullRomSpline instance.
    Returns:
    -------
    * trajectory: CatmullRomSpline
        Spline parameterized by arc length.
    * unconstrained_indices: list of indices
        Lists of indices of degrees of freedom to ignore in the constraint evaluation.
    """
    trajectory_constraint = extract_trajectory_constraint(constraint_list,joint_name)
    if  trajectory_constraint is not None:
        #print "found trajectory constraint"
        return create_trajectory_from_constraint(trajectory_constraint)
    else:
        return None, None

def extract_constraints_of_elementary_action(bvh_reader, morphable_subgraph, constraint_list):
    """ Extracts keyframe and trajectory constraints from constraint_list
    Returns:
    -------
    * trajectory: CatmullRomSpline
        Spline parameterized by arc length.
    * unconstrained_indices: list of indices
        lists of indices of degrees of freedom to ignore in the constraint evaluation.
    * keyframe_constraints: dict of lists
        Lists of constraints for each motion primitive in the subgraph.
    """
    root_joint_name = bvh_reader.root# currently only trajectories on the Hips joint are supported
    trajectory, unconstrained_indices = extract_trajectory_from_constraint_list(constraint_list, root_joint_name)

    keyframe_constraints = extract_all_keyframe_constraints(constraint_list,
                                                            morphable_subgraph)
    keyframe_constraints = prepare_keyframe_constraints_for_motion_primitves(morphable_subgraph,
                                                                             keyframe_constraints)
    return trajectory,unconstrained_indices, keyframe_constraints
    
    
def get_random_start_state(morphable_graph,elementary_action, prev_action_name, prev_mp_name):
    """ Get random start state based on edge from previous elementary action if possible
    """
    next_state = ""
    if prev_action_name in morphable_graph.subgraphs.keys() and \
           prev_mp_name in morphable_graph.subgraphs[prev_action_name].nodes.keys():
                               
       to_key = morphable_graph.subgraphs[prev_action_name]\
                       .nodes[prev_mp_name].generate_random_action_transition(elementary_action)
       if to_key is not None:
           next_state = to_key.split("_")[1]
           return next_state
       else:
           return None
       print "generate start from transition of last action", prev_action_name, prev_mp_name, to_key
       
    # if there is no previous elementary action or no action transition
    #  use transition to random start state
    if next_state == "" or next_state not in morphable_graph.subgraphs[elementary_action].nodes.keys():
        print next_state,"not in", elementary_action,prev_action_name,prev_mp_name
        next_state = morphable_graph.subgraphs[elementary_action].get_random_start_state()
        print "generate random start",next_state
    return next_state
    
def get_random_transition_state(morphable_subgraph, prev_state, prev_frames, trajectory, travelled_arc_length, arc_length_of_end):
    """ Get next state of the elementary action based on previous iteration.
    """
    if trajectory is not None :
            
         #test end condition for trajectory constraints
        if not check_end_condition(morphable_subgraph,prev_frames,trajectory,\
                                travelled_arc_length,arc_length_of_end) :            

            #make standard transition to go on with trajectory following
            next_state_type = NODE_TYPE_STANDARD
        else:
            # threshold was overstepped. remove previous step before 
            # trying to reach the goal using a last step
            #TODO replace with more efficient solution or optimization
#                    deleted_state,deleted_travelled_arc_length,deleted_number_of_frames = mp_sequence.pop(-1)
#                    current_state,travelled_arc_length,prev_number_of_frames = mp_sequence.pop(-1)
#                    euler_frames = euler_frames[:prev_number_of_frames]
#                    print "deleted data for state",deleted_state
            next_state_type = NODE_TYPE_END
            
        print "generate",next_state_type,"transition from trajectory"
    else:
        n_standard_transitions = len([e for e in morphable_subgraph.nodes[prev_state].outgoing_edges.keys() if morphable_subgraph.nodes[prev_state].outgoing_edges[e].transition_type == "standard"])
        if n_standard_transitions > 0:
            next_state_type = NODE_TYPE_STANDARD
        else:
            next_state_type = NODE_TYPE_END
        print "generate",next_state_type,"transition without trajectory",n_standard_transitions

    to_key = morphable_subgraph.nodes[prev_state].generate_random_transition(next_state_type)
    
    if to_key is not None:
        current_state = to_key.split("_")[1]
        return current_state, next_state_type
    else:
        return None, next_state_type
       
    

def convert_elementary_action_to_motion(elementary_action,
                                        constraint_list,
                                        morphable_graph,
                                        options=None,
                                        prev_action_name="",
                                        prev_mp_name="",
                                        prev_frames=None,
                                        prev_parameters=None,
                                        bvh_reader=None,
                                        node_name_map=None,
                                        start_pose=None,
                                        step_count=0,
                                        max_step=-1,
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
    * elementary_action : string
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
    quat_frames = prev_frames
    morphable_subgraph = morphable_graph.subgraphs[elementary_action]
    trajectory_following_settings = options["trajectory_following_settings"]#  TODO move trajectory_following_settings to different key of the options

    trajectory,unconstrained_indices, keyframe_constraints = extract_constraints_of_elementary_action(bvh_reader, morphable_subgraph, constraint_list)
    arc_length_of_end = morphable_subgraph.nodes[morphable_subgraph.get_random_end_state()].average_step_length
#    number_of_standard_transitions = len([n for n in \
#                                 morphable_subgraph.nodes.keys() if morphable_subgraph.nodes[n].node_type == "standard"])
#
    #create sequence of list motion primitives,arc length and number of frames for backstepping 
    mp_sequence = [ (prev_mp_name,0,0) ]
    current_state = None
    current_state_type = ""
    temp_step = 0
    travelled_arc_length = 0.0
    print "start converting elementary action",elementary_action
    while current_state_type != NODE_TYPE_END:

        if max_step > -1 and step_count + temp_step > max_step:
            print "reached max step"
            break
        #######################################################################
        # Get motion primitive = extract from graph based on previous last step + heuristic
        if temp_step == 0:  
             current_state = get_random_start_state(morphable_graph,elementary_action, prev_action_name, prev_mp_name)
             if current_state is None:
                 print "Error: Could not find a transition of type action_transition from ",prev_action_name,prev_mp_name ," to state",current_state
                 break
        elif len(morphable_subgraph.nodes[current_state].outgoing_edges) > 0:
            prev_state = current_state
            current_state, current_state_type = get_random_transition_state(morphable_subgraph, prev_state, quat_frames, trajectory, travelled_arc_length, arc_length_of_end)
            if current_state is None:
                 print "Error: Could not find a transition of type",current_state_type,"from state",prev_state
                 break
        else:
            print "Error: Could not find a transition from state",current_state
            break

        print "transitioned to state",current_state
        #######################################################################
        #Generate constraints from action_constraints
        if quat_frames is None:
            prev_pos = start_pose["position"]  
        else:
            prev_pos = quat_frames[-1][:3]

        try: 
            constraints, temp_arc_length, use_optimization = create_constraints_for_motion_primitive(morphable_subgraph,\
                      current_state,trajectory,travelled_arc_length,prev_pos,unconstrained_indices,trajectory_following_settings,\
                      bvh_reader.root,quat_frames,bvh_reader,node_name_map,keyframe_constraints,is_last_step=(current_state_type == NODE_TYPE_END) )
        except PathSearchError as e:
                print "moved beyond end point using parameters",
                str(e.search_parameters)
                break
        # get optimal parameters, Back-project to frames in joint angle space,
        # Concatenate frames to motion and apply smoothing

        options_copy = copy.copy(options)
        options_copy["use_optimization"] = use_optimization

        quat_frames, parameters, tmp_action_list = get_optimal_motion(morphable_graph,elementary_action,current_state,constraints,\
                                            prev_action_name=prev_action_name, prev_mp_name=mp_sequence[-1][0], prev_frames=quat_frames, prev_parameters=prev_parameters,\
                                            options=options_copy, bvh_reader=bvh_reader,node_name_map=node_name_map,\
                                            start_pose=start_pose,keyframe_annotations=keyframe_annotations,verbose=verbose)                                            
        action_list = merge_two_dicts(action_list,tmp_action_list )      
        if trajectory is not None:
            #update arc length based on new closest point
            closest_point,distance = trajectory.find_closest_point(quat_frames[-1][:3],min_arc_length = mp_sequence[-1][1])
            travelled_arc_length,eval_point = trajectory.get_absolute_arc_length_of_point(closest_point,min_arc_length=travelled_arc_length)
            if travelled_arc_length == -1 :
                travelled_arc_length = trajectory.full_arc_length

        prev_parameters = parameters
        prev_action_name = elementary_action
        mp_sequence.append( (current_state,travelled_arc_length,len(quat_frames)) )
        temp_step += 1

    step_count += temp_step

    print "reached end of elementary action", elementary_action

    if trajectory is not None:
        print "info", trajectory.full_arc_length, \
               travelled_arc_length,arc_length_of_end, \
               np.linalg.norm(trajectory.get_last_control_point() - quat_frames[-1][:3]), \
               check_end_condition(morphable_subgraph,quat_frames,trajectory,\
                                        travelled_arc_length,arc_length_of_end)
    return quat_frames, prev_action_name, mp_sequence[-1][0], prev_parameters, step_count, action_list


def transform_from_left_to_right_handed_cs(start_pose):
    """ Transform transition and rotation of the start pose from CAD to Opengl 
        coordinate system.
    """
    start_pose_copy = copy.copy(start_pose)
    start_pose["orientation"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["orientation"])
    start_pose["position"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["position"])
    return start_pose

def convert_elementary_action_list_to_motion(morphable_graph,elementary_action_list,\
    options=None, bvh_reader=None,node_name_map=None, \
     max_step=-1, start_pose=None, keyframe_annotations={},verbose=False):
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
    if options is None:
        options = generate_algorithm_settings()
    if verbose:
        print_options(options)
        print "max_step", max_step
    if start_pose is not None:
        start_pose = transform_from_left_to_right_handed_cs(start_pose)
        print "transform start pose",start_pose
        
    action_list = {}
    frame_annotation = {}
    frame_annotation['elementaryActionSequence'] = []
    quat_frames = None
    prev_parameters = None
    prev_action_name = ""
    prev_mp_name = ""
    step_count = 0
    for action_index in range(len(elementary_action_list)):
        if max_step > -1 and step_count > max_step:
            print "reached max step"
            break
        action = elementary_action_list[action_index]["action"]
        if verbose:
            print "convert",action,"to graph walk"

        action_annotations = keyframe_annotations[action_index]
        constraints = elementary_action_list[action_index]["constraints"]

        if quat_frames is not None:
            start_frame=len(quat_frames)            
        else:
            start_frame=0

        try:
            quat_frames,prev_action_name,prev_mp_name,prev_parameters,step_count,tmp_action_list = convert_elementary_action_to_motion(action, constraints,
                                                                                                 morphable_graph,options=options,
                                                                                                 prev_action_name=prev_action_name, prev_mp_name=prev_mp_name,
                                                                                                 prev_frames=quat_frames,
                                                                                                 prev_parameters=prev_parameters,
                                                                                                 bvh_reader=bvh_reader,node_name_map=node_name_map,
                                                                                                 start_pose=start_pose, step_count=step_count, max_step=max_step,
                                                                                                 keyframe_annotations=action_annotations, verbose=verbose)                                                      
            action_list = merge_two_dicts(action_list,tmp_action_list)

        except SynthesisError as e:
            print "Arborting conversion",e.message
            return e.euler_frames,frame_annotation,action_list

        #update frame annotation
        action_frame_annotation = {}
        action_frame_annotation["startFrame"] = start_frame
        action_frame_annotation["elementaryAction"] = action
        action_frame_annotation["endFrame"] = len(quat_frames)-1
        frame_annotation['elementaryActionSequence'].append(action_frame_annotation)

    return quat_frames, frame_annotation, action_list


