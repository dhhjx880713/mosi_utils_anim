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
from lib.morphable_graph import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from utilities.motion_editing import convert_quaternion_to_euler, \
                                get_cartesian_coordinates2, \
                                transform_quaternion_frames, \
                                fast_quat_frames_alignment
from lib.input_processing import create_trajectory_from_constraint,\
                                    extract_all_keyframe_constraints,\
                                    extract_trajectory_constraint,\
                                    extract_key_frame_constraint,\
                                    transform_point_from_cad_to_opengl_cs
from constrain_motion import get_optimal_parameters,\
                             generate_algorithm_settings
from constrain_gmm import ConstraintError



class SynthesisError(Exception):
    def __init__(self,  quat_frames, bad_samples):
        message = "Could not process input file"
        super(SynthesisError, self).__init__(message)
        self.bad_samples = bad_samples
        self.quat_frames = quat_frames


class PathSearchError(Exception):
    def __init__(self, parameters):
        self.search_parameters = parameters
        message = "Error in the navigation goal generation"
        super(PathSearchError, self).__init__(message)



class GraphWalkEntry(object):
    def __init__(self, action_name, motion_primitive_name, parameters, arc_length):
        self.action_name = action_name
        self.motion_primitive_name = motion_primitive_name
        self.parameters = parameters
        self.arc_length = arc_length


class AnnotatedMotion(object):
    def __init__(self):
        self.action_list = {}
        self.frame_annotation = {}
        self.frame_annotation['elementaryActionSequence'] = []
        self.graph_walk = []
        self.quat_frames = None
        self.step_count = 0
        self.n_frames = 0


    def update_action_list(self, tmp_action_list):
        self.action_list = merge_two_dicts(self.action_list,tmp_action_list )    
        
    def update_frame_annotation(self,action_name, start_frame, end_frame):
            #update frame annotation
        action_frame_annotation = {}
        action_frame_annotation["startFrame"] =  start_frame
        action_frame_annotation["elementaryAction"] = action_name
        action_frame_annotation["endFrame"] = end_frame
        self.frame_annotation['elementaryActionSequence'].append(action_frame_annotation)  




def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.
    source: http://stackoverflow.com/questions/38987/how-can-i-merge-two-python-dictionaries-in-a-single-expression
    '''
    z = x.copy()
    z.update(y)
    return z

def transform_from_left_to_right_handed_cs(start_pose):
    """ Transform transition and rotation of the start pose from CAD to Opengl 
        coordinate system.
    """
    start_pose_copy = copy.copy(start_pose)
    start_pose["orientation"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["orientation"])
    start_pose["position"] = transform_point_from_cad_to_opengl_cs(start_pose_copy["position"])
    return start_pose

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


def get_optimal_motion(morphable_graph,
                       action_name,
                       mp_name,
                       constraints,
                       algorithm_config,
                       prev_motion,
                       bvh_reader=None,
                       node_name_map=None,
                       start_pose=None,
                       keyframe_annotations={},
                       verbose=False):
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
    * euler_frames : list of np.ndarray
        list of skeleton pose parameters.
    * parameters : np.ndarray
        low dimensional motion parameters used to generate the frames
    * step_length : float
       length of the generated motion
    * action_list :
    """
    apply_smoothing = algorithm_config["apply_smoothing"]

    try:
        if len(prev_motion.graph_walk)> 0:
            prev_action_name = prev_motion.graph_walk[-1].action_name
            prev_mp_name =  prev_motion.graph_walk[-1].motion_primitive_name
            prev_parameters =  prev_motion.graph_walk[-1].parameters
            prev_frames = prev_motion.quat_frames
        else:
            prev_action_name = ""
            prev_mp_name =  ""
            prev_parameters =  None
            prev_frames = None
        parameters = get_optimal_parameters(morphable_graph,
                                            action_name,
                                            mp_name,
                                            constraints,
                                            algorithm_config=algorithm_config,
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
        
        
    if prev_frames is not None:
        frame_offset = len(prev_frames)
    else:
        frame_offset = 0
        
    use_time_parameters = True
    quat_frames = get_aligned_frames(morphable_graph, action_name, mp_name,
                                     parameters, prev_frames, start_pose,
                                     use_time_parameters, apply_smoothing)

#            print 'length of quat frames in get optimal motion from no previous frames: ' + str(len(quat_frames))
    #associate keyframe annotation to quat_frames
            
    if mp_name in morphable_graph.subgraphs[action_name].mp_annotations.keys():
        time_information = morphable_graph.subgraphs[action_name].mp_annotations[mp_name]
    else:
        time_information = {}
    action_list = get_action_list(quat_frames,time_information,constraints,keyframe_annotations,offset=frame_offset)
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


def make_guess_for_goal_arc_length(morphable_subgraph, current_motion_primitive, trajectory,
                                   last_arc_length, last_pos, unconstrained_indices=None,
                                    step_length_factor=1.0, method = "arc_length"):
    """ Makes a guess for a reachable arc length based on the current position.
        It searches for the closest point on the trajectory, retrieves the absolute arc length
        and its the arc length of a random sample of the next motion primitive
    Returns
    -------
    * arc_length : float
      The absolute arc length of the new goal on the trajectory.
      The goal should then be extracted using get_point_and_orientation_from_arc_length
    """
    if unconstrained_indices is None:
        unconstrained_indices = []
    step_length = morphable_subgraph.nodes[current_motion_primitive].average_step_length * step_length_factor
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

def create_constraints_for_motion_primitive(morphable_subgraph,current_motion_primitive,trajectory,
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
                                                   current_motion_primitive,trajectory,
                                                   last_arc_length,last_pos,
                                                   unconstrained_indices,
                                                   step_length_factor,method)
        else:
            arc_length = trajectory.full_arc_length

        goal,orientation,dir_vector = get_point_and_orientation_from_arc_length(trajectory,arc_length,unconstrained_indices)

#        print  "starting from",last_pos,last_arc_length,"the new goal for", \
#                current_motion_primitive,"is",goal,"at arc length",arc_length
        print "starting from: "
        print last_pos
        print "the new goal for " + current_motion_primitive
        print goal
        print "arc length is: " + str(arc_length)



        if settings["use_frame_constraints"] and  prev_frames is not None and bvh_reader is not None and node_name_map is not None:
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
        if current_motion_primitive in keyframe_constraints.keys():
            constraints+= keyframe_constraints[current_motion_primitive]
            
        # generate frame constraints for the last step basd on the previous state
        # if not already done for the trajectory following
        if not pose_constraint_set and is_last_step and prev_frames is not None:
            frame_constraint= create_frame_constraint(bvh_reader,node_name_map,prev_frames)
            constraints.append(frame_constraint)
  
        
    use_optimization = len(keyframe_constraints.keys()) > 0 or is_last_step
    return constraints, arc_length, use_optimization 



def extract_trajectory_from_constraint_list(constraint_list,joint_name):
    """ Extract the trajectory information from the constraints and constructs
        a trajectory as an ParameterizedSpline instance.
    Returns:
    -------
    * trajectory: ParameterizedSpline
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
    * trajectory: ParameterizedSpline
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
       

def convert_elementary_action_to_motion(action_name,
                                        constraint_list,
                                        morphable_graph,
                                        algorithm_config,
                                        motion,
                                        bvh_reader=None,
                                        node_name_map=None,
                                        start_pose=None,
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
   *  quat_frames : np.ndarray
    * last_action_name : string
    * last_mp_name : string
    * last_parameters : np.ndarray
    * step_count : integer
    * action_list : dict of lists of dict
     \t euler_frames + information for the transiton to another elementary action
    """
    
    #motion = prev_motion
    
    start_frame = motion.n_frames    

    morphable_subgraph = morphable_graph.subgraphs[action_name]
    trajectory_following_settings = algorithm_config["trajectory_following_settings"]#  TODO move trajectory_following_settings to different key of the algorithm_config
    trajectory,unconstrained_indices, keyframe_constraints = \
    extract_constraints_of_elementary_action(bvh_reader, morphable_subgraph, constraint_list)
    arc_length_of_end = morphable_subgraph.nodes[morphable_subgraph.get_random_end_state()].average_step_length
    
#    number_of_standard_transitions = len([n for n in \
#                                 morphable_subgraph.nodes.keys() if morphable_subgraph.nodes[n].node_type == "standard"])
#
    #create sequence of list motion primitives,arc length and number of frames for backstepping 
    current_motion_primitive = None
    current_motion_primitive_type = ""
    temp_step = 0
    travelled_arc_length = 0.0
    print "start converting elementary action",action_name
    while current_motion_primitive_type != NODE_TYPE_END:

        if max_step > -1 and motion.step_count + temp_step > max_step:
            print "reached max step"
            break
        #######################################################################
        # Get motion primitive = extract from graph based on previous last step + heuristic
        if temp_step == 0:  
             current_motion_primitive = get_random_start_state(motion, morphable_graph, action_name)
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
            current_motion_primitive, current_motion_primitive_type = get_random_transition_state(motion, morphable_subgraph, trajectory, travelled_arc_length, arc_length_of_end)
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
            last_pos = start_pose["position"]  
        else:
            last_pos = motion.quat_frames[-1][:3]

        try: 
            constraints, temp_arc_length, use_optimization = create_constraints_for_motion_primitive(morphable_subgraph,\
                      current_motion_primitive,trajectory,travelled_arc_length,last_pos,unconstrained_indices,trajectory_following_settings,\
                      bvh_reader.root,motion.quat_frames,bvh_reader,node_name_map,keyframe_constraints,is_last_step=(current_motion_primitive_type == NODE_TYPE_END) )
        except PathSearchError as e:
                print "moved beyond end point using parameters",
                str(e.search_parameters)
                break
        # get optimal parameters, Back-project to frames in joint angle space,
        # Concatenate frames to motion and apply smoothing

        algorithm_config_copy = copy.copy(algorithm_config)
        algorithm_config_copy["use_optimization"] = use_optimization

        motion.quat_frames, parameters, tmp_action_list = get_optimal_motion(morphable_graph, action_name, current_motion_primitive, constraints,\
                                                            prev_motion=motion, algorithm_config=algorithm_config_copy, bvh_reader=bvh_reader,node_name_map=node_name_map,\
                                                            start_pose=start_pose,keyframe_annotations=keyframe_annotations,verbose=verbose)                                            

        
        #update arc length based on new closest point
        if trajectory is not None:
            if motion.step_count > 0:
                min_arc_length = motion.graph_walk[-1].arc_length
            else:
                min_arc_length = 0.0
            closest_point,distance = trajectory.find_closest_point(motion.quat_frames[-1][:3],min_arc_length=min_arc_length)
            travelled_arc_length,eval_point = trajectory.get_absolute_arc_length_of_point(closest_point,min_arc_length=travelled_arc_length)
            if travelled_arc_length == -1 :
                travelled_arc_length = trajectory.full_arc_length

        motion.update_action_list(tmp_action_list)
        graph_walk_entry = GraphWalkEntry(action_name,current_motion_primitive, parameters, travelled_arc_length)
        motion.graph_walk.append(graph_walk_entry)

        temp_step += 1

    motion.step_count += temp_step
    motion.n_frames = len(motion.quat_frames)
    motion.update_frame_annotation(action_name, start_frame, motion.n_frames)
    
    print "reached end of elementary action", action_name

#    if trajectory is not None:
#        print "info", trajectory.full_arc_length, \
#               travelled_arc_length,arc_length_of_end, \
#               np.linalg.norm(trajectory.get_last_control_point() - quat_frames[-1][:3]), \
#               check_end_condition(morphable_subgraph,quat_frames,trajectory,\
#                                        travelled_arc_length,arc_length_of_end)
        
  
    return motion# quat_frames, prev_action_name, mp_sequence[-1][0], prev_parameters, step_count, action_list


  

def convert_elementary_action_list_to_motion(morphable_graph,elementary_action_list,\
    algorithm_config=None, bvh_reader=None,node_name_map=None, \
     max_step=-1, start_pose=None, keyframe_annotations={},verbose=False):
    """ Converts a constrained graph walk to euler frames
     Parameters
    ----------
    * morphable_graph : MorphableGraph
        Data structure containing the morphable models
    * elementary_action_list : list of dict
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
    if verbose:
        for key in algorithm_config.keys():
            print key,algorithm_config[key]
        print "max_step", max_step
    if start_pose is not None:
        start_pose = transform_from_left_to_right_handed_cs(start_pose)
        print "transform start pose",start_pose
        
    motion = AnnotatedMotion()

    for action_index in range(len(elementary_action_list)):
        if max_step > -1 and motion.step_count > max_step:
            print "reached max step"
            break
        action = elementary_action_list[action_index]["action"]
        if verbose:
            print "convert",action,"to graph walk"

        action_annotations = keyframe_annotations[action_index]
        constraints = elementary_action_list[action_index]["constraints"]

        try:
            motion = convert_elementary_action_to_motion(action, constraints, morphable_graph,algorithm_config,
                                                                  motion, bvh_reader=bvh_reader,node_name_map=node_name_map,
                                                                 start_pose=start_pose, max_step=max_step,
                                                                 keyframe_annotations=action_annotations, verbose=verbose)

        except SynthesisError as e:
            print "Arborting conversion",e.message
            return e.quat_frames,motion.frame_annotation,motion.action_list


    return motion


