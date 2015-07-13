# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 17:15:22 2015

Implements functions used for the processing of the constraints from the input file
generated by CNL processing.

@author: erhe01
"""

import collections
from math import sqrt
import numpy as np
from utilities.io_helper_functions import load_json_file
from utilities.parameterized_spline import ParameterizedSpline, plot_splines
from utilities.motion_editing import euler_to_quaternion
from cgkit.cgtypes import quat

TRAJECTORY_DIM = 3 # spline in cartesian space

def transform_point_from_cad_to_opengl_cs(point):
    """ Transforms a 3d point represented as a list from left handed to the 
        right handed coordinate system
    """

    transform_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    return np.dot(transform_matrix,point).tolist()
    
def transform_unconstrained_indices_from_cad_to_opengl_cs(unconstrained_indices):
    new_indices = []
    for i in unconstrained_indices:
        if i == 0:
            new_indices.append(i)
        elif i == 1:
            new_indices.append(2)
        elif i == 2:
            new_indices.append(1)
    return new_indices
    
    
def cgkit_mat_to_numpy4x4(matrix):
    """ Converts to cgkit matrix a numpy matrix  """
    return np.array(matrix.toList(), np.float32).reshape(4,4)    

def transform_point(transformation_matrix,point):
    """ Transforms a 3d point represented as a list by a numpy transformation
    
    Parameters
    ----------
    *transformation_matrix: np.ndarray
    \tHomogenous 4x4 transformation matrix
    *point: list
    \tCartesian coordinates
    
    Returns
    -------
    * point: list
    \tThe transformed point as a list
    """
    return np.dot(transformation_matrix,np.array(point+[1,]))[:3].tolist()   


    
def extract_start_transformation(mg_input) :   
    """ Create a mapping from action name to index in the elementary action list.
    Parameters
    ----------
    *mg_input: dict
    \t The dictionary read from the Morphable Graphs input json file.
    
    Returns
    -------
    *transformation_matrix: np.ndarray
    \tHomogenous 4x4 transformation matrix.
    """
    assert "startPose" in mg_input.keys()
    translation = mg_input["startPose"]["position"]
    euler_angles = mg_input["startPose"]["orientation"]
    q = euler_to_quaternion(euler_angles,['Xrotation','Yrotation','Zrotation'])
    matrix = cgkit_mat_to_numpy4x4(quat(q).toMat4())
    matrix[:,3] = translation+[1]
    #print matrix
    return matrix
    
    
def extract_elementary_actions(mg_input):
    """ Create a mapping from action name to index in the elementary action list.
    Parameters
    ----------
    *mg_input: dict
    \t The dictionary read from the Morphable Graphs input json file.
    
    Returns
    -------
    *action_dict: OrderedDict
    \t A dictionary that maps actions to indices in the elementary action list
    """
    action_dict = collections.OrderedDict()
    index = 0
    for e in mg_input["elementaryActions"]:
        action_dict[e["action"]] = index
        index += 1
    return action_dict


def extract_keyframe_annotations(elementary_action_list):
    """
    Returns
    ------
    * keyframe_annotations : a list of dicts
      Contains for every elementary action a dict that associates of events/actions with certain keyframes
    """
    keyframe_annotations = []
    for entry in elementary_action_list:
        print  "entry#################",entry
        if "keyframeAnnotations" in entry.keys():
            annotations = {}
       
            for annotation in entry["keyframeAnnotations"]:
                key = annotation["keyframe"]
                annotations[key] = annotation
            keyframe_annotations.append(annotations)
        else:
            keyframe_annotations.append({})
    return keyframe_annotations
    
def extract_trajectory_constraints_for_plotting(mg_input,action_index,scale_factor= 1.0):
    """ Extracts 2d control points for trajectory constraints for a given action
    
    Parameters
    ----------
    *mg_input: dict
    \tElementary action list with constraints read from the json input file
    *action_index: integer
    \tIndex of an entry in the elementary action
    *scale_factor: float
    \tIs applied on cartesian coordinates
    
    Returns
    -------
    *control_points: dict of lists
    \t list of 2d control points for each joint
    """
    assert action_index < len(mg_input["elementaryActions"])
    inv_start_transformation = np.linalg.inv(extract_start_transformation(mg_input)   )
    constraints_list = mg_input["elementaryActions"][action_index]["constraints"]
    control_points = {}
    for entry in constraints_list:
        joint_name = entry["joint"]
        if "trajectoryConstraints" in entry.keys():
            control_points[joint_name] = []
            control_points[joint_name].append([0,0,0])#add origin as point
            for c in entry["trajectoryConstraints"]:
                #point = [ p*scale_factor  for p in c["position"] if p!= None]
                point = [ p*scale_factor if p!= None else 0 for p in c["position"] ]
                point = transform_point(inv_start_transformation,point)
                point = [point[0],point[2]]
                control_points[joint_name].append(point)
    return control_points



def construct_trajectories_for_plotting(mg_input,scale_factor):
    """Calls extract_trajectory_constraints for each action in the input 
    and creates a spline using the ParameterizedSpline class
    """
    elementary_action_dict = extract_elementary_actions(mg_input)
    
    traj_constraints = collections.OrderedDict()
    for key in elementary_action_dict.keys():
        control_points = extract_trajectory_constraints_for_plotting(mg_input,elementary_action_dict[key],scale_factor)
        #print control_points        
        traj_constraints[key] = {}
        for joint_name in control_points.keys():
            traj_constraints[key][joint_name] = ParameterizedSpline(control_points[joint_name],2)        
    return traj_constraints

def plot_trajectory_from_mg_input_file(filename,scale_factor = 1.0):
    """ Reads the Morphable Graphs input file and plots trajectories for testing.
    
     Parameters
    ----------
    * filename: string
    \tThe path to the saved json file.
    * scalefactor: float
    \tIs applied on cartesian coordinates
    """
    
    mg_input = load_json_file(filename)
    traj_constraints = construct_trajectories_for_plotting(mg_input,scale_factor)
    plot_splines("Trajectories",traj_constraints["walk"].values())
  
def extract_root_positions(euler_frames):
    roots_2D = []
    for i in xrange(len(euler_frames)):
        position_2D = np.array([ euler_frames[i][0],euler_frames[i][1], euler_frames[i][2] ])
        #print "sample",position2D
        roots_2D.append(position_2D)
    return np.array(roots_2D) 
        
   
def extract_trajectory_constraints(constraints_list,scale_factor= 1.0):
    """ Extracts the control points for trajectory constraints for a given action.
        Components containing None are set to 0, but marked as ignored in the unconstrained_indices list.
        Note all elements in constraints_list must have the same dimensions constrained and unconstrained.
    
    Parameters
    ----------
    * constraints_list : dict
      Elementary action list with constraints read from the json input file
    * action_index : integer
      Index of an entry in the elementary action
    * scale_factor : float
      Is applied on cartesian coordinates.

    Returns
    -------
    * control_points : dict of lists
      list of control points for each joint
    * unconstrained_indices : dict of lists
      indices that should be ignored
    """
    unconstrained_indices = {}
  
    #create a control point list that can be used as input for the ParameterizedSpline class   
    control_points = {}
    for entry in constraints_list:
        joint_name = entry["joint"]
        if "trajectoryConstraints" in entry.keys():
            assert len(entry["trajectoryConstraints"])>0  and "position" in entry["trajectoryConstraints"][0].keys()
           
            #extract unconstrained dimensions
            unconstrained_indices[joint_name] = []
            idx = 0
            for v in entry["trajectoryConstraints"][0]["position"]:
                if v == None:
                    unconstrained_indices[joint_name].append(idx)
                idx += 1            
            
            control_points[joint_name] = []
            for c in entry["trajectoryConstraints"]:
                point = [ p*scale_factor if p!= None else 0 for p in c["position"] ]# else 0  where the array is None set it to 0
                control_points[joint_name].append(point)
    return control_points, unconstrained_indices
       
def get_arc_length_from_points(points):
    """
    Note: accuracy depends on the granulariy of points
    """
    arc_length = 0.0
    last_p = None
    for p in points:
        if last_p is not None:
            delta = p - last_p
            #print delta
            arc_length += sqrt( delta[0]**2 + delta[1]**2 +delta[2]**2) #-arcLength
        else:
            delta = p
        last_p = p            
    return arc_length

def get_step_length_for_sample(motion_primitive, s, method = "arc_length"):
    """Compute step length from a list of euler frames
    """
    # get quaternion frames from s_vector
    quat_frames = motion_primitive.back_project(s,use_time_parameters = False).get_motion_vector()
    if method == "arc_length":
        root_pos = extract_root_positions(quat_frames)        
        step_length = get_arc_length_from_points(root_pos)
    elif method == "distance":
        root_pos = extract_root_positions(quat_frames)  
        step_length = np.linalg.norm(root_pos[-1][:3] - root_pos[0][:3])
    else:
        raise NotImplementedError
    return step_length
  
def get_step_length(morphable_subgraph,motion_primitive_name,method = "arc_length"):
    """Backproject the motion and get the step length and the last keyframe on the canonical timeline
    Parameters
    ----------
    * morphable_subgraph : MorphableSubgraph
      Represents an elementary action
    * motion_primitive_name : string
      Identifier of the morphable model
    * method : string
      Can have values arc_length or distance. If any other value distance is used.
    Returns
    -------
    *step_length: float
    \tThe arc length of the path of the motion primitive
    """
    assert motion_primitive_name in morphable_subgraph.nodes.keys()
    step_length = 0

    current_parameters = morphable_subgraph.nodes[motion_primitive_name].sample_parameters()
    quat_frames = morphable_subgraph.nodes[motion_primitive_name].mp.back_project(current_parameters,use_time_parameters = False).get_motion_vector()
    if method == "arc_length":
        root_pos = extract_root_positions(quat_frames)
        #print root_pos
        step_length = get_arc_length_from_points(root_pos)
    else:# use distance
        vector = quat_frames[-1][:3] - quat_frames[0][:3] 
        magnitude = 0
        for v in vector:
            magnitude += v**2
        step_length = sqrt(magnitude)
    #print "step length",step_length
    #print "step length####################",motion_primitive_name,step_length
    return step_length

def create_constraint(joint_name,position=[None,None,None],orientation=[None,None,None],semanticAnnotation=None):
    """ Wrapper around a dict object creation
    Returns 
    -------
    * constraint : dict
      A dict containing joint, position,orientation and semanticAnnotation describing a constraint
    """
    constraint = {"joint":joint_name,"position":position,"orientation":[None,None,None],"semanticAnnotation":semanticAnnotation} # (joint, [pos_x, pos_y, pos_z],[rot_x, rot_y, rot_z])
    return constraint

    
def create_trajectory_from_constraint(trajectory_constraint,scale_factor = 1.0):
    """ Create a spline based on a trajectory constraint. 
        Components containing None are set to 0, but marked as ignored in the unconstrained_indices list.
        Note all elements in constraints_list must have the same dimensions constrained and unconstrained.
    
    Parameters
    ----------
    * trajectory_constraint: list of dict
    \t Defines a list of control points
    * scale_factor: float
    \t Scaling applied on the control points
    
    Returns
    -------
    * trajectory: ParameterizedSpline
    \t The trajectory defined by the control points from the trajectory_constraint
    * unconstrained_indices : list
    \t List of indices of unconstrained dimensions
    """
   
    
    assert len(trajectory_constraint)>0  and "position" in trajectory_constraint[0].keys()
           
    #extract unconstrained dimensions
    unconstrained_indices = []
    idx = 0
    for v in trajectory_constraint[0]["position"]:
        if v == None:
            unconstrained_indices.append(idx)
        idx += 1            
    unconstrained_indices = transform_unconstrained_indices_from_cad_to_opengl_cs(unconstrained_indices)
    #create control points by setting constrained dimensions to 0
    control_points = []

    for c in trajectory_constraint:
        point = [ p*scale_factor if p is not None else 0 for p in c["position"] ]# else 0  where the array is None set it to 0
        point = transform_point_from_cad_to_opengl_cs(point)
        control_points.append(point)
    #print "####################################################"
    #print "control points are: "
    #print control_points
    trajectory =  ParameterizedSpline(control_points, TRAJECTORY_DIM)
    
    return trajectory, unconstrained_indices
  


def extract_trajectory_constraint(constraints,joint_name = "Hips"):
    """Returns a single trajectory constraint definition for joint joint out of a elementary action constraint list
    """
    for c in constraints:
        if joint_name == c["joint"]: 
            if "trajectoryConstraints" in c.keys():
                return c["trajectoryConstraints"]
    return None
    
    
def constraint_definition_has_label(constraint_definition,label):
    """ Checks if the label is in the semantic annotation dict of a constraint
    """
    #print "check######",constraint_definition
    if "semanticAnnotation" in constraint_definition.keys():
        annotation = constraint_definition["semanticAnnotation"]
        #print "semantic Annotation",annotation
        if label in annotation.keys():
            return True
    return False
        
def extract_keyframe_constraints_for_label(constraint_list,label):
    """ Returns the constraints associated with the given label. Ordered 
        based on joint names.
    Returns
    ------
    * key_constraints : dict of lists
    \t contains the list of the constrained joints
    """
    key_constraints= {}
    for c in constraint_list:
        joint_name = c["joint"]
        #print "read constraint"
        #print c
        if "keyframeConstraints" in c.keys():# there are keyframe constraints
            key_constraints[joint_name] = []
            for constraint_definition in c["keyframeConstraints"]:
                print "read constraint",constraint_definition
                if constraint_definition_has_label(constraint_definition,label):
                    key_constraints[joint_name].append(constraint_definition)
    return key_constraints
                
def extract_all_keyframe_constraints(constraint_list,morphable_subgraph):
    """Orders the keyframe constraint for the labels found in the metainformation of
       the elementary actions based on labels as keys
    Returns
    -------
    * keyframe_constraints : dict of dict of lists
      access as keyframe_constraints["label"]["joint"][index]
    """
    keyframe_constraints = {}
    annotations = morphable_subgraph.annotation_map.keys()#["start_contact",]
    for label in annotations:
#        print "extract constraints for annotation",label
        keyframe_constraints[label] = extract_keyframe_constraints_for_label(constraint_list,label)
        #key_frame_constraints = extract_keyframe_constraints(constraints,annotion)
    return keyframe_constraints


    
def extract_key_frame_constraint(joint_name,constraint,morphable_subgraph,time_information):
    """ Creates a dict containing all properties stated explicitly or implicitly in the input constraint
    Parameters
    ----------
    * joint_name : string
      Name of the joint
    * constraint : dict
      Read from json input file 
    * time_information : string
      Time information corresponding to an annotation read from morphable graph meta information 
      
     Returns
     -------
     *constraint_desc : dict
      Contains the keys joint, position, orientation, semanticAnnotation
    """
   
    position = [None, None, None]       
    orientation = [None, None, None]
    first_frame = None
    last_frame = None
    if "position" in constraint.keys():
         position = constraint["position"]
    if "orientation" in constraint.keys():
        orientation =constraint["orientation"]
    #check if last or fist frame from annotation

    position = transform_point_from_cad_to_opengl_cs(position)
 
    if time_information == "lastFrame":
        last_frame = True
    elif time_information == "firstFrame":
        first_frame = True
    if "semanticAnnotation" in constraint.keys():
        semanticAnnotation = constraint["semanticAnnotation"]
    else:
        semanticAnnotation = {}
        
    semanticAnnotation["firstFrame"] = first_frame
    semanticAnnotation["lastFrame"] = last_frame
    constraint_desc = {"joint":joint_name,"position":position,"orientation":orientation,"semanticAnnotation": semanticAnnotation}
    return constraint_desc

