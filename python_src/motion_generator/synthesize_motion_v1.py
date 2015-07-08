# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:11:11 2015#
Runs the complete Morhable Graphs Pipeline to generate a motion based on an
json input file.
Runs the optimization sequentially based on the elementary action 
breakdown.

@author: erhe01, hadu01, FARUPP , mamauer
"""

import os
from lib.bvh2 import BVHReader, create_filtered_node_name_map
from lib.morphable_graph import MorphableGraph,\
                                print_morphable_graph_structure
from lib.helper_functions import get_morphable_model_directory, \
                                 get_transition_model_directory, \
                                 load_json_file, \
                                 export_euler_frames_to_bvh
from lib.motion_editing import convert_quaternion_to_euler,align_frames, \
                                transform_euler_frames,\
                                smoothly_concatenate
                                
from constrain_motion import get_optimal_parameters,\
                                generate_algorithm_settings,\
                                print_options
from constrain_gmm import ConstraintError
from lib.graph_walk_extraction import elementary_action_breakdown,write_graph_walk_to_file



def convert_graph_walk_to_motion(morphable_graph,graph_walk,  \
    options = None,bvh_reader=None, node_name_map = None,   max_step =  -1, start_pose=None, \
    keyframe_annotations = [],verbose = False):
    """ Converts a constrained graph walk to euler frames
     Parameters
    ----------
    * morphable_graph : MorphableGraph
        Data structure containing the morphable models
    * graph_walk : list of dict
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
      A list of euler frames representing a motion.
    * frame_annotation : dict
       Associates the euler frames with the elementary actions
    * action_list : dict of dicts
     Contains actions/events for some frames based on the keyframe_annotations 
    """
    
    if options == None:
        options = generate_algorithm_settings()
    if verbose:
        print_options(options)
        print "max_step",max_step
    action_list = []
    frame_annotation = {}
    frame_annotation['elementaryActionSequence'] = []
    apply_smoothing = options["apply_smoothing"]
    step_count = 0
    prev_parameters = None
    concatenated_frames = None
    prev_action_name = ""
    prev_mp_name = ""

    for entry in graph_walk:
        if max_step > -1 and step_count > max_step:
            break
        action_name = entry["elementaryAction"]
        mp_name = entry["motionPrimitive"]
        constraints = entry["constraints"]
        action_index = entry["actionIndex"]
        
        if action_index < len(keyframe_annotations):
            motion_primitive_annotation = keyframe_annotations[action_index]
        else:
            motion_primitive_annotation = {}
            
            
        if concatenated_frames != None:
            start_frame=len(concatenated_frames)
        else:
            start_frame = 0
        
        
        if action_name in morphable_graph.subgraphs.keys() and mp_name in morphable_graph.subgraphs[action_name].nodes.keys():
            print "step",step_count,"convert ", action_name, mp_name
            try:
                parameters = get_optimal_parameters(morphable_graph,action_name,mp_name,constraints,\
                                                    options=options,
                                                    prev_action_name=prev_action_name, prev_mp_name=prev_mp_name, \
                                                    prev_frames=concatenated_frames, prev_parameters=prev_parameters,\
                                                    bvh_reader=bvh_reader, node_name_map=node_name_map,\
                                                    start_pose=start_pose,verbose=verbose) 
            except ConstraintError as e:
                print e.message,"for",action_name,mp_name,"at step",step_count    
                break
                
                
            prev_parameters = parameters
            prev_action_name = action_name
            prev_mp_name = mp_name
          
            #back project to euler frames#TODO implement FK based on quaternions 
            quaternion_frames = morphable_graph.subgraphs[action_name].nodes[mp_name].mp.back_project(parameters).get_motion_vector()
            tmp_euler_frames = convert_quaternion_to_euler(quaternion_frames.tolist())

            #concatenate with frames from previous steps
     
            if concatenated_frames != None:
                concatenated_frames = align_frames(bvh_reader,concatenated_frames,tmp_euler_frames, node_name_map,apply_smoothing) 
            else:
                if start_pose != None:
                    #rotate euler frames so the transformation can be found using alignment
                    #print "convert euler frames",start_pose
                    concatenated_frames = transform_euler_frames(tmp_euler_frames,start_pose["orientation"],start_pose["position"])  
                else:
                    concatenated_frames = tmp_euler_frames
                    
  
            #TODO add to action motion_primitive_annotation
            
            #update frame annotation
            action_frame_annotation = {}
            action_frame_annotation["startFrame"]=start_frame
            action_frame_annotation["elementaryAction"]=action_name
            action_frame_annotation["endFrame"]=len(concatenated_frames)-1
            frame_annotation['elementaryActionSequence'].append(action_frame_annotation)
            step_count += 1
        else:
            print "stopped the conversion at step",step_count," Did not find motion primitive in graph",action_name,mp_name
        
            break
    

    return concatenated_frames,frame_annotation,action_list


    
def run_pipeline(mg_input_filename,output_dir="output",max_step = -1,options = None,verbose = False):
    """Converts a file with a list of elementary actions to a bvh file
    """
    
    mm_directory = get_morphable_model_directory()
    transition_directory = get_transition_model_directory()
    if "use_transition_model" in options.keys():
        use_transition_model = options["use_transition_model"]
    else:
        use_transition_model = False 
    morphable_graph = MorphableGraph(mm_directory,transition_directory,use_transition_model)
    print_morphable_graph_structure(morphable_graph)
    
    graph_walk,start_pose, keyframe_annotations = elementary_action_breakdown(mg_input_filename,morphable_graph,verbose=verbose)
    tmp_file = "tmp.path"
    write_graph_walk_to_file(tmp_file,graph_walk,start_pose, keyframe_annotations)

    skeleton_path = "lib"+os.sep + "skeleton.bvh"
    bvh_reader = BVHReader(skeleton_path)    
    node_name_map = create_filtered_node_name_map(bvh_reader)
    euler_frames,frame_annotation, action_list = convert_graph_walk_to_motion(morphable_graph,\
                    graph_walk,options, bvh_reader,node_name_map,\
                    max_step= max_step,start_pose = start_pose,\
                    keyframe_annotations = keyframe_annotations,verbose = verbose) 


    mg_input = load_json_file(mg_input_filename)
    if "session" in mg_input.keys():    
        session = mg_input["session"]
    else:
        session = ""
        
    if euler_frames != None:
        export_euler_frames_to_bvh(output_dir,bvh_reader,euler_frames,prefix = session,start_pose= None)
    else:
        print "failed to generate motion data"
     

def main():
    

    input_file = "mg_input_test001.json"
    #input_file = "mg_input_test002.json"

    verbose=True
    options = generate_algorithm_settings(use_constraints = True,\
                            use_optimization = False,\
                            use_transition_model = False,\
                            use_constrained_gmm = True,\
                            activate_parameter_check = True,\
                            apply_smoothing = True,
                            sample_size = 100,\
                            constrained_gmm_pos_precision = 5,
                            constrained_gmm_rot_precision = 0.15,
                            optimization_method= "BFGS", \
                            max_optimization_iterations = 50,\
                            optimization_quality_scale_factor = 0.001,\
                            optimization_error_scale_factor = 0.01,\
                            optimization_tolerance = 0.05)
    run_pipeline(input_file,max_step= 80,options = options,verbose=verbose)

    return



if __name__ == "__main__":
    
    main()