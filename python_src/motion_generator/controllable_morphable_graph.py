# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:30:45 2015

Motion Graphs interface for further integration


@author: erhe01
"""
import os
import time
from datetime import datetime
from lib.bvh2 import BVHReader, create_filtered_node_name_map
from lib.helper_functions import load_json_file, write_to_json_file,\
                                 get_morphable_model_directory,\
                                 get_transition_model_directory, \
                                 write_to_logfile, \
                                 export_quat_frames_to_bvh
from lib.graph_walk_extraction import elementary_action_breakdown,\
                                      write_graph_walk_to_file,\
                                      extract_keyframe_annotations                                    
from lib.morphable_graph import MorphableGraph
from constrain_motion import generate_algorithm_settings
from synthesize_motion import convert_elementary_action_list_to_motion
import numpy as np
from lib.constraint import global_counter_dict
LOG_FILE = "log.txt"
CONFIG_FILE = "config.json"



def export_synthesis_result(input_data, output_dir, output_filename, bvh_reader, quat_frames, frame_annotation, action_list, add_time_stamp=False):
      """ Saves the resulting animation frames, the annotation and actions to files. 
      Also exports the input file again to the output directory, where it is 
      used as input for the constraints visualization by the animation server.
      """
      write_to_json_file(output_dir + os.sep + output_filename + ".json", input_data) 
      write_to_json_file(output_dir + os.sep + output_filename + "_actions"+".json", action_list)
      
      frame_annotation["events"] = []
      for keyframe in action_list.keys():
        for event_desc in action_list[keyframe]:
            event = {}
            event["jointName"] = event_desc["parameters"]["joint"]
            event_type = event_desc["event"]
            target = event_desc["parameters"]["target"]
            event[event_type] = target
        
            event["frameNumber"] = int(keyframe)
            
            frame_annotation["events"].append(event)

      write_to_json_file(output_dir + os.sep + output_filename + "_annotations"+".json", frame_annotation)
      export_quat_frames_to_bvh(output_dir, bvh_reader, quat_frames, prefix=output_filename, start_pose=None, time_stamp=add_time_stamp)        

def print_runtime_statistics(seconds):
    minutes = int(seconds/60)
    seconds = seconds % 60
    total_time_string = "finished synthesis in "+ str(minutes) + " minutes "+ str(seconds)+ " seconds"
    evaluations_string = "total number of objective evaluations "+ str(global_counter_dict["evaluations"])
    error_string = "average error for "+ str(len(global_counter_dict["motionPrimitveErrors"])) +" motion primitives: " + str(np.average(global_counter_dict["motionPrimitveErrors"],axis=0))
    print total_time_string
    print evaluations_string
    print error_string        

class ControllableMorphableGraph(MorphableGraph):
    """
    Class that extends MorphableGraph with a method to synthesize a motion based on a json input file
    Parameters
    ----------
    * morphable_model_directory: string
    \tThe root directory of the morphable models of all elementary actions.
    
    * transition_model_directory: string
    \tThe directory of the morphable models of an elementary action.
    
    * transition_model_directory: string
    \tThe directory of the transition models.
    
    * skeleton_path : string
    \t Path to a bvh file that is used to extract joint hierarchy information.
    """
    def __init__(self, morphable_model_directory, transition_directory, skeleton_path, load_transtion_models=True):
        super(ControllableMorphableGraph,self).__init__(morphable_model_directory,
                                                        transition_directory,
                                                        load_transtion_models)

        self.bvh_reader = BVHReader(skeleton_path)
        self.node_name_map = create_filtered_node_name_map(self.bvh_reader)
        return
        
        
    def synthesize_motion(self, mg_input_filename, options=None, max_step=-1, verbose=False, output_dir="output", output_filename="", export=True):
        """
        Converts a json input file with a list of elementary actions and constraints into a BVH file.
        Calls either the function convert_elementary_action_list_to_motion or the function convert_graph_walk_to_motion 
        depending on the version parameter.
        
        Parameters
        ----------        
        * mg_input_filename : string
            Path to json file that contains a list of elementary actions with constraints.
        * options : dict
            Contains options for the algorithm.
            When set to None generate_algorithm_settings() is called with default settings
            use_constraints: Sets whether or not to use constraints 
            use_optimization : Sets whether to activate optimization or use only sampling
            use_constrained_gmm : Sets whether or not to constrain the GMM
            use_transition_model : Sets whether or not to predict parameters using the transition model
            apply_smoothing : Sets whether or not smoothing is applied on transitions
            optimization_settings : parameters for the optimization algorithm: 
                method, max_iterations,quality_scale_factor,error_scale_factor,
                optimization_tolerance
            constrained_gmm_settings : position and orientation precision + sample size                
            If set to None default settings are used.


        * max_step : integer
            Maximum number of motion primitives to be converted into a motion. If set to -1 this parameter is ignored
        * verbose : bool 
           Activates debug output to the console.
        * output_dir : string
            directory for the generated bvh file.
        * output_filename : string
           name of the file and its annotation
        * export : bool
            If set to True the generated motion is exported as BVH together 
            with a JSON-annotation file.
            
        Returns
        -------
        * concatenated_frames : np.ndarray
           A list of quaternion frames representing a motion.
         * frame_annotation : dict
           Associates the quaternion frames with the elementary actions
        * action_list : dict of dicts
           Contains actions/events for some frames based on the keyframe_annotations 
        """
        global_counter_dict["evaluations"] = 0
        mg_input = load_json_file(mg_input_filename)
        start = time.clock()
        if options is None:
            options = generate_algorithm_settings()

        ################################################################################
        # run the algorithm  
        # short description:
        # generate constraints based on the optimal parameters for the previous steps
        # and optimize for individual steps
        elementary_action_list = mg_input["elementaryActions"]
        start_pose = mg_input["startPose"]

        keyframe_annotations = extract_keyframe_annotations(elementary_action_list)

        quat_frames, frame_annotation, action_list = convert_elementary_action_list_to_motion(self,\
                                             elementary_action_list, options, self.bvh_reader, self.node_name_map,\
                                             max_step=max_step, start_pose=start_pose, keyframe_annotations=keyframe_annotations,\
                                             verbose=verbose)
                                             
        seconds = time.clock() - start
        print_runtime_statistics(seconds)
        
        ################################################################################
        # export the motion to a bvh file if export == True
        if export:
            if output_filename == "" and "session" in mg_input.keys():
                output_filename = mg_input["session"]
                frame_annotation["sessionID"] = mg_input["session"]

            if quat_frames is not None:
                time_stamp = unicode(datetime.now().strftime("%d%m%y_%H%M%S"))
                prefix = output_filename + "_" + time_stamp
                
                write_to_logfile(output_dir + os.sep + LOG_FILE, prefix, options)
                export_synthesis_result(mg_input, output_dir, output_filename, self.bvh_reader, quat_frames, frame_annotation, action_list, add_time_stamp=True)
            else:
                print "Error: failed to generate motion data"

        return quat_frames, frame_annotation, action_list                

    
def main():
    """ Demonstration of the ControllableMorphableGraph class"""
    mm_directory = get_morphable_model_directory()
    transition_directory = get_transition_model_directory()
    skeleton_path = "lib"+os.sep + "skeleton.bvh"
    use_transition_model = False
    start = time.clock()
    cmg = ControllableMorphableGraph(mm_directory,
                                     transition_directory,
                                     skeleton_path,
                                     use_transition_model)
    print "finished construction from file in",time.clock()-start,"seconds"
    testset_dir = "electrolux_test_set"

    input_file = testset_dir+os.sep+"right_pick_and_right_place_improved.path"
    #np.random.seed(2000)
    verbose = False
    export = True
    max_step = -1
    
    options = None
    if os.path.isfile(CONFIG_FILE):
         options = load_json_file(CONFIG_FILE)
    cmg.synthesize_motion(input_file, options=options, max_step=max_step,
                          verbose=verbose, output_dir=testset_dir, export=export)

if __name__ == "__main__":
    main()
