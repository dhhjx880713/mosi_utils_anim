# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:30:45 2015

Motion Graphs interface for further integration


@author: erhe01
"""
import sys
sys.path.append('..')
import os
import time
from datetime import datetime
import numpy as np
from utilities.io_helper_functions import load_json_file, write_to_json_file,\
                                 write_to_logfile, \
                                 export_quat_frames_to_bvh_file                     
from motion_model.elementary_action_graph import ElementaryActionGraph
from constraint.motion_constraints import MotionConstraints
from constraint.constraint_check import global_counter_dict
from synthesize_motion_primitive import generate_algorithm_settings
from synthesize_motion import generate_motion_from_constraints


LOG_FILE = "log.txt"
SKELETON_FILE = "skeleton.bvh" # TODO replace with standard skeleton in data directory


class MotionGenerator(object):
    """
    Creates a MorphableGraph instance and provides a method to synthesize a
    motion based on a json input file
    
    Parameters
    ----------
    * service_config: String
        Contains paths to the motion data.
    * use_transition_model : Booelan
        Activates the transition models.
    """
    def __init__(self,service_config, use_transition_model=False):
        
        morphable_model_directory = service_config["model_data"]
        transition_directory = service_config["transition_data"]

        self.morphable_graph = ElementaryActionGraph(SKELETON_FILE, morphable_model_directory,
                                                transition_directory,
                                                use_transition_model)

        return
        
        
    def synthesize_motion(self, mg_input, algorithm_config=None, max_step=-1, output_dir="output", output_filename="", export=True):
        """
        Converts a json input file with a list of elementary actions and constraints 
        into a motion saved to a BVH file.
        
        Parameters
        ----------        
        * mg_input_filename : string or dict
            Dict or Path to json file that contains a list of elementary actions with constraints.
        * algorithm_config : dict
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
            Debug parameter for the maximum number of motion primitives to be converted before stopping.
            If set to -1 this parameter is ignored
        * output_dir : string
            directory for the generated bvh file.
        * output_filename : string
           name of the file and its annotation
        * export : bool
            If set to True the generated motion is exported as BVH together 
            with a JSON-annotation file.
            
        Returns
        -------
        * motion : AnnotatedMotion
           Contains a list of quaternion frames and their annotation based on actions.
        """
        
        global_counter_dict["evaluations"] = 0
        if type(mg_input) != dict:
            mg_input = load_json_file(mg_input)
        start = time.clock()
        if algorithm_config is None:
            algorithm_config = generate_algorithm_settings()

        # run the algorithm
        motion_constrains = MotionConstraints(mg_input, self.morphable_graph)
        
        motion = generate_motion_from_constraints(motion_constrains, algorithm_config)
                                             
        seconds = time.clock() - start
        self.print_runtime_statistics(seconds)
        
        # export the motion to a bvh file if export == True
        if export:
            if output_filename == "" and "session" in mg_input.keys():
                output_filename = mg_input["session"]

                motion.frame_annotation["sessionID"] = mg_input["session"]

            if motion.quat_frames is not None:
                time_stamp = unicode(datetime.now().strftime("%d%m%y_%H%M%S"))
                prefix = output_filename + "_" + time_stamp
                
                write_to_logfile(output_dir + os.sep + LOG_FILE, prefix, algorithm_config)
                export_synthesis_result(mg_input, output_dir, output_filename, self.skeleton, motion.quat_frames, motion.frame_annotation, motion.action_list, add_time_stamp=True)
            else:
                print "Error: failed to generate motion data"

        return motion
    
    def print_runtime_statistics(self, seconds):
        minutes = int(seconds/60)
        seconds = seconds % 60
        total_time_string = "finished synthesis in "+ str(minutes) + " minutes "+ str(seconds)+ " seconds"
        evaluations_string = "total number of objective evaluations "+ str(global_counter_dict["evaluations"])
        error_string = "average error for "+ str(len(global_counter_dict["motionPrimitveErrors"])) +" motion primitives: " + str(np.average(global_counter_dict["motionPrimitveErrors"],axis=0))
        print total_time_string
        print evaluations_string
        print error_string
    

def export_synthesis_result(input_data, output_dir, output_filename, skeleton, quat_frames, frame_annotation, action_list, add_time_stamp=False):
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
      export_quat_frames_to_bvh_file(output_dir, skeleton, quat_frames, prefix=output_filename, start_pose=None, time_stamp=add_time_stamp)        


