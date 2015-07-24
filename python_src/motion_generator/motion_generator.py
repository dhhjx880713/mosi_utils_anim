# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:30:45 2015

Motion Graphs interface for further integration
Runs the complete Morphable Graphs Pipeline to generate a motion based on an
json input file. Runs the optimization sequentially and creates constraints
based on previous steps.

@author: Erik Herrmann, Han Du, Fabian Rupp, Markus Mauer
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
from utilities.exceptions import SynthesisError, PathSearchError
from motion_model import NODE_TYPE_START, NODE_TYPE_STANDARD, NODE_TYPE_END
from synthesize_motion_primitive import MotionPrimitiveGenerator
from algorithm_configuration import AlgorithmConfigurationBuilder
from constraint.motion_primitive_constraints import MotionPrimitiveConstraints
from annotated_motion import AnnotatedMotion, GraphWalkEntry

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
    def __init__(self,service_config, algorithm_config):
        self._service_config = service_config        
        self._algorithm_config = algorithm_config
        morphable_model_directory = self._service_config["model_data"]
        transition_directory = self._service_config["transition_data"]

        self.morphable_graph = ElementaryActionGraph(SKELETON_FILE, morphable_model_directory,
                                                transition_directory,
                                                self._algorithm_config["use_transition_model"])

        return

    def set_algorithm_config(self, algorithm_config):
        """
        Parameters
        ----------
        * algorithm_config : dict
            Contains options for the algorithm.
            When set to None AlgorithmSettingsBuilder() is used to generate default settings
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
        """
        if algorithm_config is None:
            algorithm_config_builder = AlgorithmConfigurationBuilder()
            self._algorithm_config = algorithm_config_builder.get_configuration()
        else:
            self._algorithm_config = algorithm_config
        
    def generate_motion(self, mg_input, output_dir="output", output_filename="", export=True):
        """
        Converts a json input file with a list of elementary actions and constraints 
        into a motion saved to a BVH file.
        
        Parameters
        ----------        
        * mg_input_filename : string or dict
            Dict or Path to json file that contains a list of elementary actions with constraints.
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
        motion_constrains = MotionConstraints(mg_input, self.morphable_graph)
        
        motion = self._generate_motion_from_constraints(motion_constrains)
                                             
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
                
                write_to_logfile(output_dir + os.sep + LOG_FILE, prefix, self._algorithm_config)
                self.export_synthesis_result(mg_input, output_dir, output_filename, motion, add_time_stamp=True)
            else:
                print "Error: failed to generate motion data"

        return motion
        
        
    
    def _generate_motion_from_constraints(self, motion_constraints):
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
        if self._algorithm_config["verbose"]:
            for key in self._algorithm_config.keys():
                print key,self._algorithm_config[key]
    
        motion = AnnotatedMotion()
        action_constraints = motion_constraints.get_next_elementary_action_constraints()
        while action_constraints is not None:
       
            if self._algorithm_config["debug_max_step"] > -1 and motion.step_count > self._algorithm_config["debug_max_step"]:
                print "reached max step"
                break
              
            if self._algorithm_config["verbose"]:
               print "convert",action_constraints.action_name,"to graph walk"
    
            try:
                motion = self._append_elementary_action_to_motion(action_constraints, motion)
                
            except SynthesisError as e:
                print "Arborting conversion",e.message
                return motion
            action_constraints = motion_constraints.get_next_elementary_action_constraints() 
        return motion
    
    
    
    def _append_elementary_action_to_motion(self, action_constraints, motion):
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
        * start_pose : dict
         Contains orientation and position as lists with three elements
    
        * keyframe_annotations : dict of dicts
          Contains a dict of events/actions associated with certain keyframes
    
        Returns
        -------
        * motion: AnnotatedMotion
        """
        
        if motion.step_count >0:
             prev_action_name = motion.graph_walk[-1]
             prev_mp_name = motion.graph_walk[-1]
        else:
             prev_action_name = None
             prev_mp_name = None
            
        motion_primitive_generator = MotionPrimitiveGenerator(action_constraints, self._algorithm_config, prev_action_name)
        start_frame = motion.n_frames    
        #skeleton = action_constraints.get_skeleton()
        morphable_subgraph = action_constraints.get_subgraph()
        
        trajectory_following_settings = self._algorithm_config["trajectory_following_settings"]#  TODO move trajectory_following_settings to different key of the algorithm_config
    
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
    
            if self._algorithm_config["debug_max_step"]  > -1 and motion.step_count + temp_step > self._algorithm_config["debug_max_step"]:
                print "reached max step"
                break
            #######################################################################
            # Get motion primitive = extract from graph based on previous last step + heuristic
            if temp_step == 0:  
                 current_motion_primitive = action_constraints.parent_constraint.morphable_graph.get_random_action_transition(motion, action_constraints.action_name)
                 current_motion_primitive_type = NODE_TYPE_START
                 if current_motion_primitive is None:

                     print "Error: Could not find a transition of type action_transition from ",prev_action_name,prev_mp_name ," to state",current_motion_primitive
                     break
            elif len(morphable_subgraph.nodes[current_motion_primitive].outgoing_edges) > 0:
                prev_motion_primitive = current_motion_primitive
                current_motion_primitive, current_motion_primitive_type = morphable_subgraph.get_random_transition(motion, action_constraints, travelled_arc_length, arc_length_of_end)
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
    
            except PathSearchError as e:
                    print "moved beyond end point using parameters",
                    str(e.search_parameters)
                    break
                
            # get optimal parameters, Back-project to frames in joint angle space,
            # Concatenate frames to motion and apply smoothing
            tmp_quat_frames, parameters = motion_primitive_generator.generate_motion_primitive_from_constraints(motion_primitive_constraints, motion)                                            
            
            #update annotated motion
            canonical_keyframe_labels = morphable_subgraph.get_canonical_keyframe_labels(current_motion_primitive)
            start_frame = motion.n_frames
            motion.append_quat_frames(tmp_quat_frames, action_constraints.start_pose, self._algorithm_config["apply_smoothing"])
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
    
    


    
    
    def print_runtime_statistics(self, seconds):
        minutes = int(seconds/60)
        seconds = seconds % 60
        total_time_string = "finished synthesis in "+ str(minutes) + " minutes "+ str(seconds)+ " seconds"
        evaluations_string = "total number of objective evaluations "+ str(global_counter_dict["evaluations"])
        error_string = "average error for "+ str(len(global_counter_dict["motionPrimitveErrors"])) +" motion primitives: " + str(np.average(global_counter_dict["motionPrimitveErrors"],axis=0))
        print total_time_string
        print evaluations_string
        print error_string
    

    def export_synthesis_result(self, input_data, output_dir, output_filename, motion, add_time_stamp=False):
          """ Saves the resulting animation frames, the annotation and actions to files. 
          Also exports the input file again to the output directory, where it is 
          used as input for the constraints visualization by the animation server.
          """
          write_to_json_file(output_dir + os.sep + output_filename + ".json", input_data) 
          write_to_json_file(output_dir + os.sep + output_filename + "_actions"+".json", motion.action_list)
          
          motion.frame_annotation["events"] = []
          reordered_frame_annotation = motion.frame_annotation
          for keyframe in motion.action_list.keys():
            for event_desc in motion.action_list[keyframe]:
                event = {}
                event["jointName"] = event_desc["parameters"]["joint"]
                event_type = event_desc["event"]
                target = event_desc["parameters"]["target"]
                event[event_type] = target
                event["frameNumber"] = int(keyframe)
                reordered_frame_annotation["events"].append(event)
    
          write_to_json_file(output_dir + os.sep + output_filename + "_annotations"+".json", reordered_frame_annotation)
          export_quat_frames_to_bvh_file(output_dir, self.morphable_graph.skeleton, motion.quat_frames, prefix=output_filename, start_pose=None, time_stamp=add_time_stamp)        
    
    
    
