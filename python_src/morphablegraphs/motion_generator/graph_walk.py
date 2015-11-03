# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:39:41 2015

@author: erhe01
"""

import os
from datetime import datetime
from copy import copy
import json
import numpy as np
from ..utilities.io_helper_functions import write_to_json_file,\
                                          write_to_logfile
from ..animation_data.motion_vector import MotionVector
from constraints.spatial_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION
from ..animation_data.motion_editing import align_quaternion_frames

LOG_FILE = "log.txt"


class GraphWalkEntry(object):
    def __init__(self, motion_primitive_graph, node_key, parameters, arc_length, start_frame, end_frame, motion_primitive_constraints=None):
        self.node_key = node_key
        self.parameters = parameters
        self.arc_length = arc_length
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.motion_primitive_constraints = motion_primitive_constraints
        self.n_spatial_components = motion_primitive_graph.nodes[node_key].s_pca["n_components"]
        self.n_time_components = motion_primitive_graph.nodes[node_key].t_pca["n_components"]


class GraphWalk(object):
    """ Product of the MotionGenerate class. Contains the graph walk used to generate the frames,
        a mapping of frame segments
        to elementary actions and a list of events for certain frames.
    """
    def __init__(self, motion_primitive_graph, start_pose, algorithm_config):
        self.steps = []
        self.motion_primitive_graph = motion_primitive_graph
        self.skeleton = motion_primitive_graph.skeleton
        self.frame_annotation = dict()
        self.frame_annotation['elementaryActionSequence'] = []
        self.step_count = 0
        self.mg_input = None
        self._algorithm_config = algorithm_config
        self.motion_vector = MotionVector(algorithm_config)
        self.motion_vector.start_pose = start_pose
        self.keyframe_events_dict = dict()
        self.hand_pose_generator = None
        self.use_time_parameters = True

    def convert_to_motion(self, start_step=0, complete_motion_vector=True):
        self._convert_to_quaternion_frames(start_step, complete_motion_vector)
        self._create_event_dict()
        self._create_frame_annotation(start_step)
        self._add_event_list_to_frame_annotation()
        if self.hand_pose_generator is not None and complete_motion_vector:
            print "generate hand poses"
            self.hand_pose_generator.generate_hand_poses(self.motion_vector, self.keyframe_events_dict)

    def _convert_to_quaternion_frames(self, start_step=0, complete_motion_vector=True):
        """
        :param start_step:
        :return:
        """
        if start_step == 0:
            start_frame = 0
        else:
            start_frame = self.steps[start_step].start_frame
        self.motion_vector.clear(end_frame=start_frame)
        for step in self.steps[start_step:]:
            step.start_frame = start_frame
            quat_frames = self.motion_primitive_graph.nodes[step.node_key].back_project(step.parameters, use_time_parameters=(self.use_time_parameters and complete_motion_vector)).get_motion_vector()
            self.motion_vector.append_quat_frames(quat_frames)
            step.end_frame = self.get_num_of_frames()-1
            start_frame = step.end_frame + 1
        if complete_motion_vector:
            self.motion_vector.quat_frames = self.skeleton.complete_motion_vector_from_reference(self.motion_vector.quat_frames)
            #print "temp quat", temp_quat_frames[0]

    def _create_frame_annotation(self, start_step=0):
        self.frame_annotation['elementaryActionSequence'] = self.frame_annotation['elementaryActionSequence'][:start_step]
        if start_step == 0:
            start_frame = 0
        else:
            step = self.steps[start_step-1]
            if self.use_time_parameters:
                time_function = self.motion_primitive_graph.nodes[step.node_key]._inverse_temporal_pca(step.parameters[step.n_spatial_components:])
                start_frame = len(time_function)
            else:
                start_frame = step.end_frame
        end_frame = start_frame
        prev_step = None
        for step in self.steps[start_step:]:
            action_name = step.node_key[0]
            time_function = self.motion_primitive_graph.nodes[step.node_key]._inverse_temporal_pca(step.parameters[step.n_spatial_components:])
            if prev_step is not None and action_name != prev_step.node_key[0]:
                #add entry for previous elementary action
                print "add", prev_step.node_key[0]
                self.update_frame_annotation(prev_step.node_key[0], start_frame, end_frame-1)
                start_frame = end_frame
            if self.use_time_parameters:
                end_frame += len(time_function)
            else:
                end_frame += step.end_frame - step.start_frame
            prev_step = step
        print "add", prev_step.node_key[0]
        self.update_frame_annotation(prev_step.node_key[0], start_frame, end_frame-1)

    def _create_event_dict(self):
        """
        Traverse elementary actions and motion primitives
        :return:
        """
        self.keyframe_events_dict = dict()
        n_frames = 0
        for step in self.steps:
            if self.use_time_parameters:
                time_function = self.motion_primitive_graph.nodes[step.node_key]._inverse_temporal_pca(step.parameters[step.n_spatial_components:])
            for keyframe_event in step.motion_primitive_constraints.keyframe_event_list.values():
                # inverse lookup warped keyframe
                if self.use_time_parameters:
                    closest_keyframe = min(time_function, key=lambda x: abs(x-int(keyframe_event["canonical_keyframe"])))
                    warped_keyframe = np.where(time_function == closest_keyframe)[0][0]
                    warped_keyframe = n_frames+int(warped_keyframe)
                else:
                    warped_keyframe = n_frames+int(keyframe_event["canonical_keyframe"])

                print keyframe_event["event_list"]
                n_events = len(keyframe_event["event_list"])
                if n_events == 1:
                    events = keyframe_event["event_list"]
                else:
                    events = self._merge_multiple_keyframe_events(keyframe_event["event_list"], len(keyframe_event["event_list"]))
                if warped_keyframe in self.keyframe_events_dict:
                    events = events+self.keyframe_events_dict[warped_keyframe]
                self.keyframe_events_dict[warped_keyframe] = self._merge_multiple_keyframe_events(events, len(events))
            if self.use_time_parameters:
                n_frames += len(time_function)
            else:
                n_frames += step.end_frame - step.start_frame

    def _add_event_list_to_frame_annotation(self):
        """
        self.keyframe_events_dict[keyframe] m
        :return:
        """
        #print "keyframe event dict", self.keyframe_events_dict
        keyframe_event_list = []
        for keyframe in self.keyframe_events_dict.keys():
            #rint "keyframe event dict", self.keyframe_events_dict[keyframe]
            for event_desc in self.keyframe_events_dict[keyframe]:
                print "event description", event_desc
                event = dict()
                if self.mg_input is not None and self.mg_input.activate_joint_mapping:
                    event["jointName"] = self.mg_input.inverse_map_joint(event_desc["parameters"]["joint"])
                else:
                    event["jointName"] = event_desc["parameters"]["joint"]
                event_type = event_desc["event"]
                target = event_desc["parameters"]["target"]
                event[event_type] = target
                event["frameNumber"] = int(keyframe)
                keyframe_event_list.append(event)
        self.frame_annotation["events"] = keyframe_event_list

    def get_global_spatial_parameter_vector(self, start_step=0):
        initial_guess = []
        for step in self.steps[start_step:]:
            initial_guess += step.parameters[:step.n_spatial_components].tolist()
        return initial_guess

    def get_global_time_parameter_vector(self, start_step=0):
        initial_guess = []
        for step in self.steps[start_step:]:
            initial_guess += step.parameters[step.n_spatial_components:].tolist()
        return initial_guess

    def update_spatial_parameters(self, parameter_vector, start_step=0):
        print "update spatial parameters"
        offset = 0
        for step in self.steps[start_step:]:
            new_alpha = parameter_vector[offset:offset+step.n_spatial_components]
            step.parameters[:step.n_spatial_components] = new_alpha
            offset += step.n_spatial_components

    def update_time_parameters(self, parameter_vector, start_step=0):
        offset = 0
        for step in self.steps[start_step:]:
            new_gamma = parameter_vector[offset:offset+step.n_time_components]
            print new_gamma
            step.parameters[step.n_spatial_components:] = new_gamma
            offset += step.n_time_components

    def update_frame_annotation(self, action_name, start_frame, end_frame):
        """Addes a dictionary to self.frame_annotation marking start and end 
            frame of an action.
        """
        action_frame_annotation = dict()
        action_frame_annotation["startFrame"] = start_frame
        action_frame_annotation["elementaryAction"] = action_name
        action_frame_annotation["endFrame"] = end_frame
        self.frame_annotation['elementaryActionSequence'].append(action_frame_annotation)

    def append_quat_frames(self, new_frames):
        self.motion_vector.append_quat_frames(new_frames)

    def get_quat_frames(self):
        return self.motion_vector.quat_frames

    def get_num_of_frames(self):
        return self.motion_vector.n_frames

    def _export_event_dict(self, filename):
        #print "keyframe event dict", self.keyframe_events_dict, filename
        write_to_json_file(filename, self.keyframe_events_dict)

    def export_motion(self, output_dir, output_filename, add_time_stamp=False, export_details=False):
        """ Saves the resulting animation frames, the annotation and actions to files.
        Also exports the input file again to the output directory, where it is
        used as input for the constraints visualization by the animation server.
        """
        self.convert_to_motion()
        if self.motion_vector.has_frames():
            self.motion_vector.export(self.skeleton, output_dir, output_filename, add_time_stamp)
            if self.mg_input is not None:
                write_to_json_file(output_dir + os.sep + output_filename + ".json", self.mg_input.mg_input_file)
            self._export_event_dict(output_dir + os.sep + output_filename + "_actions"+".json")
            write_to_json_file(output_dir + os.sep + output_filename + "_annotations"+".json", self.frame_annotation)
            if export_details:
                time_stamp = unicode(datetime.now().strftime("%d%m%y_%H%M%S"))
                self.export_statistics(output_dir + os.sep + output_filename + "_statistics" + time_stamp + ".json")
                #write_to_logfile(output_dir + os.sep + LOG_FILE, output_filename + "_" + time_stamp, self._algorithm_config)
        else:
           print "Error: no motion data to export"

    def _merge_multiple_keyframe_events(self, annotations, num_events):
        """Merge events if there are more than one event defined for the same keyframe.
        """
        event_list = [(annotations[i]["event"], annotations[i]) for i in xrange(num_events)]
        temp_event_dict = dict()
        for name, event in event_list:
            if name not in temp_event_dict.keys():
               temp_event_dict[name] = event
            else:
                if "joint" in temp_event_dict[name]["parameters"].keys():
                    existing_entry = copy(temp_event_dict[name]["parameters"]["joint"])
                    if isinstance(existing_entry, basestring):
                        temp_event_dict[name]["parameters"]["joint"] = [existing_entry, event["parameters"]["joint"]]
                    else:
                        temp_event_dict[name]["parameters"]["joint"].append(event["parameters"]["joint"])
                    print "event dict merged", temp_event_dict[name]
                else:
                    print "event dict merge did not happen", temp_event_dict[name]
        return temp_event_dict.values()

    def print_statistics(self):
        statistics_string = self.get_statistics_string()
        print statistics_string

    def export_statistics(self, filename=None):
        time_stamp = unicode(datetime.now().strftime("%d%m%y_%H%M%S"))
        statistics_string = self.get_statistics_string()
        constraints = self.get_generated_constraints()
        constraints_string = json.dumps(constraints)
        if filename is None:
            filename = "graph_walk_statistics" + time_stamp + ".json"
        outfile = open(filename, "wb")
        outfile.write(statistics_string)
        outfile.write("\n"+constraints_string)
        outfile.close()

    def get_statistics_string(self):
        n_steps = len(self.steps)
        objective_evaluations = 0
        average_error = 0
        for step in self.steps:
            objective_evaluations += step.motion_primitive_constraints.evaluations
            average_error += step.motion_primitive_constraints.min_error
        average_error /= n_steps
        evaluations_string = "total number of objective evaluations " + str(objective_evaluations)
        error_string = "average error for " + str(n_steps) + \
                       " motion primitives: " + str(average_error)
        average_keyframe_error = self.get_average_keyframe_constraint_error()
        average_keyframe_error_string = "average keyframe constraint error " + str(average_keyframe_error)
        return average_keyframe_error_string + "\n" + evaluations_string + "\n" + error_string

    def get_average_keyframe_constraint_error(self):
        keyframe_constraint_errors = []
        step_index = 0
        prev_frames = None
        for step in self.steps:
            quat_frames = self.motion_primitive_graph.nodes[step.node_key].back_project(step.parameters, use_time_parameters=False).get_motion_vector()
            aligned_frames = align_quaternion_frames(quat_frames, prev_frames, self.motion_vector.start_pose)
            for constraint in step.motion_primitive_constraints.constraints:
                if (constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION or constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION) and\
                    not ("generated" in constraint.semantic_annotation.keys()):
                    #joint_position = self.skeleton.joint_map[constraint.joint_name].get_global_position(aligned_frames[constraint.canonical_keyframe])
                    #joint_position = constraint.skeleton.get_cartesian_coordinates_from_quaternion(constraint.joint_name, aligned_frames[constraint.canonical_keyframe])
                    #print "position constraint", joint_position, constraint.position
                    error = constraint.evaluate_motion_sample(aligned_frames)
                    print error
                    keyframe_constraint_errors.append(error)
            prev_frames = aligned_frames
            step_index += 1

        return np.average(keyframe_constraint_errors)

    def get_generated_constraints(self):
        step_count = 0
        generated_constraints = dict()
        for step in self.steps:
            key = str(step.node_key) + str(step_count)
            generated_constraints[key] = []
            for constraint in step.motion_primitive_constraints.constraints:

                if constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION and\
                    "generated" in constraint.semantic_annotation.keys():
                    generated_constraints[key].append(constraint.position)
            step_count += 1
        #generated_constraints = np.array(generated_constraints).flatten()
        return generated_constraints
