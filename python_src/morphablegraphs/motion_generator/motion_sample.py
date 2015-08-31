# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:39:41 2015

@author: erhe01
"""

import os
from datetime import datetime
from copy import copy
from ..animation_data.motion_editing import DEFAULT_SMOOTHING_WINDOW_SIZE,\
                                          fast_quat_frames_alignment,\
                                          transform_quaternion_frames
from constraints.spatial_constraints.keyframe_constraints.keyframe_constraint_base import KeyframeConstraintBase
from ..utilities.io_helper_functions import write_to_json_file,\
                                          write_to_logfile, \
                                          export_quat_frames_to_bvh_file

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


class MotionSample(object):
    """ Product of the MotionGenerate class. Contains quaternion frames,
        the graph walk used to generate the frames, a mapping of frame segments 
        to elementary actions and a list of events for certain frames.
    """
    def __init__(self, skeleton, start_pose, algorithm_config):
        self.skeleton = skeleton
        self.start_pose = start_pose
        self.apply_smoothing = algorithm_config["apply_smoothing"]
        self.smoothing_window = algorithm_config["smoothing_window"]
        self.action_list = {}
        self.frame_annotation = dict()
        self.frame_annotation['elementaryActionSequence'] = []
        self.graph_walk = []
        self.quat_frames = None
        self.step_count = 0
        self.n_frames = 0
        self.mg_input = {}

    def append_quat_frames(self, new_frames):
        """Align quaternion frames to previous frames
           
        Parameters
        ----------
        * new_frames: list
            A list of quaternion frames
        """
        if self.quat_frames is not None:
            self.quat_frames = fast_quat_frames_alignment(self.quat_frames,
                                                          new_frames,
                                                          self.apply_smoothing,
                                                          self.smoothing_window)                                              
        elif self.start_pose is not None:
            self.quat_frames = transform_quaternion_frames(new_frames,
                                                      self.start_pose["orientation"],
                                                      self.start_pose["position"])
        else:
            self.quat_frames = new_frames
                            
        self.n_frames = len(self.quat_frames)

    def update_frame_annotation(self, action_name, start_frame, end_frame):
        """Addes a dictionary to self.frame_annotation marking start and end 
            frame of an action.
        """
        action_frame_annotation = {}
        action_frame_annotation["startFrame"] = start_frame
        action_frame_annotation["elementaryAction"] = action_name
        action_frame_annotation["endFrame"] = end_frame
        self.frame_annotation['elementaryActionSequence'].append(action_frame_annotation)  

    def update_action_list(self, constraints, keyframe_annotations, canonical_key_frame_annotation, start_frame, last_frame):
        """  merge the new actions list with the existing list.
        """
        new_action_list = self._associate_actions_to_frames(self.quat_frames, canonical_key_frame_annotation, constraints, keyframe_annotations, start_frame, last_frame)
        self.action_list.update(new_action_list)

    def _associate_actions_to_frames(self, quat_frames, time_information, constraints, keyframe_annotations, start_frame, last_frame):
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
        key_frame_label_pairs = set()
        #extract the set of keyframes and their annotations referred to by the constraints
        for constraint in constraints:
            if isinstance(constraint, KeyframeConstraintBase):
                for key_label in constraint.semantic_annotation.keys():  # can also contain lastFrame and firstFrame
                    if key_label in keyframe_annotations.keys() and key_label in time_information.keys():
                        if time_information[key_label] == "lastFrame":
                            key_frame = last_frame
                        elif time_information[key_label] == "firstFrame":
                            key_frame = start_frame
                        if "annotations" in keyframe_annotations[key_label].keys():
                            key_frame_label_pairs.add((key_frame, key_label))
        return self._extract_actions_from_keyframe_annotations(key_frame_label_pairs, keyframe_annotations)

    def _extract_actions_from_keyframe_annotations(self, key_frame_label_pairs, keyframe_annotations):
        """extract the annotations for the referred keyframes
        """
        action_list = {}
        for key_frame, key_label in key_frame_label_pairs:
            annotations = keyframe_annotations[key_label]["annotations"]
            num_events = len(annotations)
            if num_events > 1:
                temp_event_dict = self._merge_multiple_keyframe_events(annotations, num_events)
                action_list[key_frame] = copy(temp_event_dict.values())
            else:
                action_list[key_frame] = annotations

        return action_list

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
        return temp_event_dict

    def export(self, output_dir, output_filename, add_time_stamp=False, write_log=False):
          """ Saves the resulting animation frames, the annotation and actions to files. 
          Also exports the input file again to the output directory, where it is 
          used as input for the constraints visualization by the animation server.
          """
            
          if self.quat_frames is not None:
    
              time_stamp = unicode(datetime.now().strftime("%d%m%y_%H%M%S"))
           
              write_to_json_file(output_dir + os.sep + output_filename + ".json", self.mg_input)
              write_to_json_file(output_dir + os.sep + output_filename + "_actions"+".json", self.action_list)
              
              reordered_frame_annotation = self._add_events_to_frame_annotation(self.frame_annotation)
              if write_log:
                  write_to_logfile(output_dir + os.sep + LOG_FILE, output_filename + "_" + time_stamp, self._algorithm_config)
              write_to_json_file(output_dir + os.sep + output_filename + "_annotations"+".json", reordered_frame_annotation)
              export_quat_frames_to_bvh_file(output_dir, self.skeleton, self.quat_frames, prefix=output_filename, start_pose=None, time_stamp=add_time_stamp)        
          else:
             print "Error: no motion data to export"

    def _add_events_to_frame_annotation(self, frame_annotation):
        reordered_frame_annotation = copy(frame_annotation)
        reordered_frame_annotation["events"] = []
        for keyframe in self.action_list.keys():
            for event_desc in self.action_list[keyframe]:
                event = {}
                event["jointName"] = event_desc["parameters"]["joint"]
                event_type = event_desc["event"]
                target = event_desc["parameters"]["target"]
                event[event_type] = target
                event["frameNumber"] = int(keyframe)
                reordered_frame_annotation["events"].append(event)
        return reordered_frame_annotation