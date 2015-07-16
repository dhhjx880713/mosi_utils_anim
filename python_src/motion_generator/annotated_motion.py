# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:39:41 2015

@author: erhe01
"""
from copy import copy
from utilities.motion_editing import fast_quat_frames_alignment, transform_quaternion_frames
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


    def append_quat_frames(self, new_frames, start_pose, key_frame_annotations, apply_smoothing=True):
        """Align quaternion frames
           
        Parameters
        ----------
        * new_frames: list
        \tA list of quaternion frames
        * start_pose: dict
        \tA dictionary contains staring position and orientation
        
        Returns:
        --------
        * transformed_frames: np.ndarray
            Quaternion frames resulting from the back projection of s,
            transformed to fit to prev_frames.
            
        """
        if self.quat_frames is not None:
            self.quat_frames = fast_quat_frames_alignment(self.quat_frames, new_frames, apply_smoothing)                                              
        elif start_pose is not None:
            self.quat_frames = transform_quaternion_frames(new_frames,
                                                      start_pose["orientation"],
                                                      start_pose["position"])
        else:
            self.quat_frames = new_frames
                            
        self.n_frames = len(self.quat_frames)
        


        
    def update_frame_annotation(self, action_name, start_frame, end_frame):
        """Addes a dictionary to self.frame_annotation marking start and end 
            frame of an action.
        """
        action_frame_annotation = {}
        action_frame_annotation["startFrame"] =  start_frame
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
    
        action_list = {}
        key_frame_label_pairs = set()
        #extract the set of keyframes and their annotations referred to by the constraints
        for c in constraints:
            if "semanticAnnotation" in c.keys():
                for key_label in c["semanticAnnotation"]:  # can also contain lastFrame and firstFrame
                    if key_label in keyframe_annotations.keys() and key_label in time_information.keys():
                        if time_information[key_label] == "lastFrame":
                            key_frame = last_frame
                        elif time_information[key_label] == "firstFrame":
                            key_frame = start_frame
                            
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
                            existing_entry = copy(temp_event_dict[name]["parameters"]["joint"])
                            if isinstance(existing_entry, basestring):
                                temp_event_dict[name]["parameters"]["joint"] = [existing_entry,event["parameters"]["joint"]]
                            else:
                                temp_event_dict[name]["parameters"]["joint"].append(event["parameters"]["joint"])
                            print "event dict merged",temp_event_dict[name]
                        else:
                            print "event dict merge did not happen",temp_event_dict[name]   
                action_list[key_frame] = copy(temp_event_dict.values())
    
            else:
                action_list[key_frame] = annotations          
        return action_list
    