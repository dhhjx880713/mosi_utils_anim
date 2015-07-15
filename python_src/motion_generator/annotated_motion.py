# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:39:41 2015

@author: erhe01
"""
import numpy as np
from utilities.motion_editing import fast_quat_frames_alignment, transform_quaternion_frames, smoothly_concatenate_quaternion_frames, SMOOTHING_WINDOW_SIZE
from constraint.constraint_extraction import associate_actions_to_frames

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
        


        
    def update_frame_annotation(self,action_name, start_frame, end_frame):
            #update frame annotation
        action_frame_annotation = {}
        action_frame_annotation["startFrame"] =  start_frame
        action_frame_annotation["elementaryAction"] = action_name
        action_frame_annotation["endFrame"] = end_frame
        self.frame_annotation['elementaryActionSequence'].append(action_frame_annotation)  


    def update_action_list(self, constraints, keyframe_annotations, canonical_key_frame_annotation, start_frame, last_frame):
        """  merge the new actions list with the existing list.
        """
        new_action_list = associate_actions_to_frames(self.quat_frames, canonical_key_frame_annotation, constraints, keyframe_annotations, start_frame, last_frame)
        self.action_list.update(new_action_list)
  