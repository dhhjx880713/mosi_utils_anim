# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:38:45 2015

@author: hadu01
"""

import sys
from motion_dtw import MotionDynamicTimeWarping 



class Preprocessor(MotionDynamicTimeWarping):
    
    def __init__(self, params):
        super(Preprocessor, self).__init__(params)
        self.params = params
        
    def preprocess(self, elementary_action, primitive_type,
                   retarget_folder, annotation_file):
        self.segment_motions(elementary_action,
                             primitive_type,
                             retarget_folder,
                             annotation_file)
        self.normalize_root(self.params.ref_position,
                            self.params.touch_ground_joint)   
        self.align_motion(self.params.align_frame_idx, 
                          self.params.ref_orientation)         
        self.dtw()
    
    def save_result(self, save_path):
        self.save_warped_motion(save_path)
                           