# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:38:45 2015

@author: hadu01
"""

from morphablegraphs.construction.preprocessing.motion_dtw\
    import MotionDynamicTimeWarping


class Preprocessor(MotionDynamicTimeWarping):

    def __init__(self, params, verbose=False):
        super(Preprocessor, self).__init__(verbose)
        self.params = params

    def preprocess(self):
        # self.segment_motions(self.params.elementary_action,
        #                      self.params.motion_primitive,
        #                      self.params.retarget_folder,
        #                      self.params.annotation_file)
        self.normalize_root(self.params.ref_position)
        self.align_motion_by_vector(self.params.align_frame_idx,
                          self.params.ref_orientation)
        # self.correct_up_axis(self.params.align_frame_idx,
        #                      self.params.ref_up_vector)
        self.dtw()

    def save_result(self, save_path):
        self.save_warped_motion(save_path)
