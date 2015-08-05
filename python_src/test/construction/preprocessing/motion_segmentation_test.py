# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:22:20 2015

@author: du
"""

import os
import sys
import numpy as np
ROOTDIR = os.sep.join(['..'] * 3)
import glob
sys.path.append(ROOTDIR + os.sep)
TESTPATH = ROOTDIR + os.sep + r'construction/preprocessing'
TESTLIBPATH = ROOTDIR + os.sep + 'test/'
sys.path.append(TESTPATH)
sys.path.append(TESTLIBPATH)
TESTDATAPATH = ROOTDIR + os.sep + \
    r'../test_data/constrction/preprocessing/motion_segmentation/Take_sidestep'
TESTRESULTPATH = ROOTDIR + os.sep + \
    r'../test_output/constrction/preprocessing/motion_segmentation/cutting_results'

from motion_segmentation import MotionSegmentation
from libtest import params, pytest_generate_tests


input_file_folder = TESTDATAPATH
input_annotation_file = TESTDATAPATH + os.sep + 'key_frame_annotation.txt'
output_file_folder = TESTRESULTPATH


class TestMotionSegmentation(object):

    "MotionSegmentation test class"

    def setup_method(self, method):
        self.motion_segmentor = MotionSegmentation()

    param_load_annotation = [{'annotation_file': input_annotation_file,
                              'res': {'sidestep_001_4.bvh': {'elementary_action': 'walk',
                                                             'motion_primitive': 'sidestepLeft',
                                                             'frames': [139, 263]}}}]

    @params(param_load_annotation)
    def test_load_annotation(self, annotation_file, res):
        self.motion_segmentor.load_annotation(annotation_file)
        filename = res.keys()[0]
        assert filename in self.motion_segmentor.annotation_label.keys()
        assert res[filename] in self.motion_segmentor.annotation_label[
            filename]

    param_convert_to_json = [{'annotation_file': input_annotation_file,
                              'res': {'sidestep_001_4.bvh': {'elementary_action': 'walk',
                                                             'motion_primitive': 'sidestepLeft',
                                                             'frames': [139, 263]}
                                      }}]

    @params(param_convert_to_json)
    def test_convert_to_json(self, annotation_file, res):
        annotated_data = self.motion_segmentor._convert_to_json(
            annotation_file)
        filename = res.keys()[0]
        assert filename in annotated_data.keys()
        assert res[filename] in annotated_data[filename]

    param_cut_files = [{'elementary_action': 'walk',
                        'primitive_type': 'sidestepLeft',
                        'data_path': input_file_folder,
                        'annotation_file': input_annotation_file,
                        'res': {'filename': 'walk_001_4_sidestepLeft_139_263.bvh',
                                'shape': (124, 156)}}]

    @params(param_cut_files)
    def test_cut_files(self, elementary_action, primitive_type, data_path,
                       annotation_file, res):
        self.motion_segmentor.load_annotation(annotation_file)
        self.motion_segmentor.cut_files(elementary_action,
                                        primitive_type,
                                        data_path)
        cutted_frames = self.motion_segmentor.cutted_motions[res['filename']]
        cutted_frames = np.asarray(cutted_frames)
        assert cutted_frames.shape == res['shape']

    param_save_segments = [{'elementary_action': 'walk',
                            'primitive_type': 'sidestepLeft',
                            'data_path': input_file_folder,
                            'annotation_file': input_annotation_file,
                            'save_path': output_file_folder,
                            'res': 20}]

    @params(param_save_segments)
    def test_save_segments(self, elementary_action, primitive_type,
                           data_path, annotation_file, save_path, res):
        self.motion_segmentor.segment_motions(elementary_action,
                                              primitive_type,
                                              data_path,
                                              annotation_file)
        self.motion_segmentor.save_segments(save_path)
        segmented_files = glob.glob(TESTRESULTPATH + os.sep + '*.bvh')
        assert len(segmented_files) == res
