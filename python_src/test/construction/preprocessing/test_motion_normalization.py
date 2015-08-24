# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 09:32:14 2015

@author: hadu01
"""

import os
import glob
from ....morphablegraphs.animation_data.motion_editing import pose_orientation_euler
from ....morphablegraphs.animation_data.bvh import BVHReader
from ....morphablegraphs.construction.preprocessing.motion_normalization import MotionNormalization
from ...libtest import params, pytest_generate_tests
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-4]) + os.sep
TESTDATAPATH = ROOTDIR + os.sep + r'../test_data/constrction/preprocessing/motion_normalization/cutting_results'
TESTRESULTPATH = ROOTDIR + os.sep + r'../test_output/constrction/preprocessing/motion_normalization'


class TestMotionNormalization(object):

    def setup_method(self, method):
        self.motion_normalizer = MotionNormalization()

    param_load_data_for_normalization = [
        {"data_folder": TESTDATAPATH}]

    @params(param_load_data_for_normalization)
    def test_load_data_for_normalization(self, data_folder):

        self.motion_normalizer.load_data_for_normalization(data_folder)
        files = glob.glob(data_folder + os.sep + '*.bvh')
        assert len(self.motion_normalizer.cutted_motions) == len(files)

    param_normalize_root = [{'origin_point': {'x': 0, 'y': 0, 'z': 0},
                             'touch_ground_joint': 'Bip01_R_Toe0',
                             'data_folder': TESTDATAPATH}]

    @params(param_normalize_root)
    def test_normalize_root(self, origin_point, touch_ground_joint, data_folder):
        self.motion_normalizer.load_data_for_normalization(data_folder)
        self.motion_normalizer.normalize_root(origin_point, touch_ground_joint)
        assert self.motion_normalizer.ref_bvhreader.node_names[
            'Hips']['offset'] == [0, 0, 0]
        for filename, frames in self.motion_normalizer.translated_motions.iteritems():
            assert frames[0][0] == 0
            assert frames[0][2] == 0

    param_align_motion = [{'aligned_frame_idx': 0,
                           'ref_orientation': {'x': 0, 'y': 0, 'z': -1},
                           'data_folder': TESTDATAPATH,
                           'res': [0, -1]}]

    @params(param_align_motion)
    def test_align_motion(self, aligned_frame_idx,
                          ref_orientation, data_folder, res):
        self.motion_normalizer.load_data_for_normalization(data_folder)
        self.motion_normalizer.translated_motions = self.motion_normalizer.cutted_motions
        self.motion_normalizer.align_motion(aligned_frame_idx, ref_orientation)
        for filename, frames in self.motion_normalizer.aligned_motions.iteritems():
            orientation = pose_orientation_euler(frames[0])
            for i in xrange(len(orientation)):
                assert round(orientation[i], 3) == res[i]

    param_save_motion = [{'data_folder': TESTDATAPATH,
                          'save_path': TESTRESULTPATH}]

    @params(param_save_motion)
    def test_save_motion(self, data_folder, save_path):
        self.motion_normalizer.load_data_for_normalization(data_folder)
        self.motion_normalizer.ref_bvhreader = BVHReader(
            self.motion_normalizer.ref_bvh)
        self.motion_normalizer.aligned_motions = self.motion_normalizer.cutted_motions
        self.motion_normalizer.save_motion(save_path)
        files = glob.glob(save_path + os.sep + '*bvh')
        assert len(files) == len(self.motion_normalizer.aligned_motions)
