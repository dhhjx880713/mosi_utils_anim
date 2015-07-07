#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for preprocessing.py
===============================

by Martin Manns, Han
Daimler AG

"""

import os
import sys

from libtest import params, pytest_generate_tests

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(0, TESTPATH)
sys.path.insert(0, TESTPATH + (os.sep + os.pardir))

from preprocessing import *


class TestPreprocessing(object):
    """Unit test for preprocessing.py """
#    def setup_method(self, method):
#
#        testfile = TESTPATH + "/walk_001_1_rightStance_86_128.bvh"
#        self.bvh_reader = BVHReader(testfile)

    param_get_cartesian_frames = [
        {'bvhFile': TESTPATH + "/walk_001_1_rightStance_86_128.bvh",
         'res': 50}
    ]

    @params(param_get_cartesian_frames)
    def test_get_cartesian_frames(self, bvhFile, res):
        """Unit test for get_cartesian_frames """
        frames = get_cartesian_frames(bvhFile)
        assert len(frames) == res

    param_load_animations_as_cartesian_frames = [
        {'input_dir': os.sep.join([".."] * 6 +
                                  ["data", "1 - MoCap", "4 - Alignment",
                                   "elementary_action_walk", "leftStance"]),
         'res': 620}
    ]

    @params(param_load_animations_as_cartesian_frames)
    def test_load_animations_as_cartesian_frames(self, input_dir, res):
        """Unit test for load_animations_as_cartesian_frames"""
        os.chdir(TESTPATH + (os.sep + os.pardir))
        data = load_animations_as_cartesian_frames(input_dir)
        assert type(data) is dict and len(data) == res

    param_gen_file_paths = [
        {'input_dir': os.sep.join([".."] * 6 +
                                  ["data", "1 - MoCap", "4 - Alignment",
                                   "elementary_action_walk", "leftStance"]),
         'res': '.bvh'}
    ]

    @params(param_gen_file_paths)
    def test_get_file_paths(self, input_dir, res):
        """Unit test for test_get_file_paths """
        for item in gen_file_paths(input_dir):
            assert res in item

    param_get_input_data_folder = [
        {'elementary_action': "e1", 'motion_primitive': "m1",
         'res': os.sep.join([".."] * 6 +
                            ["data", "1 - MoCap", "4 - Alignment", "e1", "m1"]
                            )},
        {'elementary_action': "", 'motion_primitive': "_",
         'res': os.sep.join([".."] * 6 +
                            ["data", "1 - MoCap", "4 - Alignment", "", "_"])}
    ]

    @params(param_get_input_data_folder)
    def test_get_input_data_folder(self,
                                   elementary_action,
                                   motion_primitive,
                                   res):
        """Unit test for get_input_data_folder"""

        assert get_input_data_folder(elementary_action, motion_primitive) == \
            res

    params_clean_path = [{'path': '../../test', 'res': '\\\\?\\'}]

    @params(params_clean_path)
    def test_clean_path(self, path, res):
        """Unit test for clean_path """
        assert res in clean_path(path)

    params_get_output_folder = [
        {'res': os.sep.join([".."] * 6 +
                            ['data', '2 - PCA', 'spatial',
                            '1 - preprocessing', 'experiments',
                             '1 - FPCA with absolute joint positions in \
                             Cartesian space'])}
    ]

    @params(params_get_output_folder)
    def test_get_output_folder(self, res):
        """Unit test for get_output_folder"""
        assert get_output_folder() == res