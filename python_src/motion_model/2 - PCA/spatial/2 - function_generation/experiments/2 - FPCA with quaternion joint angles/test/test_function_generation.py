# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 17:30:54 2014
@brief: Unit test for function_generation.py
@author: hadu01
"""

import os
import sys

from libtest import params, pytest_generate_tests


TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(1, TESTPATH)
sys.path.insert(1, TESTPATH + (os.sep + os.pardir))

from function_generation import *


class TestFucntionGeneration(object):
    """unit test class for function-generation.py """
    params_get_input_data_folder = [{'res': os.sep.join([".."] * 6 + ['data',
                                     '2 - PCA', 'spatial',
                                     '1 - preprocessing', 'experiments',
                                     '1 - FPCA with absolute joint positions in Cartesian space'])}]

    @params(params_get_input_data_folder)
    def test_get_input_data_folder(self, res):
        """Unit test for get_input_data_folder """
        assert get_input_data_folder == res

    params_clean_path = [{'path': '../../test', 'res': '\\\\?\\'}]

    @params(params_clean_path)
    def test_clean_path(self, path, res):
        """Unit test for clean_path """
        assert res in clean_path(path)

    params_load_input_data = [{'elementary_action': 'walk',
                               'motion_primitive': 'leftStance',
                               'res': (47, 620, 57)}]

    @params(params_load_input_data)
    def test_load_input_data(self, elementary_action, motion_primitive, res):
        """Unit test for load_input_data"""
        os.chdir(TESTPATH + (os.sep + os.pardir))
        data = load_input_data(elementary_action, motion_primitive)
        assert data.shape == res

    params_function_generator = [{'elementary_action': 'walk',
                                  'motion_primitive': 'leftStance',
                                  'save_path': TESTPATH,
                                  'res': TESTPATH + os.sep +
                                  'walk_leftStance_functionalData.RData'}]

    @params(params_function_generator)
    def test_function_generator(self, elementary_action, motion_primitive,
                                save_path, res):
        """Unit test for function_generator"""
        os.chdir(TESTPATH + (os.sep + os.pardir))
        function_generator(elementary_action, motion_primitive, save_path)
        assert os.path.isfile(res) and os.stat(res).st_size != 0
        os.remove(res)

    params_get_output_folder = [{'res': os.sep.join([".."] * 6 + ['data',
                                 '2 - PCA', 'spatial',
                                 '2 - function_generation', 'experiments',
                                 '1 - FPCA with absolute joint positions in Cartesian space'])}]

    @params(params_get_output_folder)
    def test_get_output_folder(self, res):
        """Unit test for get_output_folder"""
        assert get_output_folder() == res