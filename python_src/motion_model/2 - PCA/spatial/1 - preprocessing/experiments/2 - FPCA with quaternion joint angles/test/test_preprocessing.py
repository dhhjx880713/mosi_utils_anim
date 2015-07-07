#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for preprocessing.py
===============================

by Martin Manns
Daimler AG

"""

import os
import sys

from libtest import params, pytest_generate_tests

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(0, TESTPATH)
sys.path.insert(0, TESTPATH + (os.sep + os.pardir))

from preprocessing import gen_file_paths, get_input_data_folder


def test_gen_file_paths():
    pass

param_get_input_data_folder = [
    {'elementary_action': "e1", 'motion_primitive': "m1",
     'res': os.sep.join([".."] * 6 +
                        ["data", "1 - MoCap", "4 - Alignment", "e1", "m1"])},
    {'elementary_action': "", 'motion_primitive': "_",
     'res': os.sep.join([".."] * 6 +
                        ["data", "1 - MoCap", "4 - Alignment", "", "_"])},
]


@params(param_get_input_data_folder)
def test_get_input_data_folder(elementary_action, motion_primitive, res):
    """Unit test for get_input_data_folder"""

    assert get_input_data_folder(elementary_action, motion_primitive) == res
