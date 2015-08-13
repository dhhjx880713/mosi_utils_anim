__author__ = 'hadu01'
import os
import sys
ROOT_DIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
sys.path.append(ROOT_DIR)
TEST_LIB_PATH = ROOT_DIR + 'test'
TEST_DATA_PATH = ROOT_DIR +  '../test_data/animation_data'
TEST_RESULT_PATH = ROOT_DIR + '../test_output/animation_data'
sys.path.append(TEST_LIB_PATH)
from animation_data.motion_editing import *
from animation_data.bvh import  BVHReader, BVHWriter
from libtest import params, pytest_generate_tests
import numpy as np


def test_extract_root_positions_from_frames():
    test_file = r'C:\git-repo\ulm\morphablegraphs\test_data\animation_data\walk_001_1_rightStance_86_128.bvh'
    bvhreader = BVHReader(test_file)
    root_points = test_extract_root_positions_from_frames()