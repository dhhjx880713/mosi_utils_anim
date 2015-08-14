__author__ = 'hadu01'
import os
import sys
ROOT_DIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
sys.path.append(ROOT_DIR)
TEST_LIB_PATH = ROOT_DIR + 'test'
TEST_DATA_PATH = ROOT_DIR + ".." + os.sep + "test_data" + os.sep + "animation_data" + os.sep
TEST_RESULT_PATH = ROOT_DIR + '../test_output/animation_data'
sys.path.append(TEST_LIB_PATH)
from morphablegraphs.animation_data.motion_editing import *
from morphablegraphs.animation_data.bvh import  BVHReader, BVHWriter
from libtest import params, pytest_generate_tests
import numpy as np


def test_extract_root_positions_from_frames():
    test_file = TEST_DATA_PATH + "walk_001_1_rightStance_86_128.bvh"
    print test_file
    bvhreader = BVHReader(test_file)
    assert True
    
if __name__ == "__main__":
    test_extract_root_positions_from_frames()