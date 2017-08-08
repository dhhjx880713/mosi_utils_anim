"""
test_bvh_reader.py
Unit test for bvh.py
======

Author: Han

"""

import os
import sys
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
sys.path.append(ROOTDIR)
sys.path.append(os.path.join(ROOTDIR, 'test'))
from morphablegraphs.animation_data.bvh import BVHReader, BVHWriter
from morphablegraphs.animation_data.motion_editing import get_cartesian_coordinates_from_euler_full_skeleton
from morphablegraphs.animation_data.skeleton import Skeleton
from libtest import params, pytest_generate_tests
ROOT_DIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep
TEST_DATA_PATH = ROOT_DIR + '../test_data/animation_data'
TEST_RESULT_PATH = ROOT_DIR + '../test_output/animation_data'


class TestBVHReader(object):
    """BVHReader test class"""

    def setup_class(self):
        testfile = TEST_DATA_PATH + os.sep + "walk_001_1_rightStance_86_128.bvh"
        self.bvh_reader = BVHReader(testfile)

    def test_init(self):
        assert self.bvh_reader.filename == "walk_001_1_rightStance_86_128.bvh"

    param_read_frametime = [{'res': 0.013889}] 
    
    @params(param_read_frametime)
    def test_read_frametime(self, res):
        assert self.bvh_reader.frame_time == res
    
    param_read_skeleton = [{'res': ['Hips', 'Spine', 'Spine_1', 'Neck', 'Head', 
                                     'Head_EndSite', 'LeftShoulder', 'LeftArm', 
                                     'LeftForeArm', 'LeftHand', 'Bip01_L_Finger0', 
                                     'Bip01_L_Finger01', 'Bip01_L_Finger02', 
                                     'Bip01_L_Finger02_EndSite', 'Bip01_L_Finger1', 
                                     'Bip01_L_Finger11', 'Bip01_L_Finger12', 
                                     'Bip01_L_Finger12_EndSite', 'Bip01_L_Finger2', 
                                     'Bip01_L_Finger21', 'Bip01_L_Finger22', 
                                     'Bip01_L_Finger22_EndSite', 'Bip01_L_Finger3', 
                                     'Bip01_L_Finger31', 'Bip01_L_Finger32', 
                                     'Bip01_L_Finger32_EndSite', 'Bip01_L_Finger4', 
                                     'Bip01_L_Finger41', 'Bip01_L_Finger42', 
                                     'Bip01_L_Finger42_EndSite', 'RightShoulder', 
                                     'RightArm', 'RightForeArm', 'RightHand', 
                                     'Bip01_R_Finger0', 'Bip01_R_Finger01', 
                                     'Bip01_R_Finger02', 'Bip01_R_Finger02_EndSite', 
                                     'Bip01_R_Finger1', 'Bip01_R_Finger11', 
                                     'Bip01_R_Finger12', 'Bip01_R_Finger12_EndSite', 
                                     'Bip01_R_Finger2', 'Bip01_R_Finger21', 
                                     'Bip01_R_Finger22', 'Bip01_R_Finger22_EndSite', 
                                     'Bip01_R_Finger3', 'Bip01_R_Finger31', 
                                     'Bip01_R_Finger32', 'Bip01_R_Finger32_EndSite', 
                                     'Bip01_R_Finger4', 'Bip01_R_Finger41', 
                                     'Bip01_R_Finger42', 'Bip01_R_Finger42_EndSite', 
                                     'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'Bip01_L_Toe0', 
                                     'Bip01_L_Toe0_EndSite', 'RightUpLeg', 'RightLeg', 
                                     'RightFoot', 'Bip01_R_Toe0', 'Bip01_R_Toe0_EndSite']}] 
                                      
    @params(param_read_skeleton)                                  
    def test_read_skeleton(self, res):
        assert list(self.bvh_reader.node_names.keys()) == res
    
    param_read_frames = [{  'frame_index': 0,
                            'res': [-1.15780000e+00,   9.00000000e+01,   4.68537285e+00,  -1.79901167e+02,
                                    2.79680763e-01,  -8.96900865e+01,   1.94603800e+00,   2.17988400e+00,
                                    5.59918000e-01,  -0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                    -1.69722400e+00,  -2.26801020e+01,  -2.22689700e+00,  -2.93576300e+00,
                                    1.38533000e+01,   1.75337400e+00,   1.73755896e+02,  -1.25893040e+01,    
                                    1.08381949e+02,   2.89928280e+01,   3.03452120e+01,   4.18017790e+01,
                                    -3.00000000e-06,   4.33379430e+01,   9.00000000e-06,  -9.75027060e+01,
                                    -4.04822000e-01,  -8.44253000e+00,  -1.00000000e-06,  -1.00000000e-06,
                                    1.00000000e-06,  -1.00000000e-06,   0.00000000e+00,   1.00000000e-06,
                                    -0.00000000e+00,   0.00000000e+00,  -1.00000000e-06,  -1.00000000e-06,
                                    -1.00000000e-06,   1.00000000e-06,  -1.00000000e-06,   0.00000000e+00,
                                    1.00000000e-06,   2.00000000e-06, -1.00000000e-06,  -1.00000000e-06,
                                    -1.00000000e-06,  -1.00000000e-06,  -5.64980006e-30,   6.00000000e-06,
                                    0.00000000e+00,   1.00000000e-06,   7.00000000e-06,   0.00000000e+00,
                                    1.00000000e-06,  -1.00000000e-06,  -1.00000000e-06,  -5.64980006e-30,
                                    -2.00000000e-06,  -1.00000000e-06,  -1.00000000e-06,   2.00000000e-06,
                                    0.00000000e+00,  -1.00000000e-06,  -1.00000000e-06,  -1.00000000e-06,
                                    1.00000000e-06,   6.00000000e-06,   4.51984005e-29,   2.00000000e-06,
                                    7.00000000e-06,   1.00000000e-06,   1.00000000e-06,  -1.66268059e+02,
                                    -1.17269520e+01,  -1.07405649e+02,  -4.62179910e+01,   2.89285590e+01,
                                    -2.73243120e+01,   1.00000000e-06,   5.36433220e+01,   1.40000000e-05,
                                    1.18473429e+02,   1.74674810e+01,  -8.16834900e+00,   2.00000000e-06,
                                    2.00000000e-06,  -2.00000000e-06,   2.00000000e-06,   0.00000000e+00,
                                    5.00000000e-06,  -0.00000000e+00,   2.00000000e-06,  -1.00000000e-06,
                                    1.00000000e-06,   1.00000000e-06,  -1.00000000e-06,   2.00000000e-06,
                                    -2.25992002e-29,   4.00000000e-06,   2.00000000e-06,  -1.12996001e-29,
                                    2.00000000e-06,   2.00000000e-06,   2.00000000e-06,  -1.00000000e-06,
                                    3.00000000e-06,  -1.00000000e-06,   4.00000000e-06,  -0.00000000e+00,
                                    1.00000000e-06,   0.00000000e+00,  -0.00000000e+00,   1.00000000e-06,
                                    0.00000000e+00,   2.00000000e-06,   3.00000000e-06,   1.00000000e-06,
                                    2.00000000e-06,   0.00000000e+00,   0.00000000e+00,  -0.00000000e+00,
                                    1.00000000e-06,  -2.00000000e-06,   3.00000000e-06,   1.00000000e-06,
                                    4.00000000e-06,  -1.00000000e-06,   0.00000000e+00,  -1.00000000e-06,
                                    2.00732790e+01,   2.66166500e+01,   1.73715634e+02,  -2.00000000e-06,
                                    1.70752270e+01,   4.00000000e-06,   3.84696400e+00,  -8.23398300e+00,
                                    -1.95822700e+00,   9.00000000e+01,   3.99999999e-06,   7.50000000e+01,
                                    -1.49263250e+01,  -9.97119900e+00,   1.74878073e+02,  -2.00000000e-06,
                                    2.22030360e+01,   4.00000000e-06,  -9.85101500e+00,  -1.84251180e+01,
                                    -8.31699000e-01,   9.00000000e+01,  -1.00000001e-06,   7.50000000e+01]}]
                                    
    @params(param_read_frames)
    def test_read_frames(self, frame_index, res):
        for i in range(len(self.bvh_reader.frames[frame_index])):
            assert round(self.bvh_reader.frames[frame_index][i], 5) == round(res[i], 5)                               

    param_get_angles = [{'node_channel': ('RightShoulder', 'Xrotation'),
                         'res': -166.268},
                        {'node_channel': ('RightShoulder', 'Yrotation'),
                         'res': -11.727},
                        {'node_channel': ('RightShoulder', 'Zrotation'),
                         'res': -107.406}]

    @params(param_get_angles)
    def test_get_angles(self, node_channel, res):
        """Unit test for get_angles"""
        angle = self.bvh_reader.get_angles(node_channel)[0]
        assert round(angle, 3) == round(res, 3)
                                                                