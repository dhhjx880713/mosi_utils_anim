# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 15:18:54 2015

@author: hadu01
"""

import os
import numpy as np
import glob
import json
from ....morphablegraphs.animation_data.bvh import BVHReader
from ....morphablegraphs.construction.construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from ....morphablegraphs.construction.fpca.motion_dimension_reduction import MotionDimensionReduction
from ...libtest import params, pytest_generate_tests
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-4]) + os.sep
TEST_DATA_PATH = ROOTDIR + os.sep + r'../test_data/constrction/fpca'
MOTION_DATA_FILE = TEST_DATA_PATH + os.sep + 'motion_data.json'


def gen_test_data():
    data_folder = r'C:\repo\data\1 - MoCap\4 - Alignment\elementary_action_pick\firstLeftHand'
    bvhfiles = glob.glob(data_folder + os.sep + '*.bvh')
    with open(data_folder + os.sep + 'timewarping.json', 'rb') as infile:
        timewarping_indices = json.load(infile)
    bvh_data = {}
    for bvhfile in bvhfiles:
        filename = os.path.split(bvhfile)[-1]
        bvhreader = BVHReader(bvhfile)
        bvh_data[filename] = bvhreader.frames.tolist()
    
    motion_data = {}
    for filename, warping_index in timewarping_indices.iteritems():
        if filename not in bvh_data.keys():
            print filename
            raise ValueError('cannot find bvh file in the folder')
        motion_data[filename] = {'frames': bvh_data[filename],
                                 'warping_index': warping_index}
    outfilename = r'C:\git-repo\ulm\morphablegraphs\test_data\constrction\fpca\motion_data.json'
    with open(outfilename, 'wb') as outfile:
        json.dump(motion_data, outfile)

class TestMotionDimensionReduction(object):
    
    def setup_class(self):
        config_params = ConstructionAlgorithmConfigurationBuilder('walk', 'left')
        skeleton_bvh = BVHReader(config_params.ref_bvh)
        with open(MOTION_DATA_FILE, 'rb') as infile:
            motion_data = json.load(infile)
        self.dimension_reduction = MotionDimensionReduction(motion_data,
                                                            skeleton_bvh,
                                                            config_params)
        self.dimension_reduction.convert_euler_to_quat()
        
    param_check_quat = [{'ref_quat': [0.57357643635104616, 0, 0, 0.8191520442889918],
                         'test_quat': [0.5, 0, 0, -0.8660254037844386],
                         'res': [-0.5, 0, 0, 0.8660254037844386]}] 
    @params(param_check_quat)
    def test_check_quat(self, test_quat, ref_quat, res):
        res_quat = self.dimension_reduction.check_quat(test_quat, ref_quat)
        for i in xrange(len(res_quat)):
            assert round(res_quat[i], 3) == round(res[i], 3)
    
    param_scale_rootchannels = [{'res': [12.6410959658, 101.367162709, 15.7177557248]}]

    @params(param_scale_rootchannels)
    def test_scale_rootchannels(self, res):
        self.dimension_reduction.scale_rootchannels()
        assert self.dimension_reduction.scale_vector == res

    param_get_quat_frames_from_euler = [{'res':(103, 79)}]

    @params(param_get_quat_frames_from_euler)
    def test_get_quat_frames_from_euler(self, res):
        test_filename = self.dimension_reduction.spatial_data.keys()[0]
        frames = self.dimension_reduction.spatial_data[test_filename]
        quat_frames = self.dimension_reduction.get_quat_frames_from_euler(frames)
        assert np.asarray(quat_frames).shape == res

