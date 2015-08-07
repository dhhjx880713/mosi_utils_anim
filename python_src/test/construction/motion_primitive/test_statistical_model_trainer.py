# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 16:51:58 2015

@author: hadu01
"""

import os
import sys
ROOTDIR = os.sep.join(['..'] * 3)
import json
sys.path.append(ROOTDIR + os.sep)
TESTPATH = ROOTDIR + os.sep + r'construction/motion_primitive'
TESTLIBPATH = ROOTDIR + os.sep + 'test/'
sys.path.append(TESTPATH)
sys.path.append(TESTLIBPATH)
sys.path.append(ROOTDIR + os.sep + 'construction')
from statistical_model_trainer import StatisticalModelTrainer
from fpca.motion_dimension_reduction import  MotionDimensionReduction
from construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from animation_data.bvh import BVHReader
TEST_DATA_PATH = ROOTDIR + os.sep + r'../test_data/constrction/fpca'
TEST_RESULT_PATH = ROOTDIR + os.sep + r'../test_output/constrction'
from libtest import params, pytest_generate_tests


class TestStatisticalModelTrainer(object):
    
    "Unittest for StatisticalModelTrainer class"
    
    def setup_class(self):
        with open(TEST_DATA_PATH + os.sep + 'motion_data.json') as infile:
            motion_data = json.load(infile)
        params = ConstructionAlgorithmConfigurationBuilder('pickLeft', 'first')
        skeleton_bvh = BVHReader(params.ref_bvh)
        dimension_reduction = MotionDimensionReduction(motion_data, skeleton_bvh, params)
        dimension_reduction.gen_data_for_modeling()
        self.modelTrainer = StatisticalModelTrainer(dimension_reduction.fdata)
        # self.modelTrainer.gen_motion_primitive_model()
    

    
    param_combine_spatial_temporal_parameters = [{"res": 11}]
    
    @params(param_combine_spatial_temporal_parameters)
    
    def test_combine_spatial_temporal_parameters(self, res):
        """
        Unit test for _combine_spatial_temporal_parameters
        """
        self.modelTrainer._combine_spatial_temporal_parameters()
        assert self.modelTrainer._motion_parameters.shape[1] == res
    
    param_train_gmm = [{"res": 3}]
    @params(param_train_gmm)
    def test_train_gmm(self, res):
        """
        Unit test for _train_gmm
        """
        self.modelTrainer._train_gmm()
        assert self.modelTrainer.numberOfGaussian == res
    
    param_create_gmm = [{"res": 3}]
    @params(param_create_gmm)
    def test_create_gmm(self, res):
        """
        Unit test for _create_gmm
        """
        self.modelTrainer._create_gmm()
        assert len(self.modelTrainer.gmm.weights_) == res
    
    param_save_model = [{"save_path": TEST_RESULT_PATH,
                         "res": TEST_RESULT_PATH + os.sep + 'pickLeft_first_quaternion_mm.json'}]
    @params(param_save_model)
    def test_save_model(self, save_path, res):
        """
        Unit test for _save_model
        """
        self.modelTrainer._train_gmm()
        self.modelTrainer._create_gmm()
        self.modelTrainer._save_model(save_path)
        assert os.path.isfile(res)
