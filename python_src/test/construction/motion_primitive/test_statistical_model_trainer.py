# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 16:51:58 2015

@author: hadu01
"""

import os
import json
from ....morphablegraphs.construction.motion_primitive.statistical_model_trainer import StatisticalModelTrainer
from ....morphablegraphs.construction.fpca.motion_dimension_reduction import  MotionDimensionReduction
from ....morphablegraphs.construction.construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from ....morphablegraphs.animation_data.bvh import BVHReader
from ...libtest import params, pytest_generate_tests
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-4]) + os.sep
TEST_DATA_PATH = ROOTDIR + os.sep + r'../test_data/constrction/fpca'
TEST_RESULT_PATH = ROOTDIR + os.sep + r'../test_output/constrction'


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

    
    param_combine_spatial_temporal_parameters = [{"res": 11}]
    
    @params(param_combine_spatial_temporal_parameters)
    
    def test_combine_spatial_temporal_parameters(self, res):
        """
        Unit test for _combine_spatial_temporal_parameters
        """
        self.modelTrainer.comb_params()
        assert self.modelTrainer._motion_parameters.shape[1] == res
    
    param_train_gmm = [{"res": 3}]
    @params(param_train_gmm)
    def test_train_gmm(self, res):
        """
        Unit test for _train_gmm
        """
        self.modelTrainer._train_gmm()
        assert self.modelTrainer.n_gaussians == res
    
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
