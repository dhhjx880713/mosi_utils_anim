# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 16:51:58 2015

@author: hadu01
"""

import os
import sys
import numpy as np
from libtest import params, pytest_generate_tests
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(1, TESTPATH)
sys.path.insert(1, TESTPATH + (os.sep + os.pardir))

from statistical_model_trainer import StatisticalModelTrainer


class TestStatisticalModelTrainer(object):
    
    "Unittest for StatisticalModelTrainer class"
    
    def setup_method(self, method):
        spatial_data_file = TESTPATH + 'walk_leftStance_low_dimensional_data.json'
        temporal_data_file = TESTPATH + 'b_splines_walk_leftStance.RData'
        self.modelTrainer = StatisticalModelTrainer(spatial_data_file,
                                                    temporal_data_file)

    param_load_spatial_data = [{"spatial_data_file": 
        TESTPATH + 'walk_leftStance_low_dimensional_data.json',
        "res": 7}] 
                                                    
    @params(param_load_spatial_data)                
      
    def test_load_spatial_data(self, spatial_data_file, res):
        """
        Unit test for _load_spatial_data
        """
        self.modelTrainer._load_spatial_data(spatial_data_file)
        assert self.modelTrainer._n_basis == res
    
    param_load_temporal_data = [{"temporal_data_file": 
        TESTPATH + 'b_splines_walk_leftStance.RData', 
        "res": [620, 3]}]

    @params(param_load_temporal_data)
    def test_load_temporal_data(self, temporal_data_file, res):
        """
        Unit test for _load_temporal_data
        """
        self.modelTrainer._load_temporal_data(temporal_data_file)
        assert self.modelTrainer._temporal_parameters.shape[0] == res[0]
        assert self.modelTrainer._temporal_parameters.shape[1] == res[1]
    
    param_combine_spatial_temporal_parameters = [{"res": 9}]
    
    @params(param_combine_spatial_temporal_parameters)
    
    def test_combine_spatial_temporal_parameters(self, res):
        """
        Unit test for _combine_spatial_temporal_parameters
        """
        self.modelTrainer._combine_spatial_temporal_parameters()
        assert self.modelTrainer._motion_parameters.shape[1] == res
    
    param_train_gmm = [{"res": 12}]
    @params(param_train_gmm)
    def test_train_gmm(self):
        """
        Unit test for _train_gmm
        """
        self.modelTrainer._train_gmm()
        assert self.modelTrainer.numberOfGaussian == 12
    
    param_create_gmm = [{"res": 12}]
    @params(param_create_gmm)
    def test_create_gmm(self, res):
        """
        Unit test for _create_gmm
        """
        self.modelTrainer._create_gmm()
        assert len(self.modelTrainer.gmm.weights_) == res
    
    param_save_model = [{"filename": 'walk_leftStance_quaternion_mm.json'}]
    @params(param_save_model)
    def test_save_model(self, filename):
        """
        Unit test for _save_model
        """
        self.modelTrainer._save_model(filename)
        assert os.path.isfile(filename)
