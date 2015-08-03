# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:15:52 2015

@author: du
"""

import os
import sys
ROOT_DIR = os.sep.join(['..'] * 1)
sys.path.append(ROOT_DIR)
from preprocessing.preprocessing import Preprocessor
from construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from fpca.motion_dimension_reduction import MotionDimensionReduction
from motion_primitive.statistical_model_trainer import StatisticalModelTrainer

def main():
    # initialization
    elementary_action = 'walk'
    motion_primitive = 'sidestepLeft'   
    params = ConstructionAlgorithmConfigurationBuilder(elementary_action,
                                                       motion_primitive)
    # preprocessing
    preprocessor = Preprocessor(params)
    preprocessor.preprocess()
    # dimension reduction
    dimension_reduction = MotionDimensionReduction(preprocessor.warped_motions,
                                                   preprocessor.bvhreader,
                                                   params)  
    dimension_reduction.gen_data_for_modeling()                                             
    # statistical modeling
    modelTrainer = StatisticalModelTrainer(dimension_reduction.fdata,
                                           params.save_path)
    modelTrainer.gen_motion_primitive_model()                                       

if __name__ == "__main__":
    main()                                   