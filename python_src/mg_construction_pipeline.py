# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:15:52 2015

@author: du
"""

import sys
from .morphablegraphs.construction.preprocessing.preprocessing import Preprocessor
from .morphablegraphs.construction.construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from .morphablegraphs.construction.fpca.motion_dimension_reduction import MotionDimensionReduction
from .morphablegraphs.construction.motion_primitive.statistical_model_trainer import StatisticalModelTrainer


def main():
    if len(sys.argv) != 3:
        raise IOError(('please give elementary action and motion primitive name. E.g.: morphable_model_construction_pipeline.py walk sidestepLeft'))
    elementary_action = sys.argv[1]
    motion_primitive = sys.argv[2]
    params = ConstructionAlgorithmConfigurationBuilder(elementary_action,
                                                       motion_primitive)
    preprocessor = Preprocessor(params)
    preprocessor.preprocess()
    dimension_reduction = MotionDimensionReduction(preprocessor.warped_motions,
                                                   preprocessor.bvhreader,
                                                   params)
    dimension_reduction.gen_data_for_modeling()
    modelTrainer = StatisticalModelTrainer(dimension_reduction.fdata,
                                           params.save_path)
    modelTrainer.gen_motion_primitive_model()

if __name__ == "__main__":
    main()
