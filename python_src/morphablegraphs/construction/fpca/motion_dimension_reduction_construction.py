# encoding: UTF-8
from motion_dimension_reduction import MotionDimensionReduction
from morphablegraphs.utilities import get_data_analysis_folder, get_aligned_data_folder, load_aligned_data
from morphablegraphs.construction.construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
from morphablegraphs.animation_data import BVHReader, Skeleton
import glob
import os
import sys
sys.path.append(os.path.dirname(__file__))


class MotionDimensionReductionConstructor(object):

    def __init__(self, elementary_action, motion_primitive):
        self.elemetary_action = elementary_action
        self.motion_primitive = motion_primitive

    def load(self):
        aligned_motion_data = load_aligned_data(self.elemetary_action,
                                                self.motion_primitive)
        params = ConstructionAlgorithmConfigurationBuilder(self.elemetary_action,
                                                           self.motion_primitive)
        skeleton_file = r'../../../skeleton.bvh'
        bvhreader = BVHReader(skeleton_file)
        skeleton = Skeleton(bvhreader)
        return MotionDimensionReduction(aligned_motion_data,
                                        bvhreader,
                                        params)
