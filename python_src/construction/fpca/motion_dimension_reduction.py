# -*- coding: utf-8 -*-
"""
Created on Sun Aug 02 13:15:01 2015

@author: hadu01
"""

import sys
import os
import json
ROOT_DIR = os.sep.join(['..'] * 2)
sys.path.append(ROOT_DIR)
sys.path.append('..//')
from animation_data.quaternion_frame import QuaternionFrame
import numpy as np
from FPCA_temporal_data import FPCATemporalData
from FPCA_spatial_data import FPCASpatialData


class MotionDimensionReduction(object):

    def __init__(self, motion_data, skeleton_bvh, params):
        """
        * motion_data: dictionary
        \t{'filename': {'frames': euler frames, 'warping_index': warping frame index}}
        :param motion_data:
        :param skeleton_bvh:
        :param params:
        :return:
        """
        """
        :param motion_data:
        :param skeleton_bvh:
        :param params:
        :return:
        """
        self.params = params
        self.motion_data = motion_data
        self.spatial_data = {}
        self.temporal_data = {}
        self.len_quaterion = 4
        self.len_root_position = 3
        for filename, data in self.motion_data.iteritems():
            self.spatial_data[filename] = data['frames']
            self.temporal_data[filename] = data['warping_index']
        self.skeleton_bvh = skeleton_bvh
        self.n_frames = len(self.spatial_data[self.spatial_data.keys()[0]])

    def use_fpca_on_temporal_params(self):
        self.fpca_temporal = FPCATemporalData(self.temporal_data,
                                              self.params.n_basis_functions_temporal,
                                              self.params.npc_temporal)
        self.fpca_temporal.fpca_on_temporal_data()

    def use_fpca_on_spatial_params(self):
        self.convert_euler_to_quat()
        self.scale_rootchannels()
        self.fpca_spatial = FPCASpatialData(self.rescaled_quat_frames,
                                            self.params.n_basis_functions_spatial,
                                            self.params.fraction)
        self.fpca_spatial.fpca_on_spatial_data()

    def convert_euler_to_quat(self):
        self.quat_frames = {}
        for filename, frames in self.spatial_data.iteritems():
            self.quat_frames[filename] = self.get_quat_frames_from_euler(frames)

    def get_quat_frames_from_euler(self, frames):
        quat_frames = []
        last_quat_frame = None
        for frame in frames:
            quat_frame = QuaternionFrame(self.skeleton_bvh, frame)
            if last_quat_frame is not None:
                for joint_name in self.skeleton_bvh.node_names.keys():
                    if joint_name in quat_frame.keys():
                        quat_frame[joint_name] = self.check_quat(quat_frame[joint_name],
                                                                 last_quat_frame[joint_name])
            last_quat_frame = quat_frame
            root_translation = frame[0:3]
            quat_frame_values = [root_translation, ] + quat_frame.values()
            quat_frame_values = self.convert_quat_frame_value_to_array(quat_frame_values)
            quat_frames.append(quat_frame_values)
        return quat_frames
    
    def convert_quat_frame_value_to_array(self, quat_frame_values):
        n_channels = len(quat_frame_values)
        quat_channels = n_channels - 1
        self.n_dims = quat_channels * self.len_quaterion + self.len_root_position
        # in order to use Functional data representation from Fda(R), the
        # input data should be a matrix of shape (n_frames * n_samples *
        # n_dims)
        quat_frame_value_array = []
        for item in quat_frame_values:
            if not isinstance(item, list):
                item = list(item)
            quat_frame_value_array += item
        assert isinstance(quat_frame_value_array, list) and len(quat_frame_value_array) == self.n_dims, \
        ('The length of quaternion frame is not correct! ')
        return quat_frame_value_array
        
    
    def check_quat(self, test_quat, ref_quat):
        """check test quat needs to be filpped or not
        """
        test_quat = np.asarray(test_quat)
        ref_quat = np.asarray(ref_quat)
        dot = np.dot(test_quat, ref_quat)
        if dot < 0:
            test_quat = - test_quat
        return test_quat.tolist()

    def scale_rootchannels(self):
        """ Scale all root channels in the given frames.
        It scales the root channel by taking its absolut maximum 
        (max_x, max_y, max_z) and devide all values by the maximum, 
        scaling all positions between -1 and 1    
        """

        self.rescaled_quat_frames = {}
        max_x = 0
        max_y = 0
        max_z = 0
        for key, value in self.quat_frames.iteritems():
            tmp = np.asarray(value)
            max_x_i = np.max(np.abs(tmp[:, 0]))
            max_y_i = np.max(np.abs(tmp[:, 1]))
            max_z_i = np.max(np.abs(tmp[:, 2]))
            if max_x < max_x_i:
                max_x = max_x_i
            if max_y < max_y_i:
                max_y = max_y_i
            if max_z < max_z_i:
                max_z = max_z_i

        for key, value in self.quat_frames.iteritems():
            value = np.array(value)
            value[:, 0] /= max_x
            value[:, 1] /= max_y
            value[:, 2] /= max_z
            self.rescaled_quat_frames[key] = value.tolist()
        self.scale_vector = [max_x, max_y, max_z]

    def gen_data_for_modeling(self):
        self.use_fpca_on_temporal_params()
        self.use_fpca_on_spatial_params()
        self.fdata = {}
        self.fdata['motion_type'] = self.params.elementary_action + '_' + \
            self.params.motion_primitive
        self.fdata['spatial_parameters'] = self.fpca_spatial.fpcaobj.lowVs
        self.fdata['file_order'] = self.fpca_spatial.fileorder
        self.fdata['spatial_eigenvectors'] = self.fpca_spatial.fpcaobj.eigenvectors
        self.fdata['scale_vector'] = self.scale_vector
        self.fdata['n_frames'] = self.n_frames
        self.fdata['mean_motion'] = self.fpca_spatial.fpcaobj.centerobj.mean
        self.fdata['n_dim_spatial'] = self.params.n_basis_functions_spatial
        self.fdata['n_dim_spatial'] = self.n_dims
        self.fdata['n_basis'] = self.params.n_basis_functions_spatial
        self.fdata['temporal_pcaobj'] = self.fpca_temporal.temporal_pcaobj


def main():
    from construction_algorithm_configuration import ConstructionAlgorithmConfigurationBuilder
    from animation_data.bvh import BVHReader
    TESTDATAPATH = ROOT_DIR + os.sep + r'../test_data/constrction/fpca'
    with open(TESTDATAPATH + os.sep + 'motion_data.json') as infile:
        motion_data = json.load(infile)
    params = ConstructionAlgorithmConfigurationBuilder('pickLeft', 'first')
    skeleton_bvh = BVHReader(params.ref_bvh)
    dimension_reduction = MotionDimensionReduction(motion_data, skeleton_bvh, params)
    dimension_reduction.gen_data_for_modeling()
    with open(TESTDATAPATH + os.sep + 'fdata.json', 'wb') as infile:
        json.dump(dimension_reduction.fdata, infile)

if __name__ == '__main__':
    main()
