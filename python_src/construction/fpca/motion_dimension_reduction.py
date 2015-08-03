# -*- coding: utf-8 -*-
"""
Created on Sun Aug 02 13:15:01 2015

@author: hadu01
"""

import sys
import os
ROOT_DIR = os.sep.join(['..'] * 2)
sys.path.append(ROOT_DIR)
from animation_data.quaternion_frame import QuaternionFrame
import numpy as np
from FPCA_temporal_data import FPCATemporalData
from FPCA_spatial_data import FPCASpatialData
 
class MotionDimensionReduction(object):
    
    def __init__(self, motion_data, skeleton_bvh, params):
        self.params = params
        self.motion_data = motion_data
        self.spatial_data = {}
        self.temporal_data = {}
        for filename, data in self.motion_data:
            self.spatial_data[filename] = data['frames']
            self.temporal_data[filename] = data['warping_index']
        self.skeleton_bvh = skeleton_bvh
    
    def use_fpca_on_temporal_params(self):
        self.fpca_temporal = FPCATemporalData(self.temporal_data,
                                         self.params.n_basis_functions_temporal,
                                         self.params.npc_temporal)
        self.fpca_temporal.fpca_on_temporal_data()   
                                      
    
    def use_pca_on_spatial_params(self):
        print "To be implemented"
    
    def use_fpca_on_spatial_params(self):
        self.convert_euler_to_quat()
        self.scale_rootchannels()
        self.fpca_spatial = FPCASpatialData(self.rescaled_quat_frames,
                                            self.params.n_basis_functions_spatial,
                                            self.params.fraction)
        self.fpca_spatial.fpca_on_spatial_data()                               
    
    def convert_euler_to_quat(self):
        """Convert euler frames to quaternion frames
        """
        self.quat_frames = {}
        for filename, frames in self.spatial_data.iteritems():
            self.quat_frames[filename] = self.get_quat_frames_from_euler(frames)

    def get_quat_frames_from_euler(self, frames):
        quat_frames = []
        last_quat_frame = None
        for frame in frames:
            quat_frame = QuaternionFrame(self.skeleton_bvh, frame)
            if last_quat_frame is not None:
                for joint_name in self.skeleton_bvh.node_name.keys():
                    if joint_name in quat_frame.keys():
                        dot = quat_frame[joint_name][0]*last_quat_frame[joint_name][0] + \
                              quat_frame[joint_name][1]*last_quat_frame[joint_name][1] + \
                              quat_frame[joint_name][2]*last_quat_frame[joint_name][2] + \
                              quat_frame[joint_name][3]*last_quat_frame[joint_name][3]                        
                        if dot < 0:
                            quat_frame[joint_name] = [-v for v in quat_frame[joint_name]]              
            last_quat_frame = quat_frame
            root_translation = frame[0:3] 
            quat_frame_values = [root_translation,]+quat_frame.values()
            quat_frames.append(quat_frame_values)
        return quat_frames

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
            tmp = np.array(value)
            # Bit confusing conversion needed here, since of numpys view system
            rootchannels = tmp[:, 0].tolist()                                              
            rootchannels = np.array(rootchannels)
            max_x_i = np.max(np.abs(rootchannels[:, 0]))
            max_y_i = np.max(np.abs(rootchannels[:, 1]))
            max_z_i = np.max(np.abs(rootchannels[:, 2]))
           
            if max_x < max_x_i:
                max_x = max_x_i
                
            if max_y < max_y_i:
                max_y = max_y_i
                
            if max_z < max_z_i:
                max_z = max_z_i
                
        for key, value in self.quat_frames.iteritems():
            tmp = np.array(value)        
            # Bit confusing conversion needed here, since of numpys view system
            rootchannels = tmp[:, 0].tolist()                                              
            rootchannels = np.array(rootchannels)        
            
            rootchannels[:, 0] /= max_x
            rootchannels[:, 1] /= max_y
            rootchannels[:, 2] /= max_z
            
            self.rescaled_quat_frames[key] = value
            for frame in xrange(len(tmp)):
                self.rescaled_quat_frames[key][frame][0] = tuple(rootchannels[frame].tolist())
        self.scale_vector = [max_x, max_y, max_z]
    
    def gen_data_for_modeling(self):
        self.fdata = {}
        self.fdata['motion_type'] = self.params.elementary_action + '_' + \
                                    self.params.motion_primtive
        self.fdata['spatial_parameters'] = self.fpca_spatial.fpcaobj.lowVs
        self.fdata['file_order'] = self.fpca_spatial.fileorder
        self.fdata['spatial_eigenvectors'] = self.fpca_spatial.fpcaobj.eigenvectors
        self.fdata['scale_vector'] = self.scale_vector
        self.fdata['n_frames'] = self.fpca_spatial.n_frames
        self.fdata['mean_motion'] = self.fpca_spatial.fpcaobj.centerobj.mean
        self.fdata['n_dim_spatial'] = self.params.n_basis_functions_spatial
        self.fdata['n_dim_spatial'] = self.fpca_spatial.n_dims
        self.fdata['n_basis'] = self.params.n_basis_functions_spatial
        self.fdata['temporal_pcaobj'] = self.fpca_temporal.temporal_pcaobj
