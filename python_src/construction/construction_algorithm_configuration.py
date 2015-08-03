# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:28:00 2015

@author: du
"""

import os
import sys
ROOT_DIR = os.sep.join(['..'] * 1)
sys.path.append(ROOT_DIR)
SERVICE_CONFIG_FILE = ROOT_DIR + os.sep + "config" + os.sep + "service.json"
from utilities.io_helper_functions import load_json_file

class ConstructionAlgorithmConfigurationBuilder(object):
        
    def __init__(self, elementary_action, motion_primitive):
        self.ref_orientation = {'x': 0, 'y': 0, 'z': -1}
        self.ref_position = {'x': 0, 'y': 0, 'z': 0}
        self.ref_bvh = ROOT_DIR + os.sep + 'skeleton.bvh'
        self.n_basis_functions_spatial = 7
        self.n_basis_functions_temporal = 8
        self.touch_ground_joint = 'Bip01_R_Toe0'
        self.align_frame_idx = 0
        self.npc_temporal = 3
        self.fraction = 0.95
        self.elementary_action = elementary_action
        self.motion_primitive = motion_primitive
        path_data = load_json_file(SERVICE_CONFIG_FILE)
        self.data_path = path_data['data_folder']
        self.save_path = path_data['model data']
        self._get_annotation_file()
        self._get_retarget_folder()
    
    def _get_annotation_file(self):
        self.annotation = self.retarget_folder + os.sep + 'key_frame_annotation.txt'
    
    def _get_retarget_folder(self):
        self.retarget_folder = self.data_path + os.sep + r'2 - Rocketbox retargeting\Take_sidestep' 
    