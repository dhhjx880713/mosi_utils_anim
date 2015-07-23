# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:15:52 2015

@author: du
"""

import os
import sys
ROOT_DIR = os.sep.join(['..'] * 1)
sys.path.append(ROOT_DIR)
from preprocessing.motion_segmentation import MotionSegmentation
from preprocessing.motion_normalization import MotionNormalization
from preprocessing.motion_dtw import MotionDTW
from fpca.preprocessing import load_data
from fpca.fpca_spatial import fpca_spatial
from fpca.z_t import z_t_transform
from fpca.b_splines import b_splines
from fpca.temporal import fpca_temporal
from motion_primitive.statistical_model_trainer import StatisticalModelTrainer
from utilities.io_helper_functions import load_json_file

SERVICE_CONFIG_FILE = ROOT_DIR + os.sep + "config" + os.sep + "service.json"

def main():
    # initialization
    elementary_action = 'walk'
    motion_primitive = 'sidestepLeft'
    path_data = load_json_file(SERVICE_CONFIG_FILE)
    data_path = path_data['data_folder']
    model_path = path_data['model data']
    retarget_folder = data_path + os.sep + r'2 - Rocketbox retargeting\Take_sidestep'
    cutting_folder = data_path + os.sep + r'3 - Cutting\test'
    dtw_folder = data_path + os.sep + r'4 - Alignment\walk\sidestepLeft
    motion_primitive_folder = 
    annotation = retarget_folder + os.sep + 'key_frame_annotation.txt'    
    
    # motion segmentation
    motion_segmentor = MotionSegmentation(retarget_folder, annotation,
                                          cutting_folder)
    motion_segmentor.segment_motions(elementary_action,
                                     motion_primitive)
    
    # motion normalization
    ref_orientaiton = [0, -1]   
    ref_position = [0, 0, 0]
    ref_bvh = ROOT_DIR + os.sep + 'skeleton.bvh'
    touch_ground_joint = 'Bip01_R_Toe0'
    motion_normalizer = MotionNormalization(motion_segmentor.cut_motions,
                                            ref_orientaiton,
                                            ref_position,
                                            ref_bvh,
                                            touch_ground_joint)   
    motion_normalizer.normalize_root()
    motion_normalizer.align_motion()  
                                      
    # motion alignment (to be implemented)
    dynamic_time_warper = MotionDTW(motion_normalizer.aligned_motions)     
    dynamic_time_warper.dtw()
    dynamic_time_warper.save_warpped_motion(dtw_folder)
    dynamic_time_warper.save_time_function(dtw_folder)
    # dimension reduction
    # spatial
    spatial_data, root_scale = load_data(dtw_folder)
    n_frames, n_samples, n_dims = spatial_data.shape
    n_basis = 7
    spatial_fpcaobj = fpca_spatial(spatial_data, root_scale, n_basis)
    
    # temporal
    z_t = z_t_transform(dtw_folder+os.sep+'timewarping.json')
    n_knots = 8
    bspline_temporal = b_splines(z_t, n_knots)
    temporal_fpcaobj = fpca_temporal(bspline)

    fdata = {}
    fdata['motion_type'] = elementary_action + '_' + motion_primitive
    fdata['spatial_parameters'] = spatial_fpcaobj.lowVs.tolist()
    fdata['file_order'] = fileorder
    fdata['spatial_eigenvectors'] = spatial_fpcaobj.eigenvectors.tolist()
    fdata['n_frames'] = n_frames
    fdata['n_basis'] = n_basis
    fdata['scale_vector'] = root_scale
    fdata['mean_motion'] = spatial_fpcaobj.centerobj.mean.tolist()
    fdata['n_dim_spatial'] = n_dims    
    
    # statistical modeling
    modelTrainer = StatisticalModelTrainer(fdata, temporal_fpcaobj)
    modelTrainer._train_gmm()
    modelTrainer._create_gmm()
    modelTrainer._save_model(model_path)    
    

if __name__ == "__main__":
    main()                                   