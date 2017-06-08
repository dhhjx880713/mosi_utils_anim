# encoding: UTF-8
from ..fpca.fpca_time_semantic import FPCATimeSemantic
from statistical_model_trainer import StatisticalModelTrainer
import os
import json
from ...utilities.io_helper_functions import get_aligned_data_folder, \
                                                          get_low_dimensional_spatial_file, \
                                                          get_motion_primitive_filepath
from ..preprocessing import gen_walk_annotation, \
                           gen_synthetic_semantic_annotation_pick_and_place, \
                           gen_synthetic_semantic_annotation_for_screw, \
                           gen_synthetic_semantic_annotation_for_transfer
from motion_primitive_converter import covnert_motion_primitive_data_new_format
from ...utilities.io_helper_functions import write_to_json_file


def get_motion_primitive_model_dir():
    motion_primitive_model_dir = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95\elementary_action_temporal_semantic_models'
    if not os.path.exists(motion_primitive_model_dir):
        raise IOError('Please configure motion primitive folder')
    return motion_primitive_model_dir


def create_semantic_motion_primitive(elementary_action,
                                     motion_primitive,
                                     update_semantic_annotation=False):
    print(elementary_action)
    print(motion_primitive)
    aligned_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
    temporal_semantic_low_dimensional_data_file = aligned_data_folder + os.sep + 'temporal_semantic_low_dimensional_data.json'
    spatial_temporal_file = get_low_dimensional_spatial_file(elementary_action, motion_primitive)
    motion_primitive_file = get_motion_primitive_filepath(elementary_action, motion_primitive)
    # add missing values to spatial_temporal_file if they are missing
    with open(spatial_temporal_file, 'r') as infile:
        spatial_temporal_data = json.load(infile)

    if 'n_dim_spatial' not in spatial_temporal_data.keys():
        with open(motion_primitive_file, 'r') as infile:
            motion_primitive_data = json.load(infile)
        spatial_temporal_data['n_dim_spatial'] = motion_primitive_data['n_dim_spatial']
        spatial_temporal_data['mean_spatial_vector'] = motion_primitive_data['mean_spatial_vector']

    # if temporal_semantic_annotation file is not in the data folder, then create one
    if not os.path.exists(temporal_semantic_low_dimensional_data_file) or update_semantic_annotation:
        print('create temporal_semantic_low_dimensional_data file')
        timewarping_file = aligned_data_folder + os.sep + 'timewarping.json'
        semantic_annotation_file = aligned_data_folder + os.sep + '_'.join([elementary_action,
                                                                            motion_primitive,
                                                                            'semantic',
                                                                            'annotation.json'])
        if not os.path.exists(semantic_annotation_file) or update_semantic_annotation:
            if 'pick' in elementary_action.lower() or 'place' in elementary_action.lower():
                print('create synthetic semantic annotation for ' + elementary_action)
                gen_synthetic_semantic_annotation_pick_and_place(elementary_action,
                                                                 motion_primitive)
            elif 'walk' in elementary_action.lower() or 'carry' in elementary_action.lower():
                print('create synthetic semantic annotation for ' + elementary_action)
                gen_walk_annotation(elementary_action,
                                    motion_primitive)
            elif 'screw' in elementary_action.lower():
                print('create synthetic semnatic annotation for ' + elementary_action)
                gen_synthetic_semantic_annotation_for_screw(elementary_action,
                                                            motion_primitive)
            elif 'transfer' in elementary_action.lower():
                print('create synthetic semnatic annotation for ' + elementary_action)
                gen_synthetic_semantic_annotation_for_transfer(elementary_action,
                                                               motion_primitive)
            else:
                raise KeyError('Unknow action type')
        fpca_temporal_semantic = FPCATimeSemantic()
        fpca_temporal_semantic.load_time_warping_data(timewarping_file)
        fpca_temporal_semantic.load_semantic_annotation(semantic_annotation_file)
        fpca_temporal_semantic.merge_temporal_semantic_data()
        fpca_temporal_semantic.functional_pca()
        temporal_semantic_file = aligned_data_folder + os.sep + 'temporal_semantic_low_dimensional_data.json'
        fpca_temporal_semantic.save_data(temporal_semantic_file)
    motion_primitive_path = get_motion_primitive_model_dir()
    model_trainer = StatisticalModelTrainer(save_path=motion_primitive_path)
    model_trainer.load_data_from_file(spatial_temporal_data,
                                      temporal_semantic_low_dimensional_data_file)
    model_trainer.weight_temporal_semantic_data(0.1)

    mm_data = model_trainer.gen_motion_primitive_model()
    new_format_data = covnert_motion_primitive_data_new_format(mm_data)
    target_folder = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95_temporal_semantic'
    if motion_primitive == 'first':
        motion_primitive = 'reach'
    if motion_primitive == 'second':
        motion_primitive = 'retrieve'
    elementary_action_folder = os.path.join(target_folder,
                                            'elementary_action_' + elementary_action)
    if not os.path.exists(elementary_action_folder):
        os.mkdir(elementary_action_folder)

    output_filename = os.path.join(elementary_action_folder,
                                   '_'.join([elementary_action,
                                             motion_primitive,
                                             'quaternion_mm.json']))
    write_to_json_file(output_filename, new_format_data)


def train_multiple_semantic_motion_primitives():
    aligned_data_folder = r'C:\repo\data\1 - MoCap\4 - Alignment'
    for elementary_action_folder in os.walk(aligned_data_folder).next()[1]:
        elementary_action = elementary_action_folder.split('_')[-1]

        print(elementary_action)
        if 'carry' in elementary_action.lower():
            for motion_primitive_folder in os.walk(os.path.join(aligned_data_folder,
                                                                elementary_action_folder)).next()[1]:
                print('_'.join([elementary_action,
                                motion_primitive_folder]))
                create_semantic_motion_primitive(elementary_action,
                                                 motion_primitive_folder,
                                                 update_semantic_annotation=True)

if __name__ == "__main__":
    elementary_action = 'walk'
    motion_primitive = 'leftStance_female'
    create_semantic_motion_primitive(elementary_action,
                                     motion_primitive,
                                     update_semantic_annotation=True)
