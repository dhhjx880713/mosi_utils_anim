# encoding: UTF-8
'''
This script is used to convert old format motion primitive model to MGRD compatible format
'''
from ...utilities.io_helper_functions import get_motion_primitive_filepath, \
                                                          load_json_file, \
                                                          write_to_json_file
import numpy
import scipy
import os


def gen_semnatic_motion_primitive_folder(elementary_action):
    folder_path = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95_temporal_semantic'
    assert os.path.exists(folder_path), ('Please configure motion primitive data path!')
    elementary_action_folder = folder_path + os.sep + 'elementary_action_' + elementary_action
    if not os.path.exists(elementary_action_folder):
        os.mkdir(elementary_action_folder)
    return elementary_action_folder

def get_old_motion_primitive_semantic_model_file(elementary_action,
                                                 motion_primitive):
    folder_path = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95\elementary_action_temporal_semantic_models'
    filename = elementary_action + '_' +  motion_primitive + '_quaternion_mm_with_semantic.json'
    return folder_path + os.sep + 'elementary_action_' + elementary_action + os.sep + filename


def get_old_motion_primitive_model_file(elementary_action,
                                        motion_primitive):
    folder_path = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95'
    elementary_action_folder = 'elementary_action_' + elementary_action
    filename = elementary_action + '_' +  motion_primitive + '_quaternion_mm.json'
    return os.path.join(folder_path,
                        elementary_action_folder,
                        filename)


def embedded_scale_factor_into_mean_and_eigen(translation_maxima,
                                              n_coeffs,
                                              n_dims,
                                              eigen_spatial,
                                              mean_spatial):
    eigen_spatial = numpy.asarray(eigen_spatial)
    mean_spatial = numpy.asarray(mean_spatial)
    index_list = [0, 1, 2]
    root_indices = []
    for row in range(n_coeffs):
        for idx in index_list:
            root_indices.append(n_dims * row + idx)
    indices_range = range(len(root_indices))
    x_indices = [root_indices[i] for i in indices_range if i%3 == 0]
    y_indices = [root_indices[i] for i in indices_range if i%3 == 1]
    z_indices = [root_indices[i] for i in indices_range if i%3 == 2]
    eigen_spatial[:, x_indices] *= translation_maxima[0]
    eigen_spatial[:, y_indices] *= translation_maxima[1]
    eigen_spatial[:, z_indices] *= translation_maxima[2]
    mean_spatial[x_indices] *= translation_maxima[0]
    mean_spatial[y_indices] *= translation_maxima[1]
    mean_spatial[z_indices] *= translation_maxima[2]
    return eigen_spatial, mean_spatial


def convert_spatial_motion_primitive_data_new_format(mm_data):
    if 'frame_time' not in mm_data.keys():
        mm_data['frame_time'] = 0.013889
    if 'animated_joints' not in mm_data.keys():
        mm_data['animated_joints'] = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm",
                                      "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm",
                                      "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg",
                                      "RightFoot"]
    if 'gmm_eigen' not in mm_data.keys():
        gaussian_eigens = gen_gaussian_eigen(mm_data['gmm_covars'])
        mm_data['gmm_eigen'] = gaussian_eigens.tolist()
    eigen_spatial, mean_spatial = embedded_scale_factor_into_mean_and_eigen(mm_data['translation_maxima'],
                                                                            mm_data['n_basis_spatial'],
                                                                            mm_data['n_dim_spatial'],
                                                                            mm_data['eigen_vectors_spatial'],
                                                                            mm_data['mean_spatial_vector'])
    motion_primitive_data = {}
    motion_primitive_data['sspm'] = {}
    motion_primitive_data['gmm'] = {}
    motion_primitive_data['sspm']['eigen'] = eigen_spatial.tolist()
    motion_primitive_data['sspm']['mean'] = mean_spatial.tolist()
    motion_primitive_data['sspm']['n_coeffs'] = mm_data['n_basis_spatial']
    motion_primitive_data['sspm']['n_dims'] = mm_data['n_dim_spatial']
    motion_primitive_data['sspm']['knots'] = mm_data['b_spline_knots_spatial']
    motion_primitive_data['sspm']['animated_joints'] = mm_data['animated_joints']
    motion_primitive_data['sspm']['degree'] = 3
    motion_primitive_data['gmm']['covars'] = mm_data['gmm_covars']
    motion_primitive_data['gmm']['means'] = mm_data['gmm_means']
    motion_primitive_data['gmm']['weights'] = mm_data['gmm_weights']
    motion_primitive_data['gmm']['eigen'] = mm_data['gmm_eigen']
    return motion_primitive_data


def covnert_motion_primitive_data_new_format(mm_data):
    if 'frame_time' not in mm_data.keys():
        mm_data['frame_time'] = 0.013889
    if 'animated_joints' not in mm_data.keys():
        mm_data['animated_joints'] = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm",
                                      "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm",
                                      "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg",
                                      "RightFoot"]
    if 'gmm_eigen' not in mm_data.keys():
        gaussian_eigens = gen_gaussian_eigen(mm_data['gmm_covars'])
        mm_data['gmm_eigen'] = gaussian_eigens.tolist()
    eigen_spatial, mean_spatial = embedded_scale_factor_into_mean_and_eigen(mm_data['translation_maxima'],
                                                                            mm_data['n_basis_spatial'],
                                                                            mm_data['n_dim_spatial'],
                                                                            mm_data['eigen_vectors_spatial'],
                                                                            mm_data['mean_spatial_vector'])
    motion_primitive_data = {}
    motion_primitive_data['sspm'] = {}
    motion_primitive_data['tspm'] = {}
    motion_primitive_data['gmm'] = {}
    motion_primitive_data['sspm']['eigen'] = eigen_spatial.tolist()
    motion_primitive_data['sspm']['mean'] = mean_spatial.tolist()
    motion_primitive_data['sspm']['n_coeffs'] = mm_data['n_basis_spatial']
    motion_primitive_data['sspm']['n_dims'] = mm_data['n_dim_spatial']
    motion_primitive_data['sspm']['knots'] = mm_data['b_spline_knots_spatial']
    motion_primitive_data['sspm']['animated_joints'] = mm_data['animated_joints']
    motion_primitive_data['sspm']['degree'] = 3
    if 'eigen_vectors_temporal_semantic' in mm_data.keys():
        motion_primitive_data['tspm']['eigen'] = mm_data['eigen_vectors_temporal_semantic']
    else:
        motion_primitive_data['tspm']['eigen'] = numpy.transpose(mm_data['eigen_vectors_time']).tolist()
    if 'mean_temporal_semantic_vector' in mm_data.keys():
        motion_primitive_data['tspm']['mean'] = mm_data['mean_temporal_semantic_vector']
    else:
        motion_primitive_data['tspm']['mean'] = mm_data['mean_time_vector']
    if 'n_basis_temporal_semantic' in mm_data.keys():
        motion_primitive_data['tspm']['n_coeffs'] = mm_data['n_basis_temporal_semantic']
    else:
        motion_primitive_data['tspm']['n_coeffs'] = mm_data['n_basis_time']
    if 'n_dim_temporal_semantic' in mm_data.keys():
        motion_primitive_data['tspm']['n_dims'] = mm_data['n_dim_temporal_semantic']
    else:
        motion_primitive_data['tspm']['n_dims'] = 1

    if 'b_spline_knots_temporal_semantic' in mm_data.keys():
        motion_primitive_data['tspm']['knots'] = mm_data['b_spline_knots_temporal_semantic']
    else:
        motion_primitive_data['tspm']['knots'] = mm_data['b_spline_knots_time']


    motion_primitive_data['tspm']['n_canonical_frames'] = mm_data['n_canonical_frames']

    if 'semantic_annotation' in mm_data.keys():
        motion_primitive_data['tspm']['semantic_labels'] = mm_data['semantic_annotation']
    else:
        motion_primitive_data['tspm']['semantic_labels'] = []
    motion_primitive_data['tspm']['frame_time'] = mm_data['frame_time']

    motion_primitive_data['tspm']['degree'] = 3
    motion_primitive_data['gmm']['covars'] = mm_data['gmm_covars']
    motion_primitive_data['gmm']['means'] = mm_data['gmm_means']
    motion_primitive_data['gmm']['weights'] = mm_data['gmm_weights']
    motion_primitive_data['gmm']['eigen'] = mm_data['gmm_eigen']
    return motion_primitive_data

def old_semantic_mm_to_new_semantic_mm_converter(elementary_action, motion_primitive):
    # add frame_time, gmm_eigen and animated_joints to old mm file if they are missing
    old_model_file = get_old_motion_primitive_semantic_model_file(elementary_action, motion_primitive)
    mm_data = load_json_file(old_model_file)
    motion_primitive_data = covnert_motion_primitive_data_new_format(mm_data)
    motion_primitive_file_name = os.path.split(old_model_file)[-1]
    output_folder = gen_semnatic_motion_primitive_folder(elementary_action)
    outputfilename = output_folder + os.sep + motion_primitive_file_name
    write_to_json_file(outputfilename, motion_primitive_data)

def old_mm_to_new_semantic_mm(elementary_action,
                              motion_primitive):
    old_model_file = get_old_motion_primitive_model_file(elementary_action,
                                                         motion_primitive)
    mm_data = load_json_file(old_model_file)
    new_mm_data = covnert_motion_primitive_data_new_format(mm_data)
    motion_primitive_file_name = os.path.split(old_model_file)[-1]
    output_folder = gen_semnatic_motion_primitive_folder(elementary_action)
    outputfilename = output_folder + os.sep + motion_primitive_file_name
    write_to_json_file(outputfilename, new_mm_data)


def gen_gaussian_eigen(covars):
    covars = numpy.asarray(covars)
    eigen = numpy.empty(covars.shape)
    for i, covar in enumerate(covars):
        s, U = scipy.linalg.eigh(covar)
        s.clip(0, out=s)
        numpy.sqrt(s, out=s)
        eigen[i] = U * s
        eigen[i] = numpy.transpose(eigen[i])
    return eigen


def convert_models_from_one_folder_to_another(src_folder,
                                              target_folder):
    for subfolder in os.walk(src_folder).next()[1]:
        subdir = os.path.join(src_folder, subfolder)
        for filename in os.walk(subdir).next()[2]:
            if 'mm_with_semantic.json' in filename:
                segs = filename.split('_')
                elementary_action = segs[0]
                motion_primitive = segs[1]
                try:
                    mm_data = load_json_file(os.path.join(subdir, filename))
                except:
                    raise IOError("Cannot open motion primitive file.")
                print(elementary_action)
                print(motion_primitive)
                new_mm_data = covnert_motion_primitive_data_new_format(mm_data)
                output_filename = os.path.join(target_folder,
                                               'elementary_action_'+elementary_action,
                                               '_'.join([elementary_action,
                                                         motion_primitive,
                                                         'quaternion_mm.json']))
                write_to_json_file(output_filename, new_mm_data)


def test():
    test_model = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95\elementary_action_lookAt\lookAt_lookAt_quaternion_mm.json'
    model_data = load_json_file(test_model)
    print(model_data.keys())
    new_model_data = covnert_motion_primitive_data_new_format(model_data)


if __name__ == "__main__":
    # elementary_action = 'pickBoth'
    # motion_primitive = 'retrieve'
    # old_semantic_mm_to_new_semantic_mm_converter(elementary_action, motion_primitive)
    src_folder = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95\elementary_action_temporal_semantic_models'
    target_folder = r'C:\repo\data\3 - Motion primitives\motion_primitives_quaternion_PCA95_temporal_semantic'
    convert_models_from_one_folder_to_another(src_folder,
                                              target_folder)
    # test()
    # elementary_action = 'transfer'
    # motion_primitive = 'transfer'
    # old_mm_to_new_semantic_mm(elementary_action,
    #                           motion_primitive)