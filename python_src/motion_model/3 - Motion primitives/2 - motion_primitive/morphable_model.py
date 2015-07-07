# -*- coding: utf-8 -*-
"""
Created on Thu Feb 05 11:41:58 2015

@author: hadu01
"""

from motion_sample import MotionSample
from motion_primitive import MotionPrimitive
import json
import os
import numpy as np
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
ROOT_DIR = os.sep.join(['..'] * 3)


def get_input_data_folder():
    '''
    Return folder path without trailing os.sep
    '''
    data_dir_name = 'data'
    motion_primitives_dir = '2 - PCA'
    model_type = 'combine_spatial_temporal'
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             motion_primitives_dir,
                             model_type])
    return input_dir


def get_motion_primitive_folder():
    '''
    Return folder path without trailing os.sep
    '''
    data_dir_name = 'data'
    motion_primitives_dir = '3 - Motion primitives'
    model_type = 'motion_primitives_quaternion_PCA95'
    elementray_action_type = 'elementary_action_walk'
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             motion_primitives_dir,
                             model_type,
                             elementray_action_type])
    return input_dir


def clean_path(path):
    """
    Generate absolute path starting with '\\\\?\\' to avoid failure of loading
    because of long path in windows

    Parameters
    ----------
    * path: string
    \tRelative path

    Return
    ------
    * path: string
    \tAbsolute path starting with '\\\\?\\'
    """
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [directory for directory in os.getcwd().split(os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] + [directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path


class MorphableModel(MotionPrimitive):

    def __init__(self, motionPrimitiveFile):
        MotionPrimitive.__init__(self, motionPrimitiveFile)

    def _back_projection(self, low_dimensional_vector):
        """Backproject a low_dimensional_vector to MotionSample object
        """
        spatial_coefs = self._inverse_spatial_pca(low_dimensional_vector[:self.s_pca["n_components"]])
        time_fd = self._inverse_temporal_pca(low_dimensional_vector[self.s_pca["n_components"]:])
        return MotionSample(spatial_coefs, self.n_canonical_frames, time_fd)

    def back_projection_training_data(self, dataFile, saveFolder=None):
        """Back project all the low dimensional training data to a folder
        """
        with open(dataFile, 'rb') as infile:
            data = json.load(infile)
            infile.close()
        low_dimensional_motion_data_dic = data['motion_data']
        file_order = low_dimensional_motion_data_dic.keys()
        low_dimensional_motion_data = np.asarray(low_dimensional_motion_data_dic.values())
        n_samples = len(low_dimensional_motion_data)
        for i in xrange(n_samples):
            back_motion = self._back_projection(low_dimensional_motion_data[i])
            if saveFolder is None:
                filename = file_order[i]
            else:
                if saveFolder[-1] == os.sep:
                    save_folder = saveFolder
                else:
                    save_folder = saveFolder + os.sep
                filename = save_folder + file_order[i]
            back_motion.save_motion_vector(filename)

    def eval_probs_for_sample(self, low_dimensional_vector):
        """For given low dimensional vector, evalute its probabilities for each
        Gaussian
        """
        tmp = np.reshape(low_dimensional_vector,
                         (1, len(low_dimensional_vector)))
        log_likelihoods = _log_multivariate_normal_density_full(tmp,
                                                                self.gmm.means_,
                                                                self.gmm.covars_)
        log_likelihoods = np.ravel(log_likelihoods)
        # use relative value than absolute value to avoid extreme small value
        # normalize probability in the range (0, 1)
        vmax = np.max(log_likelihoods)
        log_likelihoods = log_likelihoods - vmax
        probs = np.exp(log_likelihoods)
        return probs

    def gen_Svector_for_clusters(self, dataFile, save_folder=None):
        """Catergorize training samples based on their probabilities from each
           Gaussian, save classification result as a json file
        """
        with open(dataFile, 'rb') as infile:
            data = json.load(infile)
            infile.close()
        low_dimensional_motion_data_dic = data['motion_data']
        file_order = low_dimensional_motion_data_dic.keys()
        low_dimensional_motion_data = np.asarray(low_dimensional_motion_data_dic.values())
        n_clusters = len(self.gmm.weights_)
        # initialize clusters
        clusters = []
        for i in xrange(n_clusters):
            tmp = {}
            tmp['mean'] = self.gmm.means_[i].tolist()
            tmp['covar'] = self.gmm.covars_[i].tolist()
            tmp['weight'] = self.gmm.weights_[i]
            tmp['n_samples'] = 0
            tmp['files'] = []
            tmp['low_dimension_data'] = []
            clusters.append(tmp)
        n_samples = len(low_dimensional_motion_data)
        tmp = low_dimensional_motion_data.tolist()
        for i in xrange(n_samples):
            probs = self.eval_probs_for_sample(tmp[i])
            index = max(xrange(len(probs)), key=probs.__getitem__)
            clusters[index]['files'].append(file_order[i])
            clusters[index]['low_dimension_data'].append(tmp[i])
            clusters[index]['n_samples'] += 1
        output_data = {}
        for i in xrange(len(clusters)):
            output_data['class_' + str(i)] = clusters[i]
        if save_folder is None:
            output_filename = self.name + '_s_vector.json'
        else:
            if save_folder[-1] == os.sep:
                save_folder = save_folder
            else:
                save_folder = save_folder + os.sep
            output_filename = save_folder + self.name + '_s_vector.json'
        with open(output_filename, 'wb') as outfile:
            json.dump(output_data, outfile)
            outfile.close()


if __name__ == '__main__':
    elementary_motion = 'walk'
    motion_primitive = 'rightStance'
    motion_data_dir = get_input_data_folder()
    if len(motion_data_dir) > 116:
        motion_data_dir = clean_path(motion_data_dir)
    motionDataFile = motion_data_dir + os.sep + '%s_%s_low_dimensional_motion_data.json' % (elementary_motion, motion_primitive)
    motion_primitive_dir = get_motion_primitive_folder()
    if len(motion_primitive_dir) > 116:
        motion_primitive_dir = clean_path(motion_primitive_dir)
    motionPrimitiveFile = motion_primitive_dir + os.sep + '%s_%s_quaternion_mm.json' % (elementary_motion, motion_primitive)   
    mm = MorphableModel(motionPrimitiveFile)
    mm.gen_Svector_for_clusters(motionDataFile)
#    with open(motionDataFile, 'rb') as infile:
#        data = json.load(infile)
#        infile.close()
#    low_dimension_dic = data['motion_data']
#    file_order = low_dimension_dic.keys()
#    low_dimension_data = low_dimension_dic.values()
#    index = 1
#    print 'file name is: ' + file_order[index]
#    probs = mm.eval_probs_for_sample(low_dimension_data[index])
#    print probs
#    mm.back_projection_training_data(saveFolder=r'backprojectionMotion/')
#def load_data(filename):
#    """
#    Load low dimensional data from json file
#    """
#    with open(filename, rb) as infile:
#        data = json.load(infile)
#        infile.close()
#    low_dimensional_motion_data = np.asarray(data['motion_data'])
#    n_samples = len(low_dimensional_motion_data) 
#    print 'number of samples: ' + str(n_samples)
#    eigen_vector_spatial = np.asarray(data['eigen_vectors_spatial'])
#    print eigen_vector_spatial.shape
#    n_pcs_spatial = eigen_vector_spatial[1]
#    for i in xrange(n_samples):

        
#    low_dimensional_vector = np.ravel(self.gmm.sample())
#    print low_dimensional_vector
#    spatial_coefs = self._inverse_spatial_pca(low_dimensional_vector[:self.s_pca["n_components"]])
#    if self.use_time_parameters:
#        time_fd = self._inverse_temporal_pca(low_dimensional_vector[self.s_pca["n_components"]:])
#    else:
#        time_fd = None
#    return MotionSample(spatial_coefs, self.n_canonical_frames, time_fd)        