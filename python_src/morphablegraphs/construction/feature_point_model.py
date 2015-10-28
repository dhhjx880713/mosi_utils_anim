__author__ = 'hadu01'

import numpy as np
from ..motion_model.motion_primitive import MotionPrimitive
import json
from ..animation_data.motion_editing import pose_orientation_quat, \
                                            get_cartesian_coordinates_from_quaternion
import matplotlib.pyplot as plt
from .motion_primitive.gmm_trainer import GMMTrainer
from sklearn import mixture
from ..animation_data.motion_editing import BVHReader
from ..animation_data.skeleton import Skeleton


class FeaturePointModel(object):
    def __init__(self, motion_primitive_file, skeleton_file):
        self.motion_primitive_model = MotionPrimitive(motion_primitive_file)
        bvhreader = BVHReader(skeleton_file)
        self.skeleton = Skeleton(bvhreader)
        self.low_dimension_vectors = []
        self.feature_points = []
        self.orientations = []
        self.feature_point = None

    def create_feature_points(self, joint_name, n):
        """
        Create a set of samples, calculate target point position and orientation
        The motion sample is assumed already to be well aligned.
        They all start from [0,0] and face [0, -1] direction in 2D ground
        :param joint_name:
        :param n:
        :return:
        """
        self.feature_point = joint_name
        for i in xrange(n):
            low_dimension_vector = self.motion_primitive_model.sample_low_dimensional_vector()
            motion_spline = self.motion_primitive_model.back_project(low_dimension_vector)
            self.low_dimension_vectors.append(low_dimension_vector.tolist())
            quat_frames = motion_spline.get_motion_vector()
            start_root_point = np.array(quat_frames[0][:3])
            target_point = get_cartesian_coordinates_from_quaternion(self.skeleton,
                                                                     self.feature_point,
                                                                     quat_frames[-1])
            self.feature_points.append(target_point - start_root_point)
            ori_vector = pose_orientation_quat(quat_frames[-1]).tolist()
            self.orientations.append(ori_vector)

    def save_data(self, save_filename):
        output_data = {}
        output_data['motion_vectors'] = self.low_dimension_vectors
        output_data['feature_points'] = self.feature_points
        output_data['orientations'] = self.orientations
        with open(save_filename, 'wb') as outfile:
            json.dump(output_data, outfile)

    def plot_orientation(self):
        fig = plt.figure()
        for i in xrange(len(self.orientations)):
            plt.plot([0, self.orientations[i][0]], [0, self.orientations[i][1]], 'r')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()

    def load_data_from_json(self, data_file):
        with open(data_file, 'rb') as infile:
            training_data = json.load(infile)
        self.low_dimension_vectors = training_data['motion_vectors']
        self.feature_points = training_data["feature_points"]
        # self.orientations = training_data["orientations"]

    def model_feature_points(self):
        # training_samples = np.concatenate((self.feature_points, self.orientations), axis=1)
        training_samples = np.asarray(self.feature_points)
        gmm_trainer = GMMTrainer(training_samples)
        self.feature_point_dist = gmm_trainer.gmm

    def save_feature_distribution(self, save_filename):
        data = {'name': self.motion_primitive_model.name,
                'feature_point': self.feature_point,
                'gmm_weights': self.feature_point_dist.weights_.tolist(),
                'gmm_means': self.feature_point_dist.means_.tolist(),
                'gmm_covars': self.feature_point_dist.covars_.tolist()}
        with open(save_filename, 'wb') as outfile:
            json.dump(data, outfile)

    def sample_new_feature(self):
        return np.ravel(self.feature_point_dist.sample())

    def load_feature_dist(self, dist_file):
        """
        Load feature point distribution from json file
        :param dist_file:
        :return:
        """
        with open(dist_file, 'rb') as infile:
            data = json.load(infile)
        n_components = len(data['gmm_weights'])
        self.feature_point_dist = mixture.GMM(n_components, covariance_type='full')
        self.feature_point_dist.weights_ = np.array(data['gmm_weights'])
        self.feature_point_dist.means_ = np.array(data['gmm_means'])
        self.feature_point_dist.converged_ = True
        self.feature_point_dist.covars_ = np.array(data['gmm_covars'])

    def evaluate_target_point(self, target_point):
        return self.feature_point_dist.score([target_point,])[0]
