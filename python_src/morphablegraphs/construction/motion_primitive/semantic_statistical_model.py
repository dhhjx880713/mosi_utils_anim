# encoding: UTF-8
from statistical_model_trainer import StatisticalModelTrainer
from gmm_trainer import GMMTrainer
from sklearn import mixture
import numpy as np

class SemanticStatisticalModel(StatisticalModelTrainer):

    def __init__(self, fdata, semantic_label):
        """

        :param fdata:
        :param semantic_label: dictionary contains semantic label matching the filename, and numeric label for data
        :return:
        """
        super(SemanticStatisticalModel, self).__init__(fdata)
        self.semantic_label = semantic_label
        self.classified_data = {}
        self.semantic_classification()

    def semantic_classification(self):

        for key in self.semantic_label.keys():
            indices = [idx for idx, item in enumerate(self._file_order) if key in item]
            self.classified_data[key] = self._motion_parameters[indices]
            if len(self.classified_data[key]) == 0:
                raise KeyError(key + ' is not found in data')

    def create_gaussian_mixture_model(self):
        gmm_models = []
        semantic_labels = []
        class_weights = []
        n_gaussians = 0
        for key in self.classified_data.keys():
            gmm_trainer = GMMTrainer(self.classified_data[key])
            gmm_models.append(gmm_trainer.gmm)
            semantic_labels.append(self.semantic_label[key])
            class_weights.append(float(len(self.classified_data[key]))/float(len(self._motion_parameters)))
            n_gaussians += gmm_trainer.numberOfGaussian
        self.gmm = mixture.GMM(n_components=n_gaussians, covariance_type='full')
        new_weights = []
        new_means = []
        new_covars = []
        for i in range(len(gmm_models)):
            new_weights.append(class_weights[i] * gmm_models[i].weights_)
            new_means.append(np.concatenate((gmm_models[i].means_,
                                             np.zeros((len(gmm_models[i].means_), 1)) + semantic_labels[i]),
                                            axis=1))
            covars_shape = gmm_models[i].covars_.shape
            new_covar_mat = np.zeros((covars_shape[0], covars_shape[1] + 1, covars_shape[2] + 1))
            for j in range(covars_shape[0]):
                new_covar_mat[j][:-1, :-1] = gmm_models[i].covars_[j]
                new_covar_mat[j][-1, -1] = 1e-5
            new_covars.append(new_covar_mat)
        self.gmm.weights_ = np.concatenate(new_weights)
        self.gmm.means_ = np.concatenate(new_means)
        self.gmm.covars_ = np.concatenate(new_covars)

    def gen_semantic_motion_primitive_model(self, savepath=None):
        self.create_gaussian_mixture_model()
        self._save_model(savepath)
