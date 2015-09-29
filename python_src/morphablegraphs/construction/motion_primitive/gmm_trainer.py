__author__ = 'hadu01'

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture


class GMMTrainer(object):
    def __init__(self, data):
        assert len(data.shape) == 2, ('the data should be a 2d matrix')
        self.data = data
        self.train_gmm()
        self.create_gmm()

    def train_gmm(self, n_K=20, DEBUG=0):
        '''
        Find the best number of Gaussian using BIC score
        '''

        obs = np.random.permutation(self.data)
        lowestBIC = np.infty
        BIC = []
        K = range(1, n_K)
        BICscores = []
        for i in K:
            gmm = mixture.GMM(n_components=i, covariance_type='full')
            gmm.fit(obs)
            BIC.append(gmm.bic(obs))
            BICscores.append(BIC[-1])
            if BIC[-1] < lowestBIC:
                lowestBIC = BIC[-1]
        index = min(xrange(n_K - 1), key=BIC.__getitem__)
        print 'number of Gaussian: ' + str(index + 1)
        self.numberOfGaussian = index + 1
        if DEBUG:
            fig = plt.figure()
            plt.plot(BICscores)
            plt.show()

    def create_gmm(self):
        '''
        Using GMM to model the data with optimized number of Gaussian
        '''
        self.gmm = mixture.GMM(n_components=self.numberOfGaussian,
                               covariance_type='full')
        self.gmm.fit(self.data)
        scores = self.gmm.score(self.data)
        averageScore = np.mean(scores)
        print 'average score is:' + str(averageScore)

    def save_model(self, filename):
        model_data = {'gmm_weights': self.gmm.weights_.tolist(),
                      'gmm_means': self.gmm.means_.tolist(),
                      'gmm_covars': self.gmm.covars_.tolist()}
        with open(filename, 'wb') as outfile:
            json.dump(model_data, outfile)