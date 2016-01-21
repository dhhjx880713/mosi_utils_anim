# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:53:40 2015

@author: mamauer
"""
import numpy as np
from gp_multi import GPMulti
import os
from motion_primitive import MotionPrimitive
from sklearn import mixture
import json
import zipfile
import cPickle as pickle


class GPMixture(object):

    """ Mixture of GPs

    Parameters
    ---------
    * X : list of numpy.ndarray
    \tEach list entry is a numpy.ndarray with the X Values of the ith cluster
    * Y : list of numpy.ndarray
    \tEach list entry is a numpy.ndarray with the corresponding Y Values
    * classifier_gmm : sklearn.mixtures.gmm
    \tThe GMM of the input data
    * output_gmm : (optional) sklearn.mixtures.gmm
    \tThe GMM of the output data, default to None # can be removed
    * gp_optimizer: (optional) string
    \tA string with the the optimizer name
    \tdefault is: ('scg')
    * max_iter : (optional) int
    \t Sets the number of iterations of the gp optimizer
    * messages : (optional) bool
    \t Activates debug messages for the gp optimizer

    Attributes
    ----------
    * X : list of numpy.ndarray
    \tEach list entry is a numpy.ndarray with the X Values of the ith cluster
    * Y : list of numpy.ndarray
    \tEach list entry is a numpy.ndarray with the corresponding Y Values


    """

    def __init__(self, X, Y, classifier_gmm, output_gmm=None, gp_optimizer='scg',
                 max_iters=200, messages=False):
        self.X = X
        self.Y = Y
        succes = False
        i = 0
        while not succes:
            try:
                self.input_dim = X[i].shape[1]
                self.output_dim = Y[i].shape[1]
                succes = True
            except IndexError:
                i += 1
            except TypeError:       # Not correctly initialized...
                self.input_dim = -1
                self.output_dim = -1
                succes = True
        self.classifier_gmm = classifier_gmm
        self.output_gmm = output_gmm
        self.gp_optimizer = gp_optimizer
        self.gp_list = []
        self.cluster_to_gp_weights_ = []
        self.max_iters = max_iters
        self.opt_kwargs = {"max_iters": max_iters, "messages": messages}

    def train_gp_mixture(self):
        """
        trains a GPMixture object with given data
        """
        inputcluster = len(self.classifier_gmm.weights_)
        outputcluster = len(self.X)
        # a list for each input cluster containing the votes for each output cluster
        input_to_output_cluster_votes_ = [[] for i in xrange(inputcluster)]
        for output_cluster_index in xrange(outputcluster):
            X_c = self.X[output_cluster_index]
            Y_c = self.Y[output_cluster_index]
            #predict the clusters of each input sample and count votes to estimate weights for the clusters from the gp
            xlabels = self.classifier_gmm.predict(X_c).tolist()
            for i in xrange(inputcluster):
                votes = xlabels.count(i)
                input_to_output_cluster_votes_[i].append(votes)

            if X_c.shape[0] == 0:
                continue
            success = False
            while not success:
                try:
                    gp = GPMulti(X_c, Y_c)
                    gp.optimize(self.gp_optimizer, **self.opt_kwargs)
                    success = True
                except np.linalg.linalg.LinAlgError:
                    self.opt_kwargs['max_iters'] /= 2
            self.opt_kwargs['max_iters'] = self.max_iters
            self.gp_list.append(gp)

        for input_cluster_index in xrange(inputcluster):
            if sum(input_to_output_cluster_votes_[input_cluster_index]) != 0:
                self.cluster_to_gp_weights_.append(
                    [float(votes) / sum(input_to_output_cluster_votes_[input_cluster_index]) for votes in input_to_output_cluster_votes_[input_cluster_index]])
            else:
                input_to_output_cluster_votes_[input_cluster_index] = [input_cluster_index + 1 for input_cluster_index in input_to_output_cluster_votes_[input_cluster_index]]

                self.cluster_to_gp_weights_.append(
                    [float(votes) / sum(input_to_output_cluster_votes_[input_cluster_index]) for votes in input_to_output_cluster_votes_[input_cluster_index]])

    def predict(self, Xnew):
        """ Predict a GMM distribution for the output values by
            using a list of predicted gaussian distributions as components with offline trained weights

        Parameters
        ----------
        * Xnew: numpy.ndarray
            The s vector

        Returns
        -------
        * gmm: sklearn.mixture.gmm
            The corresponding GMM distribution
        """
        means_ = []
        covars_ = []
        weights_ = []

        # self.gmm.predict(Xnew[None, :])[0] suspected to cause problems
        input_cluster_index = self.classifier_gmm.predict(Xnew)[0]
        for gp_index, gp in enumerate(self.gp_list):
            if self.cluster_to_gp_weights_[input_cluster_index][gp_index] != 0:
                gaussian_distribution = gp.predict(Xnew, full_cov=True)
                means_.append(np.ravel(gaussian_distribution[0]))
                covars_.append(gaussian_distribution[1])
                weights_.append(self.cluster_to_gp_weights_[input_cluster_index][gp_index])

        gmm = mixture.GMM(len(weights_), covariance_type='full')
        gmm.weights_ = np.array(weights_)
        gmm.means_ = np.array(means_)
        gmm.covars_ = np.array(covars_)
        gmm.converged_ = True
        print "New GMM has %d clusters, the original has %d" % \
            (len(weights_), len(self.classifier_gmm.weights_))

        return gmm

    def save(self, filepath):
        """Saves a GPMixture object to zip file

        Parameter
        ---------

        * filepath : str
        \tThe path where to save the result"""

        fpath = filepath.replace('/', os.sep)
        folder = fpath.split(os.sep)[:-1]
        folder = os.sep.join(folder)
        zfile = zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED)
        for i, gp in enumerate(self.gp_list):
            gpname = 'gp%d.pickle' % i
            gppath = folder + os.sep + gpname
            gp.pickle(gppath)
            zfile.write(gppath, gpname)
            os.remove(gppath)

        weightfile = folder + os.sep + "weights"
        with open(weightfile, 'w') as outfile:
            json.dump(self.cluster_to_gp_weights_, outfile, sort_keys=True,
                      indent=4, separators=(',', ': '))

        zfile.write(weightfile, "weights.json")
        os.remove(weightfile)
        zfile.close()

    @classmethod
    def load(cls, filepath, input_gmm, output_gmm=None):
        """Updates self.gp_list from a zip file"""
        gpm = cls(X=None, Y=None, gmm=input_gmm, output_gmm=output_gmm)

        gpm.gp_list = []
        gpm.weights_ = []

        zfile = zipfile.ZipFile(filepath, "r", zipfile.ZIP_DEFLATED)

        gpm.weights_ = np.array(json.loads(zfile.read('weights.json')))

        for f in zfile.namelist():
            if f == 'weights.json':
                continue
            data = zfile.read(f)
            gp = pickle.loads(data)
            gpm.gp_list.append(gp)
        return gpm


def build_X_Y_pairs(action1, prim1, action2, prim2,
                    input_path):
    """Categorize data based on output clusters

    Parameters
    ----------
    * action1 : str
    \tInput elementary action

    * prim1 : str
    \tInput motion primitive

    * action2 : str
    \tOutput elementary action

    * prim2 : str
    \tOutput motion primitive

    * input_path_prefix : str
    \tThe input path for the transition file

    Output
    ------
    * X : list
    \tA list of input data, each element in the list is all the samples in
    corresponding cluster

    * Y : list
    \tA list of output data, each element in the list in all the samples in
    \tcorresponding cluster
    """
    transition_data_filename = os.sep.join((input_path,
                                            '%s_%s_to_%s_%s.json' % (action1,
                                                                     prim1,
                                                                     action2,
                                                                     prim2)))
    with open(transition_data_filename) as transition_data_file:
        data = json.load(transition_data_file)

    num_c = max(data['output_class_identity']) + 1

    X = [[] for i in xrange(num_c)]
    Y = [[] for i in xrange(num_c)]
    for i, c in enumerate(data['output_class_identity']):
        X[c].append(data['input_motion_data'][i])
        Y[c].append(data['output_motion_data'][i])

    for i in xrange(len(X)):
        X[i] = np.array(X[i])
        Y[i] = np.array(Y[i])
    return X, Y


def create_and_save(action1, prim1, action2, prim2, max_iters,
                    mm_path, transition_path, output_path):
    """ creates a GP for the transition action1_prim1_to_action2_prim2 and
    saves it to a .GPM zip file

    Parameters
    ----------
    * action1 : str
    \tInput elementary action

    * prim1 : str
    \tInput motion primitive

    * action2 : str
    \tOutput elementary action

    * prim2 : str
    \tOutput motion primitive

    * max_iters : int
    \tThe maximum number of iterations for the training

    * mm_path : str
    \tThe path for the mm file from action1_prim1

    * transition_path : str
    \tThe path for the transition file action1_prim1_to_action2_prim2.json

    *output_path : str
    \tThe path of the output folder where the file is to be saved
    """

    mmfile = os.sep.join((mm_path, '%s_%s_quaternion_mm.json' % (action1,
                                                                 prim1)))

    mm = MotionPrimitive(mmfile)

    X, Y = build_X_Y_pairs(action1, prim1, action2, prim2, transition_path)

    gpmixture = GPMixture(X, Y, mm.gaussian_mixture_model, max_iters=max_iters, messages=False)
    gpmixture.train_gp_mixture()
    gpmixture.save(os.sep.join((output_path, "%s_%s_to_%s_%s.GPM" %
                                (action1, prim1, action2, prim2))))

