# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 13:40:19 2015

@author: mamauer

This modul provides different functions for multipling two gaussian mixture
models.

LITERATUR
---------
[Schrempf05]
\tOliver C. Schrempf, Olga Feiermann, Uwe D. Hanebeck \
Optimal Mixture Approximation of the Product of Mixtures

"""
import numpy as np
from gaussian import Gaussian
import sklearn.mixture as mixture


def _cluster_mul(gmm1, gmm2):
    """ Multiply two Gaussian Mixtures by multipling each component of gmm1 \
    with the same component of gmm2.
    NOTE: Both component need the same number of clusters.

    Parameters
    ----------
    * gmm1: sklearn.mixture.gmm.GMM
    \tThe first gaussian mixture
    * gmm2: sklearn.mixture.gmm.GMM
    \tThe second gaussian mixture

    Returns
    -------
    * result: sklearn.mixture.gmm.GMM
    \tThe resulting gaussian mixture
    """
    means_ = []
    covars_ = []

    for i in xrange(gmm1.n_components):
        g1 = Gaussian(gmm1.mean_[i], gmm1.covar_[i])
        g2 = Gaussian(gmm2.mean_[i], gmm2.covar_[i])
        g3 = g1 * g2
        means_.append(g3.mu)
        covars_.append(g3.sigma)
    gmm = mixture.GMM(len(means_), covariance_type='full')
    gmm.means_ = np.array(means_)
    gmm.covars_ = np.array(covars_)
    gmm.weights = gmm1.weigths_
    return gmm


def _full_mul(gmm1, gmm2):
    """ Multiply two Gaussian Mixtures by multipling each component of gmm1 \
    with each component of gmm2.
    NOTE: The number of components grows exponentially.

    Parameters
    ----------
    * gmm1: sklearn.mixture.gmm.GMM
    \tThe first gaussian mixture
    * gmm2: sklearn.mixture.gmm.GMM
    \tThe second gaussian mixture

    Returns
    -------
    * result: sklearn.mixture.gmm.GMM
    \tThe resulting gaussian mixture
    """
    weights_ = []
    means_ = []
    covars_ = []

    for i in xrange(gmm1.n_components):
        for j in xrange(gmm2.n_components):
            g1 = Gaussian(gmm1.means_[i], gmm1.covars_[i])
            g2 = Gaussian(gmm2.means_[j], gmm2.covars_[j])
            g3 = g1 * g2
            means_.append(g3.mu)
            covars_.append(g3.sigma)

            sumsigma = np.array(g1.sigma + g2.sigma)
            deltamu = g1.mu - g2.mu

            det = np.linalg.det(2*np.pi*(sumsigma))

            weight1 = gmm1.weights_[i] * gmm2.weights_[j] / (np.sqrt(det))

            musigma = np.dot(deltamu, np.linalg.inv(sumsigma))

            musigmamu = np.dot(musigma, deltamu)
            weight2 = np.exp(-0.5 * musigmamu)
            weight = weight1 * weight2
            weights_.append(weight)

            #raw_input()
    gmm = mixture.GMM(len(means_), covariance_type='full')
    gmm.means_ = np.array(means_)
    gmm.covars_ = np.array(covars_)
    gmm.weights = np.array(weights_)
    return gmm


def mul(gmm1, gmm2, type='full'):
    """ Multiply two Gaussian Mixtures and return a new GMM

    Parameters
    ----------
    * gmm1: sklearn.mixture.gmm.GMM
    \tThe first gaussian mixture
    * gmm2: sklearn.mixture.gmm.GMM
    \tThe second gaussian mixture
    * type: string
    \tThe method how to multiply the gaussians.
    \tValid options are: full,
    \tDescription of the options:
    \tfull: Perform a full componentwise multiplication

    Returns
    -------
    * result: sklearn.mixture.gmm.GMM
    \tThe resulting gaussian mixture
    """
    if type == 'full':
        return _full_mul(gmm1, gmm2)
    if type == 'cluster':
        return _cluster_mul(gmm1, gmm2)


def main():
    """ Function to demonstrate the usage of this module """
    from lib.load import load_gmm
    leftstance = 'walk_leftStance_quaternion_mm.json'
    rightStance = 'walk_rightStance_quaternion_mm.json'

    gmm_l = load_gmm(leftstance)
    gmm_r = load_gmm(rightStance)

    gmm_l2 = mul(gmm_l, gmm_l)

    return

if __name__ == '__main__':
    main()