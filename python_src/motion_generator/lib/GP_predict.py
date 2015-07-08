# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 11:37:47 2015

@author: FARUPP
"""
import numpy as np
import GPy
import os
import json
import zipfile
from sklearn import mixture
from GPMixture import GPMixture


def load_gpm_from_path(filepath):
    """loads a saved GPMixture file and returns a GPMixture object"""
    gpm = GPMixture(X=None, Y=None, gmm=None, output_gmm=None)
    gpm.gps = []
    gpm.weights_ = []

    zfile = zipfile.ZipFile(filepath, "r", zipfile.ZIP_DEFLATED)
    gpm.weights_ = np.array(json.loads(zfile.read('weights.json')))

    for f in zfile.namelist():
        if f == 'weights.json':
            continue
        zfile.extract(f)
        gp = GPy.load(f)
        gpm.gps.append(gp)
        os.remove(f)
    return gpm


def predict(gpm, Xnew):
    """ Predicts a GPM at Xnew

    Parameters
    ----------
    * Xnew : numpy.ndarray
    \tThe s vector to predict at
    * gpm : GPMixture
    \tThe GPM to be evaluated

    Returns
    -------
    * gmm: sklearn.mixture.gmm
        The corresponding GMM distribution
        (not multiplied with the output\
        gmm!)
    """
    means_ = []
    covars_ = []
    weights_ = []
    cluster_index = gpm.gmm.predict(Xnew)[0]  # suspected to cause problems
    print cluster_index
    for c, gp in enumerate(gpm.gps):
        if gpm.weights_[cluster_index][c] != 0:
            p = gp.predict(Xnew, full_cov=True)
            means_.append(np.ravel(p[0]))
            covars_.append(p[1])
            weights_.append(gpm.weights_[cluster_index][c])

    gmm = mixture.GMM(len(weights_), covariance_type='full')
    gmm.weights_ = np.array(weights_)
    gmm.means_ = np.array(means_)
    gmm.covars_ = np.array(covars_)
    gmm.converged_ = True
    return gmm


