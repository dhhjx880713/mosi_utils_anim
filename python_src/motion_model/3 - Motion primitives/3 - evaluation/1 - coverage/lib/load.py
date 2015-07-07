# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 09:41:42 2015

@author: mamauer
"""
import numpy as np
from sklearn import mixture
import json
import os


def load_gmm(action, prim):
    """Yields the gmm from a motion primitive

    Parameters
    ----------
    * action : str
    \tElementary action
    * prim : str
    \tMotion primitive

    Returns
    -------
    * gmm : sklearn.mixture.gmm.GMM
    \tThe GMM
    """
    filepath_prefix = os.sep.join(('..', '..', 'data', '3 - Motion primitives',
                                   'motion_primitives_quaternion_PCA95',
                                   'elementary_action_%s' % action))
    filepath = os.sep.join((filepath_prefix,
                            '%s_%s_quaternion_mm.json' % (action, prim)))

    return load_gmm_from_path(filepath)


def load_gmm_from_path(f):
    """ Load and initialize a GMM based on parameters in a file

    Parameters
    ----------
    * f : str
    \tThe filepath of the gmm

    Returns
    -------
    * gmm : sklearn.mixture.gmm.GMM
    \tThe GMM
    """
    with open(f) as fp:
        data = json.load(fp)

    n_components = len(np.array(data['gmm_weights']))
    gmm = mixture.GMM(n_components, covariance_type='full')
    gmm.weights_ = np.array(data['gmm_weights'])
    gmm.means_ = np.array(data['gmm_means'])
    gmm.converged_ = True
    gmm.covars_ = np.array(data['gmm_covars'])
    return gmm
