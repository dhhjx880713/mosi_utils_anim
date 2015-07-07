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
from lib.load import load_gmm
from lib import gmm_math
#os.sys.path.append(os.sep.join(('..', '4 - Transition model')))
from lib.GPMixture import GPMixture


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


def load_gpm(action1, prim1, action2, prim2):
    """ loads the gpm file from data folder and also the input and output gmms
    """
    filepath_prefix = os.sep.join(('..', '..', 'data', '4 - Transition model',
                                  'output'))
    filename = "%s_%s_to_%s_%s.GPM" % (action1, prim1, action2, prim2)

    gpm = load_gpm_from_path(os.sep.join((filepath_prefix, filename)))
    gpm.gmm = load_gmm(action1, prim1)
    gpm.output_gmm = load_gmm(action2, prim2)
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


def load_all_gpm():
    """yields a dict with transition and the corresponding GPM"""
    gpms = {}
    primitives = [('walk', 'leftStance'), ('walk', 'rightStance'),
                  ('walk', 'beginLeftStance'), ('walk', 'beginRightStance'),
                  ('walk', 'endLeftStance'), ('walk', 'endRightStance'),
                  ('carry', 'leftStance'), ('carry', 'rightStance'),
                  ('carry', 'beginLeftStance'), ('carry', 'beginRightStance'),
                  ('carry', 'endLeftStance'), ('carry', 'endRightStance'),
                  ('pick', 'first'), ('pick', 'second'), ('place', 'first'),
                  ('place', 'second')]
    for action1, prim1 in primitives:
        for action2, prim2 in primitives:
            try:
                gpms["%s_%s_to_%s_%s" % (action1, prim1, action2, prim2)] \
                    = load_gpm(action1, prim1, action2, prim2)
            except IOError:
                continue
    return gpms


def main():
    raise NotImplementedError

if __name__ == '__main__':
    main()
