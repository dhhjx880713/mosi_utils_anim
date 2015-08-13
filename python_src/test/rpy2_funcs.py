# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 10:24:36 2015

@author: mamauer

This module defines some functions to calculate:
    * temporal mean vector (given a motion primitive)
    * spatial vector (given a lowdimensional vector)
    * temporal vector (given a lowdimensional vector)

All calculations are done using the rpy2 (High-)Interface and the
R-Library "fda"
"""
import os
import sys
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as numpy2ri
import numpy as np
import scipy.interpolate as si

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(1, TESTPATH)
sys.path.insert(1, TESTPATH + (os.sep + os.pardir))


def rpy2_temporal_mean(m):
    """ Calculate the mean of the temporal part of the motion using the
    rpy2 interface with the R - Library "fda"

    Parameters
    ----------
    m : MotionPrimitive
        The motion primitive

    Returns
    -------
    temporal_mean : numpy.ndarray
        The temporal mean vector of the motion primitive
    """
    robjects.r('library("fda")')
    fd = robjects.r['fd']
    fdeval = robjects.r['eval.fd']
    basis = m.t_pca["basis_function"]

    mean_coefs = numpy2ri.numpy2ri(m.t_pca["mean_vector"])
    meanfd = fd(mean_coefs, basis)
    return np.array(fdeval(m.canonical_time_range.tolist(), meanfd))


def rpy2_temporal(m, gamma):
    """ Calculate the temporal part of the motion using the
    rpy2 interface with the R - Library "fda"

    Parameters
    ----------
    m : MotionPrimitive
        The motion primitive

    gamma : numpy.ndarray
        The lowdimensional vector (only temporal component)

    Returns
    -------
    temporal : numpy.ndarray
        The temporal mean vector of the motion primitive
    """
    robjects.r('library("fda")')
    fd = robjects.r['fd']
    basis = m.t_pca["basis_function"]
    eigen_coefs = numpy2ri.numpy2ri(m.t_pca["eigen_vectors"])
    eigenfd = fd(eigen_coefs, basis)
    mean_coefs = numpy2ri.numpy2ri(m.t_pca["mean_vector"])
    meanfd = fd(mean_coefs, basis)
    numframes = m.n_canonical_frames

    fdeval = robjects.r['eval.fd']

    time_frame = np.arange(0, numframes).tolist()
    mean = np.array(fdeval(time_frame, meanfd))
    eigen = np.array(fdeval(time_frame, eigenfd))
    t = [0, ]
    for i in xrange(numframes):
        t.append(t[-1] + np.exp(mean[i] + np.dot(eigen[i], gamma)))

    t = np.array(t[1:])
    t -= 1
    zeroindices = t < 0
    t[zeroindices] = 0
    x_sample = np.arange(m.n_canonical_frames)
    try:
        inverse_spline = si.splrep(t, x_sample, w=None, k=2)
    except ValueError as e:  # Exception
        print "exception"
        print e.message
        print t, "#####"

    frames = np.linspace(1, t[-2], np.round(t[-2]))
    t = si.splev(frames, inverse_spline)
    t = np.insert(t, 0, 0)
    t = np.insert(t, len(t), m.n_canonical_frames - 1)
    return np.asarray(t)


def rpy2_spatial(m, alpha):
    """ Back project low dimensional parameters to frames using eval.fd
    implemented in R  """
    canonical_motion = m._inverse_spatial_pca(alpha)
    robjects.r('library("fda")')

    # define basis object
    n_basis = canonical_motion.shape[0]
    rcode = """
        n_basis = %d
        n_frames = %d
        basisobj = create.bspline.basis(c(0, n_frames - 1),
                                        nbasis = n_basis)
    """ % (n_basis, m.n_canonical_frames)
    robjects.r(rcode)
    basis = robjects.globalenv['basisobj']

    # create fd object
    fd = robjects.r['fd']
    coefs = numpy2ri.numpy2ri(canonical_motion)
    canonical_motion = fd(coefs, basis)

    return np.array(canonical_motion)


# def main():
#    mm_file = 'walk_leftStance_quaternion_mm.json'
#    m = MotionPrimitive(mm_file)
#
#    s = m.sample(return_lowdimvector=True)
#    alpha = s[:m.s_pca["n_components"]]
#    gamma = s[m.s_pca["n_components"]:]
#
#    t_rpy2 = rpy2_temporal(m, gamma)
#    t_scipy = m._inverse_temporal_pca(gamma)
#    assert np.allclose(np.ravel(t_rpy2), np.ravel(t_scipy))
#
#
# if __name__ == '__main__':
#    main()
