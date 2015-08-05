# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:56:13 2015

@author: erhe01
"""

import os
import sys

sys.path.append("..")
import numpy as np
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
from libtest import params, pytest_generate_tests
from rpy2_funcs import rpy2_temporal_mean, rpy2_temporal
from motion_generator.lib import motion_primitive

TESTPATH = os.sep.join([".."] * 2 + ["test_data"])
MP_FILE = os.sep.join([TESTPATH, "walk_leftStance_quaternion_mm.json"])
# print MP_FILE


def params(funcarglist):
    """Test function parameter decorator

    Provides arguments based on the dict funcarglist.

    """
    def wrapper(function):
        function.funcarglist = funcarglist
        return function
    return wrapper


def pytest_generate_tests(metafunc):
    """Enables params to work in py.test environment"""
    for funcargs in getattr(metafunc.function, 'funcarglist', ()):
        metafunc.addcall(funcargs=funcargs)


def back_project_rpy2(m, sample):
    """ Back project low dimensional parameters to frames using eval.fd
    implemented in R  """
    alpha = sample[:m.s_pca["n_components"]]
    gamma = sample[m.s_pca["n_components"]:]
    canonical_motion = m._inverse_spatial_pca(alpha)
    time_function = m._inverse_temporal_pca(gamma)

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

    # save time function
    time_function = time_function.tolist()
    eval_fd = robjects.r['eval.fd']
    frames = np.array(eval_fd(time_function, canonical_motion))
    frames = np.reshape(frames, (frames.shape[0],
                                 frames.shape[-1]))
    return frames


class TestMotionPrimitive(object):

    """ Unit test class for MotionPrimitive class """

    def setup_class(self):
        """Class setup method"""
        self.m = motion_primitive.MotionPrimitive(MP_FILE)
        self.s = self.m.sample(return_lowdimvector=True)

    def test_temporal_mean_vector(self):
        """ Test if the motion primitive and the rpy2 function give the same
        mean vector for temporal component """
        mean_rpy2 = rpy2_temporal_mean(self.m)
        mean_scipy = self.m._mean_temporal()
        assert np.allclose(np.ravel(mean_rpy2), np.ravel(mean_scipy))

    def test_temporal_vector(self):
        """ Test if the motion primitive and the rpy2 function give the same
        vector for temporal component """
        gamma = self.s[self.m.s_pca["n_components"]:]
        t_rpy2 = rpy2_temporal(self.m, gamma)
        t_scipy = self.m._inverse_temporal_pca(gamma)
        assert np.allclose(np.ravel(t_rpy2), np.ravel(t_scipy))


def test_compare_get_motion_vector_with_r():
    """ Compares the result of get_motion_vector with a reference implementation using R.
    """
    m = motion_primitive.MotionPrimitive(MP_FILE)
    N = 100
    for i in xrange(N):

        sample = m.sample(return_lowdimvector=True)
        frames = m.back_project(sample).get_motion_vector()
        frames_rpy2 = back_project_rpy2(m, sample)
        #assert np.isclose(frames_rpy2, frames)
        assert np.all(frames_rpy2 == frames)
    return

if __name__ == "__main__":
    test_compare_get_motion_vector_with_r()
