# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 08:53:00 2015

@author: mamauer
"""
import os
import sys
import pytest
from libtest import params, pytest_generate_tests

import rpy2.robjects.vectors
import rpy2.robjects

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(0, TESTPATH)
sys.path.insert(0, TESTPATH + (os.sep + os.pardir))

import numpy as np
from normal_pca_on_coefs import load_r_data, get_coefs_from_fd
from normal_pca_on_coefs import normal_pca_on_coefs

class Test_load_r_data(object):
    '''Unittest class for load_r_data'''
        
    def test_load_timefunctions(self):
        testfile = os.sep.join((TESTPATH, 'testdata', 
                                'test_load_r_data.RData'))
        data = load_r_data(testfile)
        assert isinstance(data, rpy2.robjects.vectors.ListVector)
        

class Test_get_coefs_from_fd(object):
    def test_not_rpy_object(self):        
        with pytest.raises(ValueError) as excinfo:
            get_coefs_from_fd(1909)
        
        assert excinfo.value.message == "fd must be a ListVector"
        
    def test_not_fd_object(self):
        testfile = os.sep.join((TESTPATH, 'testdata', 
                                'not_fd_object.RData'))
        testdata = load_r_data(testfile)
        
        with pytest.raises(ValueError) as excinfo:
            get_coefs_from_fd(testdata)
        
        assert excinfo.value.message == "fd doesn't have coefs. " \
                                        "Probably not a RPy fd-object"
    
    def test_get_numpy_arrays(self):
        testfile = os.sep.join((TESTPATH, 'testdata', 
                                'test_load_r_data.RData'))
        testdata = load_r_data(testfile)
        coefs, shape = get_coefs_from_fd(testdata, as_rpy=False)
        
        assert isinstance(coefs, np.ndarray)
        assert isinstance(shape, np.ndarray)
        
    def test_get_rpy_vectors(self):
        testfile = os.sep.join((TESTPATH, 'testdata', 
                                'test_load_r_data.RData'))
        testdata = load_r_data(testfile)
        coefs, shape = get_coefs_from_fd(testdata, as_rpy=True)
        
        assert isinstance(coefs, rpy2.robjects.vectors.Array)
        assert isinstance(shape, rpy2.robjects.vectors.IntVector)
        
    def test_is_result_correct(self):
        testfile = os.sep.join((TESTPATH, 'testdata', 
                                'test_load_r_data.RData'))
        testdata = load_r_data(testfile)
        coefs, shape = get_coefs_from_fd(testdata, as_rpy=False)
        
        assert np.equal(coefs.shape, shape).all()
        assert np.equal(coefs.shape, (5, 620, 57)).all()
        assert np.equal(shape, (5, 620, 57)).all()
        
        
class Test_normal_pca_on_coefs(object):
    '''Unittest class for normal_pca_on_coefs'''
    pass
                               