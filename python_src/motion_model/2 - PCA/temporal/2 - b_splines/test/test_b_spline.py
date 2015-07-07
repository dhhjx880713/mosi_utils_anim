# -*- coding: utf-8 -*-
"""
Created on Mon Dec 08 10:29:12 2014

@author: mamauer
"""

import os
import sys
import pytest
from libtest import params, pytest_generate_tests

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(0, TESTPATH)
sys.path.insert(0, TESTPATH + (os.sep + os.pardir))

import numpy as np
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
from b_splines import create_bsplines

class Test_create_bspline(object):
    '''Unittest class for create_bspline''' 
    paramsets = {
        'test_num_knots': [
            dict(data=[[1, 2, 3, 4, 5, 6, 7, 8, 9], 
                       [1, 2, 3, 4, 5, 6, 7, 8, 9], 
                       [1, 2, 3, 4, 5, 6, 7, 8, 9]],
                 numknots=5),
        ],
        
        'test_num_knots': [
            dict(data=[[1, 2, 3, 4, 5, 6, 7, 8, 9], 
                       [1, 2, 3, 4, 5, 6, 7, 8, 9], 
                       [1, 2, 3, 4, 5, 6, 7, 8, 9]],
                 numknots=5),
        ],
    }        
    
                        