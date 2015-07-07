# -*- coding: utf-8 -*-
import os
import sys
import pytest
from libtest import params, pytest_generate_tests

TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]) + os.sep
sys.path.insert(0, TESTPATH)
sys.path.insert(0, TESTPATH + (os.sep + os.pardir))

import numpy as np
from z_t import load_timefunctions, is_strict_incresing
from z_t import transform_timefunction, get_monotonic_indices


class Test_load_timefunctions(object):
    '''Unittest class for load_timefunctions'''

    expected = {
        'walk_005_2_leftStance_340_387.bvh':
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
             18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46],
        'walk_024_2_leftStance_41_87.bvh':     
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 
             18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
             34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45],
        'walk_030_3_leftStance_157_211.bvh':   
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
             19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             35, 36, 37, 38, 39, 41, 43, 45, 47, 49, 51, 53], 
        "walk_014_2_leftStance_354_402.bvh": 
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
             19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47], 
        "walk_003_1_leftStance_44_86.bvh": 
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
             19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
             35, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41]            
    }
        
    def test_load_timefunctions(self):
        testfile = os.sep.join((TESTPATH, 'testdata', 
                                'test_load_timefunctions.json'))
        data = load_timefunctions(testfile)
        
        for (key, value) in data.iteritems():
            assert self.expected[key] == value
            
                        
class Test_is_strict_incresing(object):
    '''Unittest class for is_strict_incresing'''
    paramsets = {
        'test_all_equal': [
            dict(indices=[1,1,1,1]),
            dict(indices=[4,4,4,4])         
        ],
        
        'test_all_incresing': [
            dict(indices=[1,2,3,4,5]), 
            dict(indices=[5,6,8,10,100]),  
            dict(indices=range(1000))      
        ],
        
        'test_unordered': [
            dict(indices=[5,4,8,1,3]), 
            dict(indices=[500,42,23,1213,234]),        
        ]
    }        
    
    @params(paramsets['test_all_equal'])
    def test_all_equal(self, indices):
        '''Test with all indices equal'''
        assert is_strict_incresing(indices) == False
        
    @params(paramsets['test_all_incresing'])
    def test_all_incresing(self, indices):
        '''Test with all indices incresing'''
        assert is_strict_incresing(indices) == True
        
    @params(paramsets['test_unordered'])
    def test_unordered(self, indices):
        '''Test with unordered indices'''
        assert is_strict_incresing(indices) == False
        
        
        
class Test_transform_timefunctions(object):
    '''Unittest class for transform_timefunctions'''
    paramsets = {
        'test_all_equal': [ 
            dict(timefunction=[1,1,1,1]),
            dict(timefunction=[4,4,4,4])
        ],
        
        'test_equaly_spaces': [
            dict(timefunction=[0,1,2,3,4,5],
                 expected=[0,0,0,0,0,0]),
        ],
        
        'test_not_monotonic': [
            dict(timefunction=[2,1,4,1,10,100]),
            dict(timefunction=[10,9,8,7,8,4])
        ],
        
        'test_widely_spreaded': [
            dict(timefunction=[0,3,4,6,8,11],
                 expected=[np.log(1), np.log(3),np.log(1),np.log(2),
                            np.log(2),np.log(3)]),
            dict(timefunction=[0,2,4,8,16,32,64],
                 expected=[np.log(1), np.log(2),np.log(2),np.log(4),np.log(8),
                            np.log(16),np.log(32)]),
            
        ],
        
        'test_not_starting_zero': [
            dict(timefunction=[1,2,3,4,5]),
            dict(timefunction=[2,3,4,5,6]),
        ],
    } 
                            
    @params(paramsets['test_all_equal'])        
    def test_all_equal(self, timefunction):
        '''Test with horizontal function'''
        with pytest.raises(ValueError) as excinfo:
            transform_timefunction(timefunction)
        assert excinfo.value.message == \
                   "The Timewarping Functions have to be monotonic"
                   
    @params(paramsets['test_equaly_spaces'])                 
    def test_equaly_spaces(self, timefunction, expected):
        '''Test with equaly spaced timefunctions'''
        result = transform_timefunction(timefunction)
        assert np.allclose(result, expected)
                 
    @params(paramsets['test_not_monotonic'])                   
    def test_not_monotonic(self, timefunction):
        ''' Test with non monotonic timefunction'''
        with pytest.raises(ValueError) as excinfo:
            transform_timefunction(timefunction)
        assert excinfo.value.message == \
                   "The Timewarping Functions have to be monotonic"
              
    @params(paramsets['test_widely_spreaded'])                      
    def test_widely_spreaded(self, timefunction, expected):
        ''' Test with widley spreaded timefunction'''
        result = transform_timefunction(timefunction)
        assert np.allclose(result, expected)
        
        
class Test_get_monotonic_indices(object):    
    '''Unittest class for transform_timefunctions'''
    paramsets = {
        'test_already_monotonic': [ 
            dict(input=[1,2,3,4,5,6], expected=[1,2,3,4,5,6]),
        ],
        
        'test_all_equal': [
            dict(input=[1,1,1,1,1]),
        ],
        
        'test_start_equal': [
            dict(input=[1,1,2,3,4,5], expected=[1,1.01,2,3,4,5]),
        ],
        
        'test_mid_equal': [
            dict(input=[1,2,2,3,3,4,5], expected=[1,2,2.01,3,3.01,4,5]),
        ],
        
        'test_end_equal': [
            dict(input=[1,2,3,4,5,5], expected=[1,2,3,4,4.99,5]),
        ],
        
        'test_start_and_mid_and_end_equal': [
            dict(input=[1,1,2,3,3,4,5,5], expected=[1,1.01,2,3,3.01,4,4.99,5]),
        ],
        
        
    } 
    
    @params(paramsets['test_already_monotonic'])
    def test_already_monotonic(self, input, expected):
        ''' Test a function which is already monotonic '''
        result = get_monotonic_indices(input)
        assert np.allclose(result, expected)
        
    @params(paramsets['test_all_equal'])
    def test_all_equal(self, input):
        ''' Test a function where all elements are equal '''
        epsilon = 0.01
        with pytest.raises(ValueError) as excinfo:
            get_monotonic_indices(input,epsilon=epsilon)
        assert excinfo.value.message == "First and Last element are equal"
           
    @params(paramsets['test_start_equal'])
    def test_start_equal(self, input, expected):
        ''' Test a function where the first two elements are equal '''
        epsilon = 0.01
        result = get_monotonic_indices(input, epsilon=epsilon)
        result = [round(item, 2) for item in result]
        assert result == expected 
            
    @params(paramsets['test_mid_equal'])
    def test_mid_equal(self, input, expected):
        ''' Test a function where the middle two elements are equal '''
        epsilon = 0.01
        result = get_monotonic_indices(input, epsilon=epsilon)
        result = [round(item, 2) for item in result]
        assert result == expected  
            
    @params(paramsets['test_end_equal'])
    def test_end_equal(self, input, expected):
        ''' Test a function where the last two elements are equal '''
        epsilon = 0.01
        result = get_monotonic_indices(input, epsilon=epsilon)
        result = [round(item, 2) for item in result]
        assert result == expected  

    @params(paramsets['test_start_and_mid_and_end_equal'])
    def test_start_and_mid_and_end_equal(self, input, expected):
        ''' Test a function where multiple elements are equal '''
        epsilon = 0.01
        result = get_monotonic_indices(input, epsilon=epsilon)
        result = [round(item, 2) for item in result]
        assert result == expected
            