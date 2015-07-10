# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:21:41 2015

@author: hadu01
"""

import os
import sys
from libtest import params, pytest_generate_tests
sys.path.append("..")

from motion_generator.lib  import constraint 



# constraints should be a dictionary, contain "orientation" and "position" 
constraints = {"position": [20, 30, 40], "orientation": [None, None, None]}
param_constraint_distance = [
    {'constraints': {"position": [20, 30, 40], 
                     "orientation": [None, None, None]},
     'target_position': [25, 30, 45],
     'target_orientation': [None, None, None],
     'res':  7.071},
    {'constraints': {"position": [None, None, None],
                      "orientation": [30, 60, None]},
     'target_position': [None, None, None],
     'target_orientation': [30, 60, None],
     'res': 0}
]

@params(param_constraint_distance)
def test_constraint_distance(constraints, 
                             target_position, 
                             target_orientation,
                             res):
    """Unit test for constraint distance.
    """
    dist = constraint.constraint_distance(constraints, 
                               target_position,
                               target_orientation)
    assert round(sum(dist), 3) == res
    
if __name__ == "__main__":
    test_constraint_distance(constraints, [20, 30, 40], [None, None, None], 0.0)
    