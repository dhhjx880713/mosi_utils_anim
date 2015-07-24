# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 17:15:22 2015

@author: erhe01
"""


import numpy as np

def transform_point_from_cad_to_opengl_cs(point):
    """ Transforms a 3d point represented as a list from left handed to the 
        right handed coordinate system
    """

    transform_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    return np.dot(transform_matrix,point).tolist()

def transform_unconstrained_indices_from_cad_to_opengl_cs(unconstrained_indices):
    """ Transforms a list indicating unconstrained dimensions from left handed to the 
        right handed coordinate system
    """
    new_indices = []
    for i in unconstrained_indices:
        if i == 0:
            new_indices.append(0)
        elif i == 1:
            new_indices.append(2)
        elif i == 2:
            new_indices.append(1)
    return new_indices
