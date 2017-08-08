# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:02:54 2015

@author: erhe01
"""

import uuid
import numpy as np
from .kdtree import KDTree
from . import KDTREE_WRAPPER_NODE


class KDTreeWrapper(object):
    """ Wrapper for a KDTree used as leaf of the ClusterTree.

    Parameters
    ---------
    * dim: Integer
        Number of dimensions of the data to be considered.
    """
    def __init__(self, dim):
        self.id = str(uuid.uuid1())
        self.kdtree = KDTree()
        self.dim = dim
        #self.indices = []
        self.type = KDTREE_WRAPPER_NODE

    def construct(self, data, indices):
        #self.indices = indices
        self.kdtree.construct(data[indices].tolist(), self.dim)

    def find_best_example(self, obj, data):
        return self.kdtree.find_best_example(obj, data, 1)[0]

    def find_best_example_exhaustive(self, obj, data):
        return self.kdtree.df_search(obj, data)

    def knn_interpolation(self, obj, data, k=50):
        """Searches for the k best examples and performs KNN-Interpolation
        between them to produce a new sample
        with a low objective function value.
        """
        results = self.kdtree.find_best_example(obj, data, k)
        if len(results) > 1:
            distances, points = list(zip(*results))
            weights = self._get_knn_weights(distances)
            new_point = np.zeros(len(points[0]))
            for i in range(len(weights)):
                new_point += weights[i] * np.array(points[i])
            return obj(new_point, data), new_point
        else:
            return results[0]

    def _get_knn_weights(self, distances):
        influences = []
        for distance in distances[:-1]:
            influences.append(1/distance - 1/distances[-1])
        ## calculate weight based on normalized influence
        weights = []
        sum_influence = np.sum(influences)
        for i in range(len(influences)):
            weights.append(influences[i]/sum_influence)
        return weights

    def get_desc(self):
        node_desc = dict()
        node_desc["id"] = str(self.id)
        node_desc["type"] = self.type
        node_desc["children"] = []
        node_desc["indices"] = []
        return node_desc
