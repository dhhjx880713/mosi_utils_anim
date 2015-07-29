# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:02:54 2015

@author: erhe01
"""

import uuid
import numpy as np
from kdtree import KDTree
from .  import KDTREE_WRAPPER_NODE


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
        self.indices = []
        self.type = KDTREE_WRAPPER_NODE
       
        
    def construct(self, X, indices):
        self.indices = indices
        self.kdtree.construct(X[indices].tolist(), self.dim)
        return
        
    def find_best_example(self,obj,data):
        return self.kdtree.find_best_example(obj,data,1)[0]

    def knn_interpolation(self, obj, data, k=50):
        
        results = self.kdtree.find_best_example(obj, data, k)#50
        if len(results)>1:#K
            distances, points = zip(*results)
            #distances, points = self.kdtree.query(target, )
        
            influences = []
            furthest_distance = distances[-1]
            #print furthest_distance,"#################"
            for d in distances[:-1]:
                 influences.append(1/d - 1/furthest_distance)
            ## calculate weight based on normalized influence
            weights = []
            n_influences = len(influences)
            sum_influence = np.sum(influences)
            for i in xrange(n_influences):
               weights.append(influences[i]/sum_influence)
    
            new_point = np.zeros(len(points[0]))
            for i in xrange(n_influences):
    
                #print index
                new_point += weights[i] * np.array(points[i])
            #print new_point,"#####"
            return obj(new_point,data), new_point # return also the evaluation of the new point
        else:
            return results[0]
        
    def get_desc(self):
        """
        used by save_to_file
        """
        node_desc = {}
        node_desc["id"] = str(self.id)
        #node_desc["depth"] = self.depth
        node_desc["type"] = self.type
     
        node_desc["children"] = []
        #node_desc["mean"] = self.mean.tolist() 
        
        node_desc["indices"] = self.indices
        return node_desc
        