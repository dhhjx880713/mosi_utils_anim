# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:58:08 2015

@author: erhe01 
"""
import numpy as np
import heapq
import json
import cPickle as pickle
from cluster_tree_node_builder import ClusterTreeNodeBuilder


DEFAULT_N_SUBDIVISIONS_PER_LEVEL = 4
DEFAULT_N_LEVELS = 4
MIN_N_SUBDIVISIONS_PER_LEVEL = 2
MIN_N_LEVELS = 1
MAX_DIMENSIONS = 10


class ClusterTree(object):
    """
    Create a hiearchy of clusters using KMeans and then use a kdtree for the leafs
    #TODO make faster
    Parameters
    -----------
    * n_subdivisions : Integer
        Number of subclusters/children per node in the tree. At least 2.
    * max_level : Integer
        Maximum levels in the tree. At least 1.
    
    """
    def __init__(self, n_subdivisions=DEFAULT_N_SUBDIVISIONS_PER_LEVEL, max_level=DEFAULT_N_LEVELS, dim=MAX_DIMENSIONS):
        self.n_subdivisions = max(n_subdivisions, MIN_N_SUBDIVISIONS_PER_LEVEL)
        self.max_level = max(max_level, MIN_N_LEVELS)
        self.dim = dim
        self.root = None
        self.data = None
  
        return
      
    def save_to_file(self,file_name):
        #save tree structure to file
        fp = open(file_name+".json", "wb")
        node_desc_list = self.root.get_node_desc_list()
        node_desc_list["data_shape"] = self.data.shape
        json.dump(node_desc_list, fp, indent=4)
        fp.close()
        ## save data to file
        self.data.tofile(file_name+".data")
        return
        
    def load_from_file(self, file_name):
        fp = open(file_name+".json", "r")
        node_desc = json.load(fp)
        fp.close()
        data_shape = node_desc["data_shape"]
        self.data = np.fromfile(file_name+".data").reshape(data_shape)
        self.dim = data_shape[1]
        root_id = node_desc["root"]
        node_builder = ClusterTreeNodeBuilder(self.n_subdivisions, self.max_level, self.dim)
        self.root = node_builder.construct_from_node_desc_list(root_id, node_desc, self.data)

    def save_to_file_pickle(self, file_name):
        pickle_file_name = file_name
        pickle_file = open(pickle_file_name, 'wb')
        pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)
        pickle_file.close()        
       
    def load_from_file_pickle(self,file_name):
        pickle_file_name = file_name
        pickle_file = open(pickle_file_name, 'rb')
        data = pickle.load(pickle_file)
        self.data = data.data
        self.dim = self.dim
        self.root = data.root
        self.max_level = data.max_level
        self.n_subdivisions = data.n_subdivisions
        pickle_file.close()     
      
    def construct(self, data):
        self.data = data
        self.dim = min(self.data.shape[1], self.dim)
        node_builder = ClusterTreeNodeBuilder(self.n_subdivisions, self.max_level, self.dim)
        self.root = node_builder.construct_from_data(self.data)

    def find_best_example(self, obj, data):
        return self.root.find_best_example(obj, data)

    def find_best_example_exhaustive(self, obj, data):
        return self.root.find_best_example_exhaustive(obj, data)

    def find_best_example_excluding_search(self, obj, data):
        node = self.root
        level = 0
        while level < self.max_level and not node.leaf:
            print "level", level
            index, value = node.find_best_cluster(obj, data, use_mean=True)
            node = node.clusters[index]
            level += 1
        print level, node.leaf
        return node.find_best_example(obj, data)
          
    def find_best_example_excluding_search_candidates(self, obj, data, n_candidates=1):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        print "search with",n_candidates,"candidates in tree with ", self.n_subdivisions," subdivisions and ", self.max_level, "levels"
        results = []
        candidates = []
        candidates.append((np.inf, self.root))
        level = 0
        while len(candidates) > 0:
            new_candidates = []
            for value, node in candidates:
                if not node.leaf:
                    good_candidates = node.find_best_cluster_candidates(obj, data, n_candidates)
                    for c in good_candidates:
                        heapq.heappush(new_candidates, c)
                else:
                    kdtree_result = node.find_best_example(obj, data) 
                    heapq.heappush(results, kdtree_result)
            
            candidates = new_candidates[:n_candidates]
            level += 1

        if len(results) > 0:
            return heapq.heappop(results)    
        else:
            print "#########failed to find a result########"
            return np.inf, self.data[self.root.indices[0]]
        
    def find_best_example_excluding_search_candidates_boundary(self, obj, data, n_candidates=5):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KDTree is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
            Uses boundary based on maximum cost of last iteration to ignore bad candidates.
            Note requires more candidates to prevent
        """
        results = []
        candidates = []
        candidates.append((np.inf, self.root))
        level = 0
        while len(candidates) > 0:
            boundary = max([c[0] for c in candidates])
            print boundary
            new_candidates = []
            for value, node in candidates:
                
                if not node.leaf:
                    good_candidates = node.find_best_cluster_candidates(obj, data, n_candidates=n_candidates)
                    for c in good_candidates:
                         heapq.heappush(new_candidates, c)
                else:
                    kdtree_result = node.find_best_example(obj, data)
                    heapq.heappush(results, kdtree_result)

            candidates = [c for c in new_candidates[:n_candidates] if c[0] < boundary]
            if len(results) == 0 and len(candidates) == 0:
                candidates = new_candidates[:n_candidates]
            level += 1

        if len(results) > 0:
            return heapq.heappop(results)    
        else:
            print "#################failed to find a good result"
            return np.inf, self.data[0]

    def find_best_example_excluding_search_candidates_knn(self, obj, data, n_candidates=1, k=50):
        """ Traverses the cluster hierarchy iteratively by evaluating the means
            of the clusters at each level based on the objective function. 
            At the last level the method find_best_example for a KNN Interpolation is used.
            Multiple candidates are kept at each level in order to find the global
            optimum.
        """
        results = []
        candidates = []
        candidates.append((np.inf, self.root))
        level = 0
        while len(candidates) > 0:
            new_candidates = []
            for value, node in candidates:
                if not node.leaf:
                    good_candidates = node.find_best_cluster_candidates(obj, data, n_candidates=n_candidates)
                    for c in good_candidates:
                        heapq.heappush(new_candidates, c)
                else:
                    kdtree_result = node.find_best_example_knn(obj, data, k)
                    heapq.heappush(results, kdtree_result)

            candidates = new_candidates[:n_candidates]
            level += 1

        if len(results)>0:
            return heapq.heappop(results)    
        else:
            print "#################failed to find a result"
            return np.inf, self.data[self.root.indices[0]]
