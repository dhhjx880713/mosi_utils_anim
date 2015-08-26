# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:54:12 2015

@author: erhe01
"""
import heapq
import numpy as np
import uuid
from . import KDTREE_WRAPPER_NODE

class ClusterTreeNode(object):
    """ Node for the ClusterTree class that subdivides a list samples into clusters 
    by containing a child node for each cluster.

    Parameters
    ---------

    """
    def __init__(self, unique_id, depth, indices, mean, clusters, node_type, is_leaf):
        self.id = str(unique_id)
        self.clusters = clusters
        self.mean = mean
        self.leaf = is_leaf
        self.type = node_type
        self.depth = depth
        self.indices = indices

    def find_best_example_knn(self, obj, data, k=50):
        """Return the best example based on the evaluation using an objective function.
            Interpolates the best k results.
        """
        if self.leaf:
            result_queue = []
            for i in xrange(len(self.clusters)):
                result = self.clusters[i].knn_interpolation(obj, data, k)
                heapq.heappush(result_queue, result)
            return heapq.heappop(result_queue)
     
    def find_best_example(self, obj, data):   
        """Return the best example based on the evaluation using an objective function.
        """
        if self.leaf:
            result_queue = []
            for i in xrange(len(self.clusters)):
                result = self.clusters[i].find_best_example(obj, data)
                heapq.heappush(result_queue, result)
            return heapq.heappop(result_queue)
        else:
            best_index, best_value = self.find_best_cluster(obj, data)
            return self.clusters[best_index].find_best_example(obj, data)

    def find_best_example_exhaustive(self, obj, data):
        """Return the best example based on the evaluation using an objective function.
        """
        result_queue = []
        for i in xrange(len(self.clusters)):
            result = self.clusters[i].find_best_example_exhaustive(obj, data)
            heapq.heappush(result_queue, result)
        return heapq.heappop(result_queue)

    def find_best_cluster(self, obj, data):
        """ Return the best cluster based on the evaluation using an objective function.

        Parameters
        ----------
        * obj: function
            Objective function of the form: scalar = obj(x,data).
        * data: Tuple
            Additional parameters for the objective function.
        * n_candidates: Integer
            Maximum number of candidates
        """
        best_value = np.inf
        best_index = 0
        for cluster_index in xrange(len(self.clusters)):
            sample = self.clusters[cluster_index].mean
            cluster_value = obj(sample,data)
            
            if cluster_value < best_value:
                best_index = cluster_index
                best_value = cluster_value
        return best_index, best_value

    def find_best_cluster_candidates(self, obj, data, n_candidates):
        """Return the clusters with the least cost based on 
        an evaluation using an objective
        function.
        
        Parameters
        ----------
        * obj: function
            Objective function of the form: scalar = obj(x,data).
        * data: Tuple
            Additional parameters for the objective function.
        * n_candidates: Integer
            Maximum number of candidates
            
        Returns
        -------
        * best_candidates: list of (value, ClusterTreeNode) tuples.
            List of candidates ordered using the objective function value.
        """
        result_queue = []
        for cluster_index in xrange(len(self.clusters)):
            sample = self.clusters[cluster_index].mean
            cluster_value = obj(sample,data)
            heapq.heappush(result_queue,(cluster_value,self.clusters[cluster_index] ) )
      
        return result_queue[:n_candidates]

    def get_desc(self):
        """Used to save the node to file.
        Returns
        ------
        * node_desc: dict
            Dictionary containing the properties of the node.
        """
        node_desc = {}
        node_desc["id"] = str(self.id)
        node_desc["depth"] = self.depth
        node_desc["type"] = self.type
        children = []
        for node in self.clusters:
            children.append(str(node.id))
        node_desc["children"] = children
        node_desc["mean"] = self.mean.tolist() 
        
        if self.depth == 0:
            node_desc["indices"] = "all"
        else:
            if self.indices is not None:
                node_desc["indices"] = self.indices
            else:
                node_desc["indices"] = []
        return node_desc

    def get_node_desc_list(self):
        """Iteratively builds a dictionary containing a description of all 
           nodes.
        Returns:
        -------
        * node_desc_dict: dict
            Contains the result of ClusterTreeNode.get_desc() for all nodes.
        """
        node_desc_dict = {}
        stack = [self]
        node_desc_dict["root"] = str(self.id)
        node_desc_dict["nodes"] = {}
        while len(stack) > 0:
            node = stack.pop(-1)
            node_desc = node.get_desc()
            node_desc_dict["nodes"][node_desc["id"]] = node_desc
            if node.type != KDTREE_WRAPPER_NODE:
                for c in node.clusters:
                    stack.append(c)
        return node_desc_dict

    def get_number_of_leafs(self):
        if self.leaf:
            print "reached leaf at depth", self.depth, "with", len(self.indices), "points"
            return 1
        else:
            n_leafs = 0
            if self.indices is not None:
                print self.depth, len(self.indices)
            for c in self.clusters:
                n_leafs += c.get_number_of_leafs()
            return n_leafs


