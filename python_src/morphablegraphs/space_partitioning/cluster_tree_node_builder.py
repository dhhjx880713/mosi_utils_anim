# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:07:11 2015

@author: erhe01
"""


import numpy as np
from sklearn import cluster
import uuid
from .  import KDTREE_WRAPPER_NODE, LEAF_NODE, INNER_NODE, ROOT_NODE
from cluster_tree_node import ClusterTreeNode
from kdtree_wrapper_node import KDTreeWrapper


class ClusterTreeNodeBuilder(object):
    """ Creates a ClusterTreeNode based on samples. It subdivides samples using KMeans and
        creates a child node for each subdivision. Child nodes can be ClusterTreeNodes
        or KDTreeNodes depending on the current depth and the maximum depth.
        Stores the indices refering to the samples stored in ClusterTree.

    Parameters
    ---------
    * n_subdivisions: Integer
        Number of subdivisions.
    * max_level: Integer
        Madataimum number of levels.
    * dim: Integer
        Number of dimensions of the data.
    """
    def __init__(self, n_subdivisions, max_level, dim, store_indices):
 
        self.n_subdivisions = n_subdivisions
        self.max_level = max_level
        self.dim = dim
        self.kmeans = None#cluster.KMeans(n_clusters=self.n_subdivisions)
        self.store_indices = store_indices

    def _calculate_mean(self, data, indices):
        if indices is None:
            n_samples = len(data)
            mean = np.mean(data[:, :self.dim], axis=0)
        else:
            n_samples = len(indices)
            mean = np.mean(data[indices, :self.dim], axis=0)
        return mean, n_samples
        
    def _get_node_type_from_depth(self, depth):
        """Decide on on the node type based on the maximum number of levels."""
        if depth < self.max_level - 1:
            if depth == 0:
                node_type = ROOT_NODE
            else:
                node_type = INNER_NODE
        else:
            node_type = LEAF_NODE
        return node_type
        
    def _detect_clusters(self, data, indices, n_samples):
        """Use the kmeans algorithm of scipy to labels to samples according 
        to clusters.
        """
        self.kmeans = cluster.KMeans(n_clusters=self.n_subdivisions)
        if indices is None:
            labels = self.kmeans.fit_predict(data[:, :self.dim])
        else:
            labels = self.kmeans.fit_predict(data[indices, :self.dim])
            
        cluster_indices = [[] for i in xrange(self.n_subdivisions)]
        if indices is None:
            for i in xrange(n_samples):
                l = labels[i]
                cluster_indices[l].append(i)
        else:
            for i in xrange(n_samples):
                l = labels[i]
                original_index = indices[i]
                cluster_indices[l].append(original_index)
                
        return cluster_indices
        
    def construct_from_data(self, data, indices=None, depth=0):
        """ Creates a divides sample space into partitions using KMeans and creates
             a child for each space partition.
             
        Parameters
        ----------
        * data: np.ndarray
            2D array of samples
        * indices: list
            indices of data that should be considered for the subdivision
        * depth: int
            current depth used with self.max_level to decide the type of node and the type of subdivisions
        """
        clusters = []

        node_type = self._get_node_type_from_depth(depth)
        #self.samples = data # has always at least 1 sample
        mean, n_samples = self._calculate_mean(data, indices)
        
        if n_samples > self.n_subdivisions:#number of samples at least equal to the number of clusters required for kmeans
            ## create subdivision
            cluster_indices = self._detect_clusters(data, indices, n_samples)
            if depth < self.max_level:
                is_leaf = False
                #node_type = INNER_NODE
                ## create inner node for each cluster
                for j in xrange(len(cluster_indices) ):
#                    if len(cluster_data[j]) > 0: #ignore clusters that are empty
                
                    if len(cluster_indices[j])>0:
                        child_node = self.construct_from_data(data, cluster_indices[j], depth+1)
                        clusters.append(child_node)
            else:
                is_leaf = True
                #node_type = LEAF_NODE
                ## create kdtree for each cluster
                cluster_indices = cluster_indices
                for j in xrange(len(cluster_indices) ):
#                    if len(cluster_data[j]) > 0: #ignore clusters that are empty
                    if len(cluster_indices[j])>0:
                        child_node = KDTreeWrapper(self.dim)
                        child_node.construct(data, cluster_indices[j])
                        clusters.append(child_node)
        else:
            #not enough samples to further divide it
            #so stop before reaching level K
            is_leaf = True
            #node_type = LEAF_NODE
            child_node = KDTreeWrapper(self.dim)
            child_node.construct(data, indices)
            clusters.append(child_node)
        if self.store_indices:
            return ClusterTreeNode(uuid.uuid1(), depth, indices, mean, clusters, node_type, is_leaf)
        else:
            return ClusterTreeNode(uuid.uuid1(), depth, None, mean, clusters, node_type, is_leaf)

    def construct_from_node_desc_list(self,node_id, node_desc, data):
        """Recursively rebuilds the cluster tree given a dictionary containing 
           a description of all nodes and the data samples.
        
        Parameters
        ---------
        * node_id: String
            Unique identifier of the node.
        * node_desc: dict
            Dictionary containing the properties of each node. The node id is used as key.
        * data: np.ndarray
            Data samples.
        """
        desc = node_desc["nodes"][node_id]
        clusters = []
        node_type = desc["type"]
        mean = np.array(desc["mean"])
        depth = desc["depth"]
        if desc["type"] != ROOT_NODE:
            indices = desc["indices"]
        else:
            indices = []
            
        if desc["type"] != LEAF_NODE:
            is_leaf = False
            for c_id in desc["children"]:
                child_node = self.construct_from_node_desc_list(c_id, node_desc, data)
                clusters.append(child_node)
        else:
            is_leaf = True
            for c_id in desc["children"]:
                child_node = KDTreeWrapper(self.dim)
                child_node.id = c_id
                indices = node_desc["nodes"][c_id]["indices"]
                child_node.construct(data, indices)
                clusters.append(child_node)
            
        return ClusterTreeNode(node_id, depth, indices, mean, clusters, node_type, is_leaf)
