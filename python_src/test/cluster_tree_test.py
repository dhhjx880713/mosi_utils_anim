# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:24:01 2015

@author: erhe01
"""
import sys
sys.path.append("..")
import time
import os
import numpy as np
from motion_model.space_partitioning.cluster_tree import ClusterTree
TESTOUTPATH = os.sep.join([".."]*2+ ["test_output"])

def distance_objective(x, test_value):
    return np.linalg.norm(np.asarray(x)-np.asarray(test_value))

def run_cluster_hierarchy_construction(N, K, X, path):
 
    cluster_tree = ClusterTree(N, K)
    cluster_tree.construct(X)
    cluster_tree.save_to_file(path)#.json
    
def run_search(N, K, path, test_value):
    cluster_tree = ClusterTree(N, K)
    cluster_tree.load_from_file(TESTOUTPATH + os.sep + "tree")
    error_margin = 0.1
    n_candidates = 2
    distance, result = cluster_tree.find_best_example_exluding_search_candidates_knn(distance_objective, test_value, n_candidates)
    print test_value, result, distance
    
    assert distance < error_margin
    return
    
def test_construction_search():
    n_samples = 10000
    n_dim = 3
    N = 4
    K = 4#0
    X_shape = (n_samples, n_dim)
    X = np.random.random(X_shape)
    test_value = X[0]
    print X.shape
    start = time.clock()
    #kdtree = KDTree()
    #test_kd_tree(X)
    path = TESTOUTPATH + os.sep + "tree"
    run_cluster_hierarchy_construction(N, K, X, path)
    print "finished construction in ",time.clock()-start, "seconds"
    start = time.clock()
    run_search(N, K, path, test_value)
    print "finished search in ",time.clock()-start, "seconds"
    return
    
if __name__ == "__main__":
    test_construction_search()

    
    
