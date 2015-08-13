# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:24:01 2015

@author: erhe01
"""
import sys
import os
sys.path.append(os.sep.join([".."]*2))
import time
import numpy as np
from space_partitioning.cluster_tree import ClusterTree
TESTOUTPATH = os.sep.join([".."]*3 + ["test_output"])

def distance_objective(x, test_value):
    return np.linalg.norm(np.asarray(x)-np.asarray(test_value))

def run_cluster_hierarchy_construction(N, K, X, path):
    cluster_tree = ClusterTree(N, K)
    cluster_tree.construct(X)
    cluster_tree.save_to_file(path)
    return cluster_tree

def run_search_from_path(N, K, path, test_value, error_margin=0.1):
    cluster_tree = ClusterTree(N, K)
    cluster_tree.load_from_file(TESTOUTPATH + os.sep + "tree")
    run_search(N, K, cluster_tree, test_value, error_margin)

def run_search(N, K, cluster_tree, test_value, error_margin=0.1):
    n_candidates = 5
    approx_distance, approx_result = cluster_tree.find_best_example_excluding_search_candidates(distance_objective, test_value, n_candidates)
    distance, result = cluster_tree.find_best_example_exhaustive(distance_objective, test_value)
    print "best result for", test_value
    print "exhaustive:", result, distance
    print "approximate:", approx_result, approx_distance
    assert distance < error_margin


def test_construction_and_search():
    n_samples = 10000
    n_dim = 3
    N = 4
    K = 4
    X_shape = (n_samples, n_dim)
    X = np.random.random(X_shape)
    test_value = X[0]
    print X.shape
    start = time.clock()
    path = TESTOUTPATH + os.sep + "tree"
    cluster_tree = run_cluster_hierarchy_construction(N, K, X, path)
    print "finished construction in ", time.clock()-start, "seconds"
    start = time.clock()
    run_search(N, K, cluster_tree, test_value)
    run_search_from_path(N, K, path, test_value)
    print "finished search in ", time.clock()-start, "seconds"
    return
    

if __name__ == "__main__":
    test_construction_and_search()
