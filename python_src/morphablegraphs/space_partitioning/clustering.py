import numpy as np
from sklearn import cluster


CLUSTERING_METHOD_KMEANS = 0

def all_equal(list):
    for x in list:
        for y in list:
            if not np.all(np.equal(x,y)):
                return False
    return True

def _max_var_dim(features, n_points):
    # alternative method from spatialtree library
    mean = np.zeros(features.shape[1])
    deviation = np.zeros(features.shape[1])
    for i in range(n_points):
        mean += features[i]
        deviation += features[i] ** 2
    # mean
    mean /= n_points
    # variance
    sigma = (deviation - (n_points * mean ** 2)) / (n_points - 1.0)
    return np.argmax(sigma)


def _labels_to_cluster_indices(labels, indices, n_subdivisions):
    cluster_indices = [[] for i in range(n_subdivisions)]
    print("cluster indices",cluster_indices,labels)
    n_samples = len(labels)
    if indices is None:
        for i in range(n_samples):
            l = labels[i]
            cluster_indices[l].append(i)
    else:
        for i in range(n_samples):
            l = labels[i]
            original_index = indices[i]
            cluster_indices[l].append(original_index)
    print(len(cluster_indices))#cluster_indices
    return cluster_indices




def _get_labels_from_kmeans(features, indices, n_subdivisions):
    kmeans = cluster.KMeans(n_clusters=n_subdivisions)
    if indices is None:
        labels = kmeans.fit_predict(features[:])
    else:
        labels = kmeans.fit_predict(features[indices])
    return labels


def _find_clusters_kmeans(features, indices, n_subdivisions):
    """Use the kmeans algorithm of scikit learn to assign labels to samples according
    to clusters.
    """
    labels = _get_labels_from_kmeans(features, indices, n_subdivisions)
    cluster_indices = [[] for i in range(n_subdivisions)]
    #print "cluster indices",cluster_indices,labels
    n_samples = len(labels)
    if indices is None:
        for i in range(n_samples):
            l = labels[i]
            cluster_indices[l].append(i)
    else:
        for i in range(n_samples):
            l = labels[i]
            original_index = indices[i]
            cluster_indices[l].append(original_index)
    return cluster_indices



def find_clusters(features, indices, options):
    print("detect clusters", options, len(indices), indices is None or len(indices) > 3)#options["n_subdivisions"]
    method = options["clustering_method"]
    n_subdivisions = options["n_subdivisions"]
    #if indices is not None and all_equal(features[indices]):
    #    clusters = [[idx] for idx in indices]
    if indices is None or len(indices) > n_subdivisions:#3

        clusters = _find_clusters_kmeans(features, indices, n_subdivisions)


        print("found",len(clusters), "clusters")
        for c in clusters:
            if len(c) == 0:
                clusters.remove(c)#del([c])
        if len(clusters) == 1 and clusters[0] == indices:
            clusters = [[idx] for idx in indices]
        for idx,c in enumerate(clusters):
            print(idx, len(c))
        return clusters

    elif len(indices) > 1:
        return [[i] for i in indices] #zip(indices)
    else:#elif len(indices) == 1:#indices is not None and
        print("return indices",indices)
        return indices
