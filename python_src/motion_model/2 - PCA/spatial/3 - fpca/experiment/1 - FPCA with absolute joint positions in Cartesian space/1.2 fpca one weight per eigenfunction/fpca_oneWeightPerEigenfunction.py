# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 16:52:39 2014
@Brief: this script compares the performance of standard PCA and functional PCA
        for root joint trajectory. For functional PCA, we use one weight for
        all dimensions of eigenfunctions. MSE are measured as evaluation for
        both approaches.
@author: hadu01
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from lib.PCA import PCA
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects

ROOT_DIR = os.sep.join([".."] * 7)


def clean_path(path):
    """
    Generate absolute path starting with '\\\\?\\' to avoid failure of loading
    because of long path in windows

    Parameters
    ----------
    * path: string
    \tRelative path

    Return
    ------
    * path: string
    \tAbsolute path starting with '\\\\?\\'
    """
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [directory for directory in os.getcwd().split(os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd)-relative_levels] + [directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path


def get_input_data_folder():
    """
    Return folder path of feature data without trailing os.sep
    """
    data_dir_name = "data"
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "1 - preprocessing"
    action = 'experiments'
    test_feature = "1 - FPCA with absolute joint positions in Cartesian space"
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action,
                             test_feature])
    return input_dir


def plot_root_trajectory(data):
    """
    Plot 3d root trajectories in data

    Parameters:
    -----------
    *data: 3d array: n_sample * n_frames * dim_point

    """
    if len(data.shape) == 3:
        # data contains multiple trajectories
        n_samples, n_frames, dim = data.shape
        # change coordinate
        temp = np.zeros((n_samples, n_frames, dim))
        temp[:, :, 0] = data[:, :, 0]
        temp[:, :, 2] = data[:, :, 1]
        temp[:, :, 1] = data[:, :, 2]
        print temp[:, 1, 2]
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x_max = np.max(temp[:, :, 0])
        x_min = np.min(temp[:, :, 0])
        y_max = np.max(temp[:, :, 1])
        y_min = np.min(temp[:, :, 1])
        z_max = np.max(temp[:, :, 2])
        z_min = np.min(temp[:, :, 2])
        x_mean = np.mean(temp[:, :, 0])
        y_mean = np.mean(temp[:, :, 1])
        z_mean = np.mean(temp[:, :, 2])
#        print x_max
#        print x_min
#        print y_max
#        print y_min
#        print z_max
#        print z_min
#        print x_mean
#        print y_mean
#        print z_mean
        for i in xrange(n_samples):
            tmp = temp[i, :, :]
            tmp = np.transpose(tmp)
            ax.plot(*tmp)
    elif len(data.shape) == 2:
        # data constains one trajectory
        n_frames, dim = data.shape
        temp = np.zeros((n_frames, dim))
        temp[:,0] = data[:,0]
        temp[:,2] = data[:,1]
        temp[:,1] = data[:,2]
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x_max = np.max(temp[:, 0])
        x_min = np.min(temp[:, 0])
        y_max = np.max(temp[:, 1])
        y_min = np.min(temp[:, 1])
        z_max = np.max(temp[:, 2])
        z_min = np.min(temp[:, 2])
        x_mean = np.mean(temp[:, 0])
        y_mean = np.mean(temp[:, 1])
        z_mean = np.mean(temp[:, 2])
        tmp = np.transpose(temp)
        ax.plot(*tmp)
    else:
        raise ValueError('The shape of data is not correct!')
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
#    print max_range
    ax.set_xlim(x_mean - max_range, x_mean + max_range)
    ax.set_ylim(y_mean - max_range, y_mean + max_range)
    ax.set_zlim(z_mean - max_range, z_mean + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.show()


def get_root_joint_data():
    """
    Return 3d root joint position
    
    Return:
    -------
    * data: 3d array
    \tData contains 3d position for root joints of 100 samples
    """
    input_dir = get_input_data_folder()
    if len(input_dir) > 116:
        input_dir = clean_path(input_dir)
    elementary_action = 'walk'
    motion_primitive = 'leftStance'
    filename = input_dir + os.sep + '%s_%s_featureVector.json' % (elementary_action, motion_primitive)
    with open(filename, 'rb') as handle:
        dic_feature_data = json.load(handle)
    temp = []
#   extract data from dic_feature_data
    for key, value in dic_feature_data.iteritems():
        temp.append(value)
    temp = np.asarray(temp)
    number_samples, number_frames, number_joint, len_point = temp.shape
#    print temp.shape
#    return temp    
    data = temp[:100, :, 0, :]
    print data.shape
    return data


def centralizeData(raw_data):
    """
    Compute mean trajectory and centralize data

    Parameters:
    -----------
    *raw_data: 3d array

    Returns:
    --------
    mean_trajectory: 2d array
    """
    n_samples, n_frames, dim = raw_data.shape
    mean_trajectory = np.zeros((n_frames, dim))
    centralized_data = np.zeros(raw_data.shape)
    for i in xrange(n_samples):
        mean_trajectory += raw_data[i, :, :]
    mean_trajectory = mean_trajectory/n_samples
    for i in xrange(n_samples):
        centralized_data[i, :, :] = raw_data[i, :, :] - mean_trajectory
    return centralized_data, mean_trajectory


def standardPCA(centralized_data):
    """
    Apply standard PCA on root trajectory data to reduce the dimension

    Parameters:
    -----------
    *centralized_data: 3d array
    \tContains cnetralized root joint trajectory data

    Returns:
    --------
    *eigenVectors: 2d array
    \tEach eigenvector is a column vector in eigenVectors
    """
    n_samples, n_frames, dim = centralized_data.shape
#   reshape data as n_samples * (n_frames * dim)
    mat = np.zeros((n_samples, n_frames * dim))
    for i in xrange(n_samples):
        tmp = np.ravel(centralized_data[i, :, :])
        mat[i, :] = tmp
    print mat.shape
    p = PCA(mat, fraction=0.99)
    npc = p.npc
    print 'number of eigenvectors is: ' + str(npc)
    eigenVectors = p.Vt[:npc]
    return eigenVectors


def evalStandardPCA(centralized_data, eigenVectors):
    """
    Evaluate performance of standard PCA by computing MSE between raw data and
    back projected data
    Parameters:
    -----------
    *centralized_data: 3d array
    \tContains cnetralized root joint trajectory data

    *eigenVectors: 2d array
    \tEach eigenvector is a column vector in eigenVectors

    Returns:
    --------
    *err: a value
    \tMES between raw data and backprojected data
    """
#   reshape data as a 2d matrix
    n_samples, n_frames, dim = centralized_data.shape
    mat = np.zeros((n_samples, n_frames * dim))
    for i in xrange(n_samples):
        tmp = np.ravel(centralized_data[i, :, :])
        mat[i, :] = tmp
#   project data into low dimensional space
    projectedVecs = []
    for i in xrange(n_samples):
        projection = np.dot(mat[i, :], eigenVectors.T)
        projection = np.ravel(projection)
        projectedVecs.append(projection)
    projectedVecs = np.array(projectedVecs)
#   backproject to high dimensional space
    backprojectedVecs = []
    for i in xrange(n_samples):
        backprojection = np.dot(np.transpose(eigenVectors), projectedVecs[i].T)
        backprojection = np.ravel(backprojection)
        backprojectedVecs.append(backprojection)
    backprojectedVecs = np.array(backprojectedVecs)
    err = 0
    for i in xrange(n_samples):
        err += np.linalg.norm(mat[i, :] - backprojectedVecs[i, :])
    err = err/n_samples
    return err


def fpca(centralized_data, n_harm=3, n_knots=8):
    """
    Represent functional data as expansion of basis functions
    """
    n_samples, n_frames, dim = centralized_data.shape
#   reshape centralized_data as a 3d array: n_frame * n_samples * dim
    tmp = np.zeros((n_frames, n_samples, dim))
    for i in xrange(n_frames):
        for j in xrange(n_samples):
            tmp[i, j] = centralized_data[j, i, :]
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.array(tmp))  # convert numpy to r data
    rcode = '''
        library(fda)
        # initialize parameters' value
        data = %s
        n_frames = %d
        n_knots = %d
        n_harm = %d
        basisobj = create.bspline.basis(c(0, n_frames - 1), n_knots)
        tmp = smooth.basis(argvals=seq(0, {n_frames-1}, len = {n_frames}),
                           y = {data}, fdParobj = basisobj)
        fd = tmp$fd
        pca = pca.fd(fd, nharm = n_harm)
        pcaVarmax = varmx.pca.fd(pca)
        scores = pcaVarmax$scores
        eigenVecs = pcaVarmax$harmonics
        # scores is a 3d matrix, n_samples * n_eigval * n_dim, sum over n_dim
        # to get oen weight for each dimension
        n_samples = dim(scores)[1]
        n_dim = dim(data)[3]
        coefficients = matrix(0, n_samples, n_harm)
        for (i in 1:n_samples){
            temp = rowSums(scores[i, , ])
            coefficients[i,] = temp
        }
        eigVecs = pcaVarmax$harmonics$coefs
        # reconstruct raw data and compute MSE
        err = 0
        for (i in 1:n_samples){
            for (j in 1:n_dim){
                weights1 = coefficients[i,j] * eigVecs[,'PC1',][,j]
                weighted_eigFun1 = fd(weights1, basisobj)
                weights2 = coefficients[i,j] * eigVecs[,'PC2',][,j]
                weighted_eigFun2 = fd(weights2, basisobj)
                weights3 = coefficients[i,j] * eigVecs[,'PC3',][,j]
                weighted_eigFun3 = fd(weights3, basisobj)
                tmpfd = weighted_eigFun1 + weighted_eigFun2 + weighted_eigFun3
                vals = eval.fd(seq(0, {n_frames-1}, len = {n_frames}), tmpfd)
                err = err + norm(as.matrix(data[,i,j] - vals))
            }
        }
        err = err/(n_samples * n_dim)
    ''' % (r_data.r_repr(), n_frames(), n_knots, n_harm)
    robjects.r(rcode)
    pca = robjects.globalenv['pcaVarmax']
#    scores = np.asarray(pca[pca.names.index('scores')])
    lowVs = robjects.globalenv['coefficients']
    err = robjects.globalenv['err']
    print 'number of eigenfunctions is: ' + str(n_harm)
    print 'MSE of fpca is: ' + str(err)
    return lowVs, pca


def centralizeDataBasedCurve(data):
    """
    centralize data use coefficents of expansion of basis function
    """
    n_samples, n_frames, dim = data.shape
#   reshape centralized_data as a 3d array: n_frame * n_samples * dim
    tmp = np.zeros((n_frames, n_samples, dim))
    for i in xrange(n_frames):
        for j in xrange(n_samples):
            tmp[i, j] = data[j, i, :]
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.array(tmp))  # convert numpy to r data
    print tmp.shape
    rcode = '''
        library(fda)
        data = %s
#        saveRDS(data, 'rootTrajectory.rds')
        n_samples = dim(data)[2]
        n_frames = dim(data)[1]
        n_dim = dim(data)[3]
        n_basis = 9
        basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        smoothed_tmp = smooth.basis(argvals = seq(0, n_frames-1,
                                                  len = n_frames), y = data,
                                                  fdParobj = basisobj)
        fd = smoothed_tmp$fd
        coefs = fd$coefs
        mean_coef = array(0, c(n_basis, n_dim))

        for (i in 1:n_samples){
            mean_coef = mean_coef + coefs[,i,]
        }
        mean_coef = mean_coef/n_samples
        # centralize coefs
        centralized_coefs = array(0, c(n_basis, n_samples, n_dim))
        for (i in 1:n_samples){
          centralized_coefs[,i,] = coefs[,i,] - mean_coef
        }
        mean_curve = array(0, c(n_frames, n_dim))
        for (i in 1:n_dim){
            fd = fd(mean_coef[,i], basisobj)
            samples = eval.fd(seq(0, n_frames - 1, len =n_frames), fd)
            mean_curve[,i] = samples
        }
        samples_mat = array(0, c(n_samples, n_frames, n_dim))
        for (i in 1:n_samples){
            for (j in 1:n_dim){
                fd = fd(centralized_coefs[,i,j], basisobj)
                samples = eval.fd(seq(0, n_frames - 1, len =n_frames), fd)
                samples_mat[i,,j] = samples
            }
        }

    ''' % (r_data.r_repr())
    robjects.r(rcode)    
    centralized_data = np.asarray(robjects.globalenv['samples_mat'])
    mean_curve = np.asarray(robjects.globalenv['mean_curve'])
    return mean_curve, centralized_data


def reconstructDataFromCurveSampling(mean_curve, centralized_data):
    '''
    Reconstruct oirginal data by add samples from each centralized curve and 
    mean curve
    
    Parameters:
    mean_curve: 2d array: n_frames * n_dim
    \tSamples from mean curve
    
    centralized_data: 3d array: n_samples * n_frames * n_dim
    \tSamples from each centralized curve
    '''
    centralized_data = np.asarray(centralized_data)
    temp = np.zeros(centralized_data.shape)
    n_samples, n_frames, n_dim = centralized_data.shape
    for i in xrange(n_samples):
        temp[i, :, :] = centralized_data[i, :, :] + mean_curve
    return temp

def MSE(raw_data, reconstructed_data):
    '''
    Compute the mean squared error bewteen original data and reconstructed 
    data
    '''
    diff = raw_data - reconstructed_data
    n_samples, n_frames, n_dim = diff.shape
    err = 0
    for i in xrange(n_samples):
        for j in xrange(n_frames):
            err += np.linalg.norm(diff[i, j])
    err = err/(n_samples * n_frames)
    return err
    
if __name__ == '__main__':
    data = get_root_joint_data()

    # data is a 3d array: n_samples * n_frames * n_dim
#    fig = plt.figure()
#    ax = p3.Axes3D(fig)
#    ax.scatter(data[:,0,0], data[:,0,2], data[:,0,1])
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_zlabel('Z')
#    ax.set_xlim(-1, 1)
#    ax.set_ylim(-1, 1)
#    ax.set_zlim(-1, 1)
#    plt.show()
#    plot_root_trajectory(data)
    mean_curve, centralized_data = centralizeDataBasedCurve(data)
#    plot_root_trajectory(mean_curve)
#    plot_root_trajectory(centralized_data)
#    centralized_data, mean_trajectory = centralizeData(data)
    reconstructed_data = reconstructDataFromCurveSampling(mean_curve,
                                                          centralized_data)
    plot_root_trajectory(reconstructed_data)
    err = MSE(data, reconstructed_data)
    print 'the MSE of reconstructed data is: ' + str(err)
#    plot_root_trajectory(mean_trajectory)
#    plot_root_trajectory(centralized_data)
#    eigenVectors = standardPCA(centralized_data)
#    err = evalStandardPCA(centralized_data, eigenVectors)
#    print 'MSE of standard PCA is: ' + str(err)
#    scores, pca = fpca(centralized_data)
