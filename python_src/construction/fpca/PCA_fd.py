'''
Created on Jan 14, 2015

@author: hadu01
'''
import numpy as np
import os
import sys
ROOT_DIR = os.sep.join(['..'] * 2)
sys.path.append(ROOT_DIR)
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
from external.PCA import Center, PCA


class PCA_fd(object):

    def __init__(self, input_data, n_basis=7, fraction=0.90):
        '''
        Parameters
        ----------
        * input_data: 3d array (n_frames * n_samples *n_dims)

        * faction: a value in [0, 1] 
        \tThe ratio of variance to maintain
        '''
        self.input_data = np.asarray(input_data)
        self.n_basis = n_basis
        self.fd = self.convert_to_fd()
        self.reshaped_fd, origin_shape = self.reshape_fd(self.fd)
        self.centerobj = Center(self.reshaped_fd)
        self.pcaobj = PCA(self.reshaped_fd, fraction=fraction)
        self.eigenvectors = self.pcaobj.Vt[:self.pcaobj.npc]
        print 'number of eigenvectors: ' + str(self.pcaobj.npc)
        self.lowVs = self.project_data(self.reshaped_fd)

    def from_pca_to_data(self, data, original_shape):
        """Reshape back projection result from PCA as input data
        """
        reconstructed_data = np.zeros(original_shape)
        n_samples, n_frames, n_dims = original_shape
        for i in xrange(n_samples):
            reconstructed_data[i, :, :] = np.reshape(
                data[i, :], (n_frames, n_dims))
        return reconstructed_data

    def convert_to_fd(self):
        '''
        represent data as a linear combination of basis function, and return 
        weights of functions

        Parameters
        ----------
        * input_data: 3d array (n_samples * n_frames *n_dim)

        Return
        ------
        * coefs: 3d array (n_coefs * n_samples * n_dim)
        '''
        assert len(self.input_data.shape) == 3, ('input data should be a 3d array')
        # reshape the data matrix for R library fda
        robjects.conversion.py2ri = numpy2ri.numpy2ri
        r_data = robjects.Matrix(self.input_data)
        rcode = '''
            library('fda')
            data = %s
            n_basis = %d
            n_samples = dim(data)[2]
            n_frames = dim(data)[1]
            n_dim = dim(data)[3]
            basisobj = create.bspline.basis(c(0, n_frames - 1),
                                            nbasis = n_basis)
            smoothed_tmp = smooth.basis(argvals=seq(0, {n_frames-1},
                            len = {n_frames}),y = {data}, fdParobj = basisobj)
            fd = smoothed_tmp$fd                                                  
        ''' % (r_data.r_repr(), self.n_basis)
        robjects.r(rcode)
        fd = robjects.globalenv['fd']
        coefs = fd[fd.names.index('coefs')]
        coefs = np.asarray(coefs)
        return coefs

    def reshape_fd(self, fd):
        '''
        reshape 3d coefficients (n_coefs * n_samples * n_dim) as 
        a 2d matrix (n_samples * (n_coefs * n_dim)) for standard PCA
        '''
        assert len(fd.shape) == 3, ("fd should be a 3d array")
        n_coefs, n_samples, n_dim = fd.shape
        pca_data = np.zeros((n_samples, n_coefs * n_dim))
        for i in xrange(n_samples):
            pca_data[i] = np.reshape(fd[:, i, :], (1, n_coefs * n_dim))
        return pca_data, (n_coefs, n_samples, n_dim)

    def reshape_fd_back(self, pca_data, origin_shape):
        assert len(pca_data.shape) == 2, ('the data should be a 2d array')
        assert len(
            origin_shape) == 3, ('the original data should be a 3d array')
        n_samples, n_dim = pca_data.shape
        fd_back = np.zeros(origin_shape)
        for i in xrange(n_samples):
            fd_back[:, i, :] = np.reshape(
                pca_data[i], (origin_shape[0], origin_shape[2]))
        return fd_back

    def project_data(self, data):
        '''
        project functional data to low dimensional space
        '''
        lowVs = []
        n_samples, n_dim = data.shape
        for i in xrange(n_samples):
            lowV = np.dot(self.eigenvectors, data[i])
            lowVs.append(lowV)
        lowVs = np.asarray(lowVs)
        return lowVs

    def backproject_data(self, lowVs):
        n_samples = len(lowVs)
        highVs = []
        for i in xrange(n_samples):
            highV = np.dot(np.transpose(self.eigenvectors), lowVs[i].T)
            highV = highV + self.centerobj.mean
            highVs.append(highV)
        highVs = np.asarray(highVs)
        return highVs

    def from_fd_to_data(self, fd, n_frames):
        '''
        generate data from weights of basis functions

        Parameter
        ---------
        * fd: 3d array (n_weights * n_samples * n_dim)
        \tThe weights of basis functions
        '''
        assert len(fd.shape) == 3, ('weights matrix should be a 3d array')
#        n_basis, n_samples, n_dim = coefs.shape
        robjects.conversion.py2ri = numpy2ri.numpy2ri
        r_data = robjects.Matrix(np.asarray(fd))
        rcode = '''
            library('fda')
            data = %s
            n_frames = %d
            n_basis = dim(data)[1]
            n_samples = dim(data)[2]
            n_dim = dim(data)[3]
            basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
            samples_mat = array(0, c(n_samples, n_frames, n_dim))
            for (i in 1:n_samples){
                for (j in 1:n_dim){
                    fd = fd(data[,i,j], basisobj)
                    samples = eval.fd(seq(0, n_frames -1, len = n_frames), fd)
                    samples_mat[i,,j] = samples
                }
            }
        ''' % (r_data.r_repr(), n_frames)
        robjects.r(rcode)
        reconstructed_data = np.asarray(robjects.globalenv['samples_mat'])
        return reconstructed_data
