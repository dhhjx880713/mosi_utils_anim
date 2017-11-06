'''
Created on Jan 14, 2015

@author: hadu01
'''
import numpy as np
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
from ...external.PCA import Center, PCA
import scipy.interpolate as si
from .functional_data import FunctionalData


class PCAFunctionalData(object):

    def __init__(self, input_data, n_basis=7, fraction=0.90, n_pc=None):
        '''
        Parameters
        ----------
        * input_data: 3d array (n_samples * n_frames *n_dims)

        * faction: a value in [0, 1]
        \tThe ratio of variance to maintain
        '''
        self.input_data = np.asarray(input_data)
        self.n_basis = n_basis
        print(("input data shape: ", self.input_data.shape))
        print("start to convert to functional data")
        self.functional_data = self.convert_to_fd()
        print("functional convertion completed!")
        self.reshaped_fd, self.origin_shape = PCAFunctionalData.reshape_fd(self.functional_data)
        self.centerobj = Center(self.reshaped_fd)
        self.pcaobj = PCA(self.reshaped_fd, fraction=fraction)
        if n_pc is None:
            self.eigenvectors = self.pcaobj.Vt[:self.pcaobj.npc]
            print('number of eigenvectors: ' + str(self.pcaobj.npc))
        else:
            self.eigenvectors = self.pcaobj.Vt[:n_pc]
            print(('number of eigenvectors: ' + str(n_pc)))
        self.low_vecs = self.project_data(self.reshaped_fd)

    @classmethod
    def from_pca_to_data(cls, data, original_shape):
        """Reshape back projection result from PCA as input data
        """
        reconstructed_data = np.zeros(original_shape)
        n_samples, n_frames, n_dims = original_shape
        for i in range(n_samples):
            reconstructed_data[i, :, :] = np.reshape(
                data[i, :], (n_frames, n_dims))
        return reconstructed_data

    def convert_to_fd(self):
        '''
        represent data as a linear combination of basis function, and return
        weights of functions

        Parameters
        ----------
        * input_data: 3d array ( n_samples * n_frames *n_dim)

        Return
        ------
        * coefs: 3d array (n_samples * n_coeffs * n_dim)
        '''
        assert len(
            self.input_data.shape) == 3, ('input data should be a 3d array')
        functional_data = FunctionalData()
        coeffs = functional_data.convert_motions_to_functional_data(self.input_data,
                                                                    self.n_basis)
        return np.asarray(coeffs)

    # def convert_to_fd(self):
    #     '''
    #     represent data as a linear combination of basis function, and return
    #     weights of functions
    #
    #     Parameters
    #     ----------
    #     * input_data: 3d array ( n_frames * n_samples *n_dim)
    #
    #     Return
    #     ------
    #     * coefs: 3d array (n_coefs * n_samples * n_dim)
    #     '''
    #     assert len(
    #         self.input_data.shape) == 3, ('input data should be a 3d array')
    #     # reshape the data matrix for R library fda
    #     robjects.conversion.py2ri = numpy2ri.numpy2ri
    #     r_data = robjects.Matrix(self.input_data)
    #     rcode = '''
    #         library('fda')
    #         data = %s
    #         n_basis = %d
    #         n_samples = dim(data)[2]
    #         n_frames = dim(data)[1]
    #         n_dim = dim(data)[3]
    #         basisobj = create.bspline.basis(c(0, n_frames - 1),
    #                                         nbasis = n_basis)
    #         smoothed_tmp = smooth.basis(argvals=seq(0, {n_frames-1},
    #                         len = {n_frames}),y = {data}, fdParobj = basisobj)
    #         fd = smoothed_tmp$fd
    #     ''' % (r_data.r_repr(), self.n_basis)
    #     robjects.r(rcode)
    #     functional_data = robjects.globalenv['fd']
    #     coefs = functional_data[functional_data.names.index('coefs')]
    #     coefs = np.asarray(coefs)
    #     return coefs

    @classmethod
    def reshape_fd(cls, functional_data):
        '''
        reshape 3d coefficients (n_samples *n_coeffs *  n_dim) as
        a 2d matrix (n_samples * (n_coefs * n_dim)) for standard PCA
        '''
        assert len(functional_data.shape) == 3, ("functional data should be a 3d array")
        n_samples, n_coefs, n_dim = functional_data.shape
        pca_data = np.zeros((n_samples, n_coefs * n_dim))
        for i in range(n_samples):
            pca_data[i] = np.reshape(functional_data[i, :, :], (1, n_coefs * n_dim))
        return pca_data, (n_samples, n_coefs, n_dim)

    @classmethod
    def reshape_fd_back(cls, pca_data, origin_shape):
        assert len(pca_data.shape) == 2, ('the data should be a 2d array')
        assert len(
            origin_shape) == 3, ('the original data should be a 3d array')
        fd_back = np.zeros(origin_shape)
        for i in range(len(pca_data)):
            fd_back[:, i, :] = np.reshape(
                pca_data[i], (origin_shape[0], origin_shape[2]))
        return fd_back

    def project_data(self, data):
        '''
        project functional data to low dimensional space
        '''
        low_vecs = []
        for i in range(len(data)):
            low_vec = np.dot(self.eigenvectors, data[i])
            low_vecs.append(low_vec)
        low_vecs = np.asarray(low_vecs)
        return low_vecs

    def backproject_data(self, low_vecs):
        n_samples = len(low_vecs)
        high_vecs = []
        for i in range(n_samples):
            high_vec = np.dot(np.transpose(self.eigenvectors), low_vecs[i].T)
            high_vec = high_vec + self.centerobj.mean
            high_vecs.append(high_vec)
        high_vecs = np.asarray(high_vecs)
        return high_vecs

    @classmethod
    def from_fd_to_data(cls, functional_data, n_frames):
        '''
        generate data from weights of basis functions

        Parameter
        ---------
        * fd: 3d array (n_weights * n_samples * n_dim)
        \tThe weights of basis functions
        '''
        assert len(functional_data.shape) == 3, ('weights matrix should be a 3d array')
        robjects.conversion.py2ri = numpy2ri.numpy2ri
        r_data = robjects.Matrix(np.asarray(functional_data))
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
