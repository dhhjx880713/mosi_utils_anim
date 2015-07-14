'''
Created on Jan 14, 2015

@author: hadu01
'''
import numpy as np
from lib.PCA import PCA, Center
import json
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import os

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
    data = temp[:, :, 0, :]
    print data.shape
    return data

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

def load_functional_data(RDataFile):
    """
    load functional data from pre-stored R data file

    Parameters
    ----------
    * RDataFile: string
    \tFile contains functional data

    Return
    ------
    * coefs: 3d numpy array
    \tCoefs contain the coefficients of functional data. The first dimension is
    the coefficients in time domain, the second dimension is the number of
    samples, the third dimension is the number of dimensions of each frame
    """
    rcode = """
        library(fda)
        fd = readRDS("%s")
        coefs = fd$coefs
    """ % (RDataFile)
    robjects.r(rcode)
    fd = robjects.globalenv['fd']
    coefs = robjects.globalenv['coefs']
    coefs = np.asarray(coefs)
#     coefs1 = fd[fd.names.index('coefs')]
#     coefs1 = np.asarray(coefs1)
#     print coefs.shape
#     print coefs1.shape
    rootCoefs = coefs[:,:100,:3]
    print rootCoefs.shape
    return rootCoefs

def reconstructRootTrajectoryFromCoefs(coefs):
    '''
    extract first three components from coefs matrix, reconstruct functions using coefficients, then sample function
    ''' 
    print coefs.shape
#     rootCoefs = coefs[:,:100,:3]
#     print rootCoefs.shape
    n_frames = 47
    n_basis, n_samples, n_dim = coefs.shape
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.asarray(coefs))
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
    
class PCA_fd(object):
    def __init__(self, input_data, n_basis = 7, fraction = 0.90, fpca = True):
        '''
        Parameters
        ----------
        * input_data: 3d array (n_samples * n_frames *n_dims)
        
        * faction: a value in [0, 1] 
        \tThe ratio of variance to maintain
        '''
        self.input_data = np.asarray(input_data)
        self.n_basis = n_basis
        self.is_fpca = fpca
        if self.is_fpca:
            self.fd = self.convert_to_fd(self.input_data, self.n_basis)
    #        print 'the shaple of functional data: '
    #        print self.fd.shape
    #        self.weight_root_joint()        
        
            self.reshaped_fd, origin_shape = self.reshape_fd(self.fd)
    #        print self.reshaped_fd.shape
            self.centerobj = Center(self.reshaped_fd)
            self.pcaobj = PCA(self.reshaped_fd, fraction = fraction)
            self.eigenvectors = self.pcaobj.Vt[:self.pcaobj.npc]
            print 'number of eigenvectors: ' + str(self.pcaobj.npc)
            self.lowVs = self.project_data(self.reshaped_fd)
            self.highVs = self.backproject_data(self.lowVs)
            self.fd_back = self.reshape_fd_back(self.highVs, origin_shape)
            
    #        self.unweight_root_joint()        
            # the shape of reconstructed_data should be the same as input data
            # n_samples * n_frames * n_dims
            self.reconstructed_data = self.from_fd_to_data(self.fd_back, 
                                                           self.n_frames)
                                                         
        else:   # apply standard PCA on input data
            self.data, origin_shape = self.reshape_data_for_PCA(self.input_data)
            self.centerobj = Center(self.data)
            self.pcaobj = PCA(self.data, fraction = fraction)
            self.eigenvectors = self.pcaobj.Vt[:self.pcaobj.npc]
            print 'number of eigenvectors: ' + str(self.pcaobj.npc)
            self.lowVs = self.project_data(self.data)
            # self.data should be a 2d array: n_samples * n_dimension
            self.highVs = self.backproject_data(self.lowVs)
            self.reconstructed_data = self.from_pca_to_data(self.highVs,
                                                            origin_shape)
     
    def reshape_data_for_PCA(self, input_data):
        """Reshape the input_data (n_samples * n_frames * n_dims) for standard
           PCA approach (n_samples * n_dimension)
        """
        assert len(input_data.shape) == 3, ('Input data matrix should be a \
                                            3d array')
        n_samples, n_frames, n_dims = input_data.shape
        reshaped_data = np.zeros((n_samples, n_frames * n_dims))
        for i in xrange(n_samples):
            reshaped_data[i, :] = np.reshape(input_data[i, :, :], 
                                            (1, n_frames * n_dims))        
        return reshaped_data, (n_samples, n_frames, n_dims)     

    def from_pca_to_data(self, data, original_shape):                                                          
        """Reshape back projection result from PCA as input data
        """
        reconstructed_data = np.zeros(original_shape)
        n_samples, n_frames, n_dims = original_shape
        for i in xrange(n_samples):
            reconstructed_data[i, :, :] = np.reshape(data[i, :], (n_frames, n_dims))
        return reconstructed_data        
    
    def weight_root_joint(self, weight=10):
        """Give a higher weight for coefficients of root joints
        """
        self.fd[:,:,3] = self.fd[:,:,3] * weight 
    
    def unweight_root_joint(self, weight=10):
        self.fd_back[:,:,3] = self.fd_back[:,:,3]/weight
    
    def convert_to_fd(self, input_data, n_basis):
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
        input_data = np.asarray(input_data)
        assert len(input_data.shape) == 3, ('input data should be a 3d array')
        self.n_samples, self.n_frames, self.n_dim = input_data.shape
        # reshape the data matrix for R library fda
        tmp = np.zeros((self.n_frames, self.n_samples, self.n_dim))
        for i in xrange(self.n_frames):
            for j in xrange(self.n_samples):
                tmp[i, j] = input_data[j, i, :]
        robjects.conversion.py2ri = numpy2ri.numpy2ri
        r_data = robjects.Matrix(np.asarray(tmp))
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
        ''' % (r_data.r_repr(), n_basis)
        robjects.r(rcode)
        fd = robjects.globalenv['fd']
        coefs = fd[fd.names.index('coefs')]
        coefs = np.asarray(coefs)   
        print coefs.shape
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
            pca_data[i] = np.reshape(fd[:,i,:], (1, n_coefs * n_dim))
        return pca_data, (n_coefs, n_samples, n_dim)
    
    def reshape_fd_back(self, pca_data, origin_shape):
        assert len(pca_data.shape) == 2, ('the data should be a 2d array')
        assert len(origin_shape) == 3, ('the original data should be a 3d array')
        n_samples, n_dim = pca_data.shape
        fd_back = np.zeros(origin_shape)
        for i in xrange(n_samples):
            fd_back[:,i,:] = np.reshape(pca_data[i], (origin_shape[0], origin_shape[2]))
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

def functional_data_convertion(data):
    '''
    represent data by the coefficients of functions to fit the data
    The data is passed to R library fda, using b-spline basis to fit the data.
    Parameters
    ----------
    * data : 3d array
    \tThe data matrix should be a 3d matrix, the first dimension is n_samples, the second dimension is n_frames, the third one is n_dim
    '''
    # the data matrix should be reshape as n_frame * n_samples * n_dim
    n_samples, n_frames, dim = data.shape
    tmp = np.zeros((n_frames, n_samples, dim))
    for i in xrange(n_frames):
        for j in xrange(n_samples):
            tmp[i, j] = data[j, i, :]
    robjects.conversion.py2ri = numpy2ri.numpy2ri
    r_data = robjects.Matrix(np.asarray(tmp))
    n_basis = 5
    rcode = '''
        library('fda')
        data = %s
        n_basis = %d
        n_samples = dim(data)[2]
        n_frames = dim(data)[1]
        n_dim = dim(data)[3]
#         print(n_samples)
#         print(n_frames)
#         print(n_dim)
        basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        smoothed_tmp = smooth.basis(argvals=seq(0, {n_frames-1}, len = {n_frames}),
                           y = {data}, fdParobj = basisobj)
        fd = smoothed_tmp$fd                                                  
    ''' % (r_data.r_repr(), n_basis)
    robjects.r(rcode)
    fd = robjects.globalenv['fd']
    coefs = fd[fd.names.index('coefs')]
    coefs = np.asarray(coefs)
    print coefs.shape
    return coefs

def reshape_data_for_PCA(data):
    '''
    Reshape data for standard PCA
    '''
    data = np.asarray(data)
    assert len(data.shape) == 3, ('Data matrix should be a 3d array')
    n_samples, n_frames, n_dims = data.shape
    reshaped_data = np.zeros((n_samples, n_frames * n_dims))
    for i in xrange(n_samples):
        reshaped_data[i, :] = np.reshape(data[i, :, :], (1, n_frames * n_dims))
    return reshaped_data, (n_samples, n_frames, n_dims)
    
def standardPCA(input_data, fraction = 0.90):
    '''
    Apply standard PCA on motion data
    
    Parameters
    ----------
    * data: 2d array
    \tThe data matrix contains motion data, which should be a matrix with 
    shape n_sample * n_dims
    '''
    input_data = np.asarray(input_data)
    assert len(input_data.shape) == 2, ('Data matrix should be a 2d array')
    n_samples, n_dims = input_data.shape
    centerobj = Center(input_data)
#    for i in xrange(n_samples):
#        tmp = input_data[i,:50]
#        zeroindexes = tmp < 0.0001
#        tmp[zeroindexes] = 0
    pcaobj = PCA(input_data, fraction = fraction)
    print 'number of principal for standard PCA is:' + str(pcaobj.npc)
    eigenvectors = pcaobj.Vt[:pcaobj.npc]
    lowVs = []
    for i in xrange(n_samples):
        projected_data = np.dot(eigenvectors, input_data[i])
        lowVs.append(projected_data)
    lowVs = np.asarray(lowVs)
    highVs = []
    for i in xrange(n_samples):
        backprojected_data = np.dot(np.transpose(eigenvectors), lowVs[i].T)
        backprojected_data = backprojected_data + centerobj.mean
        highVs.append(backprojected_data)
    highVs = np.asarray(highVs)
    return pcaobj, highVs

def backprojection_to_motion_data(data, original_shape):
    '''
    Reshape backprojection data from PCA to motion data
    '''
    reconstructed_data = np.zeros(original_shape)
    n_samples, n_frames, n_dims = original_shape
    for i in xrange(n_samples):
        reconstructed_data[i, :, :] = np.reshape(data[i, :], (n_frames, n_dims))
    return reconstructed_data
        

def test():
    data = get_root_joint_data() # 3d array n_samples*n_frames*n_dim
#     plot_root_trajectory(data)
#   apply standard PCA
    reshaped_data, original_shape = reshape_data_for_PCA(data)
    pcaobj, backprojection = standardPCA(reshaped_data, fraction = 0.95)
    reconstructed_data = backprojection_to_motion_data(backprojection, original_shape)
    err = MSE(data, reconstructed_data)
    print 'MSE for standard PCA is: ' + str(err)
#    coefs = functional_data_convertion(data)
#
##    functionalData = load_functional_data('walk_leftStance_functionalData.RData')
#
#    reconstructed_data = reconstructRootTrajectoryFromCoefs(coefs)
#     plot_root_trajectory(reconstructed_data)

#    err = MSE(data, reconstructed_data)
#    print 'MSE is' + str(err)
    # apply PCA on functional data
    fpca = PCA_fd(data, fraction = 0.95)
    reconstructed_data = fpca.reconstructed_data
#    fd_back = fpca.fd_back
#    print fd_back.shape
#    reconstructed_data = reconstructRootTrajectoryFromCoefs(fd_back)
    err = MSE(data, reconstructed_data)
#    plot_root_trajectory(reconstructed_data)
    print 'MSE is' + str(err)
#     testobj = PCAonCoefs(fd)
if __name__ == '__main__':
    test()
        