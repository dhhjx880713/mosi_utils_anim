# -*- coding: utf-8 -*-
"""
Created on Fri Jan 09 08:53:00 2015

@author: mamauer
"""
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import os
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from PCA import PCA, Center

ROOT_DIR = os.sep.join([".."] * 7)

def get_input_data_folder():
    """
    Return folder path of feature data without trailing os.sep
    """
    data_dir_name = "src"   # Use the src folder, since the R Code is not cross path yet.
                            # TODO: Fix this.
    PCA_dir_name = "2 - PCA"
    type_parameter = "spatial"
    step = "3 - fpca"
    action = 'experiment'   # Inconsistent naming here...
    test_feature = "1 - FPCA with absolute joint positions in Cartesian space"
    subtest = "1.3 fpca with pca on scores"
    input_dir = os.sep.join([ROOT_DIR,
                             data_dir_name,
                             PCA_dir_name,
                             type_parameter,
                             step,
                             action,
                             test_feature,
                             subtest])
    return input_dir
    
    
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
    
    
def load_r_data(path):
    """ Load a RData file 
        
    Parameters
    ----------
    * path: string
    \tThe Path to the RData file
    
    Return
    ------
    rpy2.robjects.vectors.ListVector
    """
    readRDS = robjects.r['readRDS']
    retvalue = readRDS(path)
    return retvalue
    

def get_coefs_from_fd(fd, as_rpy=False):
    """ Return the coefs and the shape of a given fd
    
    Parameters
    ----------
    * fd: rpy2.robjects.vectors.ListVector
    \tThe fd object
    
    * as_ndarray: boolean
    \tControlls the type of the returned parameters. 
    If True, all results are returned as RPy Objects. Otherwise, all are
    transformed to numpy.ndarray.
    
    Returns
    -------
    * coefs: rpy2.robjects.vectors.Array or numpy.ndarray
    \tThe coeficients of the fd object. The type is controlled by the parameter
    "as_rpy"
    
    * shape: rpy2.robjects.vectors.IntVector or numpy.ndarray
    \tThe shape of the coeficients. The type is controlled by the parameter
    "as_rpy"
    """    
    if not isinstance(fd, robjects.vectors.ListVector):
        raise ValueError("fd must be a ListVector")

    try:
        fd.names.index('coefs')
    except ValueError:
        raise ValueError("fd doesn't have coefs. Probably not a RPy fd-object")
                         
    coefs = fd[fd.names.index('coefs')]
    shape = robjects.r['dim'](fd[fd.names.index('coefs')])

    if as_rpy:
        return coefs, shape
        
    return np.array(coefs), np.array(shape)


def pca(data, fraction=0.9, center=True, verbose=False):
    """ Perform standard PCA on data
    
    Parameters
    ----------
    * data: numpy.ndarray
    \tThe data 
    
    * fraction: double in range [0, 1]
    \tThe fraction to determine the number of PCs
    
    * center: boolean
    \tCenter the data or not
    
    * verbose: boolean
    \tShow PCA result

    Return
    ------
    * pcaobj: PCA
    \tThe result of the standard PCA
    
    * lowdim: numpy.ndarray
    \tThe lowdimensional representation of the data
    
    * centerobj: Center
    \tThe Center of the data. Only returned when center == True
    """
    if center:
        centerobj = Center(data)

    pcaobj = PCA(data, fraction=fraction)
    eigenvectors = pcaobj.Vt[:pcaobj.npc]
    
    if verbose:
        print "Number of PCs: %d (fraction: %f)" % (pca.npc, fraction)
        plt.plot(pca.sumvariance)
        plt.xlim((0, 50))
        plt.ylim((0.4, 1))
        plt.xlabel('Number PCs')
        plt.ylabel('Sumvariance')
        plt.show()
        
    lowdim = []
    for i in xrange(len(data)):
        lowdim_i = np.dot(eigenvectors, data[i])
        lowdim.append(lowdim_i)
    lowdim = np.array(lowdim)
    
    if center:
        return pcaobj, lowdim, centerobj
    return pcaobj, lowdim

    
def normal_pca_on_coefs(fd, fraction=0.9, verbose=False):
    """ Perform a functional PCA with R, take the scores and perform a standard
    PCA to further reduce the dimensions
    
    Parameters
    ----------
    * fd: rpy2.robjects.vectors.ListVector
    \tThe fd Object as RPy representation
    * fraction: double in range [0, 1]
    \tThe fraction to determine the number of PCs
    * verbose: boolean
    \tShow PCA result

    Return
    ------
    * pcaobj: PCA
    \tThe result of the standard PCA
    
    * center: Center
    \tThe Center of the stacked scores
    
    * lowdim: numpy.ndarray
    \tThe lowdimensional representation of the data
    """                    
    coefs, shape = get_coefs_from_fd(fd, as_rpy=False)

    assert len(shape) == 3, ("Shape of Coeffizients is not three. Probably only"
                            " one sample. Multiple samples are needed.")
    
    n_basis = shape[0] 
    n_samples = shape[1] 
    n_dims = shape[2] 
    
    stacked_coefs = np.zeros(shape=(n_samples, n_basis*n_dims))
    
    for i in xrange(n_samples):
        stacked_coefs[i] = np.ravel(coefs[:,i,:])
        
    pcaobj, lowdim, center = pca(stacked_coefs, fraction=fraction, 
                                 center=True, verbose=verbose)
    
    original_shape = (n_basis, n_dims)
    return pcaobj, center, lowdim, original_shape
    

def pca_backprojection(pca_result, sampleid, basis=None):
    """ Function to demonstrate the backprojection for a specific sample
    
    Parameters
    ----------
    * pca_result: tuple
    \tThe result tuple as return from normal_pca_on_coefs
    
    * sampleid: int
    \tThe ID of the testset in the original data

    * basis: rpy2.robjects.vectors.ListVector
    \tThe basis to create a fd object. If this is None, the coefs are returned
    
    Returns
    -------
    * newfd: rpy2.robjects.vectors.ListVector or numpy.ndarray
    \tThe function ether returns a numpy.ndarray containing the coeficients of
    the function or a rpy2 ListVector representing the function in the 
    R-Library "fda"
    
    """
    pcaobj, center, lowdim, original_shape = pca_result
    
    eigenvectors = pcaobj.Vt[:pcaobj.npc]

    back = np.dot(np.transpose(eigenvectors), lowdim[sampleid].T)
    back += center.mean    

    new_coefs = np.reshape(back, original_shape)

    if basis is None:  
        return new_coefs
    
    new_coefs_r = numpy2ri.numpy2ri(new_coefs)    
    robjects.r('library("fda")')
    fd = robjects.r['fd']  
    newfd = fd(new_coefs_r, basis)
    return newfd

    
def compare_with_original(data, newfd, sampleid, is_new_evaluated=False, 
                          plot=False):
    """ Plot the channel for the original data and the backprojected data 
    
    Parameters
    ----------        
    * data: rpy2.robjects.vectors.ListVector
    \tThe frame data
    
    * newfd: rpy2.robjects.vectors.ListVector or numpy.ndarray
    \tThe R representation of the new fd object or the sampled frames
    See is_new_evaluated
        
    * channel: int
    \tThe channel in the Framedata, e.g. 0 for x-Channel of rootjoint
    
    * is_new_evaluated: boolean
    \tIf this is True, the newfd is given as numpy array. If this is False,
    newfd is represented as fd R object
    
    * plot: boolean
    \tIf this is True, the root joint is plotted
    
    Returns
    -------
    A list of errors for each dimension
    """    
    # get original fd
    root_coefs = data[data.names.index('coefs')]
    root_coefs_np = np.array(root_coefs)
    original_sample_np = root_coefs_np[:, sampleid, :]    
    original_sample = numpy2ri.numpy2ri(original_sample_np)
    
    basis = data[data.names.index('basis')]    
    fd = robjects.r['fd']
    
    origianlfd = fd(original_sample, basis)
        
    # evaluate and plot
    fdeval = robjects.r['eval.fd']
    t = robjects.r('seq(0,46)')
    
    if not is_new_evaluated:
        newframes = np.array(fdeval(t, newfd))
    else:
        newframes = np.array(newfd)
    originalframes = np.array(fdeval(t, origianlfd))

    if plot:
        channel = 0
        plt.plot(newframes[:, channel], label='new')
        plt.plot(originalframes[:, channel], label='original')
        plt.title('Dimension %d, Sample %d' % (channel, sampleid))
        plt.legend()
    
        plt.xlabel('t')
        plt.ylabel('Value of Dimension in cm')    
        
        plt.show()  
    
    # calculate maximal error
    errors = []
    for c in xrange(len(newframes[0])):
        errors.append([])
        dif = newframes[:, c] - originalframes[:, c]
        dif = np.abs(dif)
        for dif_i in dif:
            errors[c].append(dif_i)
            
    return errors
#
#    channel = [0, 1, 2]
#    
#    plt.plot(newframes[:, channel], label='new')
#    plt.plot(originalframes[:, channel], label='original')
#    plt.legend()
#    plt.show()    
#    
    
def calculate_errors(data, fraction, num_samples=620):
    """ Performs an experiment as defined in the approach document

    Parameters
    ----------
    * data: rpy2.robjects.vectors.ListVector
    \tThe frame data
    
    * fraction: double in range [0, 1]
    \tThe fraction to determine the number of PCs
    
    * num_samples: int, optional
    \tThe number of samples. This surely can be extracted from the data set.
    This has to be done.
    
    Returns
    -------
    None
    """
    result = normal_pca_on_coefs(data, fraction=fraction)
    basis = data[data.names.index('basis')]
    
    errors = None
    
    # Set this to a number if you want to plot a specific sample
    f = -1

    for n in xrange(num_samples):
        newfd = pca_backprojection(result, n, basis)
        
        if n == f:
            error_n =  compare_with_original(data, newfd, n, plot=True)

        error_n =  compare_with_original(data, newfd, n)
        
        if errors is None:
            errors = error_n
        else:
            for c in xrange(len(errors)):
                for e in error_n[c]:
                    errors[c].append(e)
                    
    npc = result[0].npc
    max_err = np.max(errors, axis=1)    
    mean = np.mean(errors, axis=1)
    std = np.std(errors, axis=1)
    
    print "Number PCs:", npc
    
    print "Max Errors per dimension:"    
    print max_err
    print "========================="
    
    print "Mean per dimension:"   
    print mean
    print "========================="

    print "Standard deviation per dimension:"   
    print std
    print "========================="
    

def main():
    """
    Main function to show the purpose of this module
    """
    
    path = clean_path(get_input_data_folder())
    datapath = os.sep.join((path, 'data', 'walk_leftStance_fd.RData'))
    data = load_r_data(datapath)
    basis = data[data.names.index('basis')]
    
    calculate_errors(data, fraction=0.99)

    result = normal_pca_on_coefs(data, fraction=0.95)
#
#    sampleid = 10
#    newfd = pca_backprojection(result, sampleid=sampleid, basis=basis)
#    compare_with_original(data, newfd, sampleid)
  
  
if __name__=='__main__':
    main()