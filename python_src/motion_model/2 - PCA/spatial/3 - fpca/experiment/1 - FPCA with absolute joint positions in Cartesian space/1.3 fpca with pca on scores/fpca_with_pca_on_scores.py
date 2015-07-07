# -*- coding: utf-8 -*-
"""
Created on Do Jan 08 10:06:02 2015

@author: mamauer, hadu01
"""
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import os
import glob
import numpy as np
import matplotlib.pylab as plt
import sys
ROOT_DIR = os.sep.join([".."] * 7)
LIB_PATH = os.sep.join(["..", 
                        "1.1 normal PCA on concatenated functional parameters",
                        "lib"])
sys.path.insert(1,LIB_PATH)
from PCA import PCA, Center

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
    
def fpca_with_pca_on_scores(rdata, nharm=12):
    """
    Perform a functional PCA with R, take the scores and perform a standard
    PCA to further reduce the dimensions
    
    Parameters
    ----------
    * rdata: rpy2.robjects.vectors.ListVector
    \tThe frame data
    * nharm: int
    \tThe number of harmonics during the fPCA

    Return
    ------
    * fpcavarmax: rpy2.robjects.vectors.ListVector
    \tThe fpca result from R
    
    * pcaobj: PCA
    \tThe result of the standard PCA
    
    * center: Center
    \tThe Center of the stacked scores
    
    * lowdim: numpy.ndarray
    \tThe lowdimensional representation of the data
    """

    # check data first:
    if not isinstance(rdata, robjects.vectors.ListVector):
        raise ValueError("rdata must be a ListVector as "
                         "returned from load_r_data")
                         
    robjects.r("library('fda')")    #load fda library

    fpcaobj = robjects.r['pca.fd'](rdata, nharm)
    fpcavarmax = robjects.r['varmx.pca.fd'](fpcaobj)
    scores = fpcavarmax[fpcavarmax.names.index('scores')]
    n_samples = robjects.r['dim'](rdata[rdata.names.index('coefs')])[1]
    
    # save scores stacked in a numpy.ndarray
    np_scores = np.array(scores)
    stacked_scores = []
    
    for i in xrange(n_samples):
        scores_i = np.ravel(np_scores[i,:,:])
        stacked_scores.append(scores_i)

    stacked_scores = np.array(stacked_scores)        
    
    # Perform standard PCA to stacked scores
    center = Center(stacked_scores)
    pcaobj = PCA(stacked_scores, fraction=0.9999)
    eigenvectors = pcaobj.Vt[:pcaobj.npc]
    
    lowdim = []
    for i in xrange(len(stacked_scores)):
        lowdim_i = np.dot(eigenvectors, stacked_scores[i])
        lowdim.append(lowdim_i)
    lowdim = np.array(lowdim)
    
    return fpcavarmax, pcaobj, center, lowdim


def _internal_backprojection_based_on_coefs(fpca, scores_i_r):
    # BACKPROJECTION BASED ON 
    # fpca backprojection in R:
    harmonic_coefs = robjects.r['coef'](fpca[fpca.names.index('harmonics')])
    mean_coefs = robjects.r['coef'](fpca[fpca.names.index('meanfd')])
    
    nbasis = robjects.r['dim'](harmonic_coefs)[0]
    ndim = robjects.r['dim'](harmonic_coefs)[2]
    
    new_coefs = mean_coefs

    robjects.r('samplescore <- %s' % scores_i_r.r_repr())
    robjects.r('harmonic_coefs <- %s' % harmonic_coefs.r_repr())
    robjects.r('mean_coefs <- %s' % mean_coefs.r_repr())
    robjects.r('new_coefs <- mean_coefs[,1,]')
    
    rcode = '''
        scores_i = (harmonic_coefs[{i},,{dim}] * samplescore[,{dim}]) 
        new_coefs[{i},{dim}] = sum(scores_i) + mean_coefs[{i},1,{dim}]
    '''
        
    # Start with 1 because the code is actually evaluated in R
    for i in xrange(1, nbasis+1):
        for dim in xrange(1, ndim+1):
            robjects.r(rcode.format(i=i, dim=dim))
    new_coefs = robjects.r['new_coefs']

    fd = robjects.r['fd']
    harmonics = fpca[fpca.names.index('harmonics')]
    basis = harmonics[harmonics.names.index('basis')]    
    
    newfd = fd(new_coefs, basis)  
    
    # plot mean
    meanfd = fpca[fpca.names.index('meanfd')]
    fdeval = robjects.r['eval.fd']
    t = robjects.r('seq(0,46)')
    mean_frames = np.array(fdeval(t, meanfd))
    plt.plot(mean_frames[:, 0, 0], label='mean')
    
    return newfd    


def _internal_backprojection_based_on_eval(fpca, scores_i):
    eigenfd = fpca[fpca.names.index('harmonics')]
    meanfd = fpca[fpca.names.index('meanfd')]
    fdeval = robjects.r['eval.fd']
    frames = []
    means = []
    for i in xrange(47):
        print i
        mean_i = fdeval(i, meanfd)
        mean_i = np.ravel(np.asarray(mean_i))
        means.append(mean_i)
        eigen_i = np.asarray(fdeval(i, eigenfd))[0]     # its a nested array

        frame = np.zeros(shape=(eigen_i.shape[-1]))
        for j in xrange(eigen_i.shape[0]):
            frame += eigen_i[j] * scores_i[j]
            print eigen_i[j][0]
            print scores_i[j][0]
            assert eigen_i[j][0] * scores_i[j][0] == (eigen_i[j]*scores_i[j])[0]
            print "==="
        print frame[0]
        print mean_i[0]

        frame = frame + mean_i
        frames.append(frame)
    frames = np.array(frames)

    means = np.array(means)
    plt.plot(means[:,0], label='mean')    
    
    return frames


def fpca_backprojection(result, sampleid=-1):
    """ backprojection example for the result
    
    Parameters
    ----------        
    * result: tuple
    \tThe result of the fpca_with_pca_on_scores function. The tuple consists of
    (fpcavarmax, pcaobj, Center, lowdim)
    * sampleid: int
    \tThe index of the data to be backprojected in the lowdim array
    
    Returns
    -------
    * newfd: rpy2.robjects.vectors.ListVector
    \tA fd Object representing the new backprojected function
    """
    assert len(result)==4
    fpca, pcaobj, center, lowdim = result

    scoredim = robjects.r['dim'](fpca[fpca.names.index('scores')])
    # save scores stacked in a numpy.ndarray
    
    eigenvectors = pcaobj.Vt[:pcaobj.npc]    
    
    back = np.dot(np.transpose(eigenvectors), lowdim[sampleid].T)
    back += center.mean
    
    
    scores = fpca[fpca.names.index('scores')]
    np_scores = np.array(scores)
    
    scores_i = np.reshape(back, newshape=(scoredim[1],scoredim[2]))

    scores_i = np_scores[sampleid, :, :]

    scores_i_r = numpy2ri.numpy2ri(scores_i)

#    scores = fpca[fpca.names.index('scores')]
#    np_scores = np.array(scores) 
#    diff = np_scores[sampleid] - scores_i
#    print np_scores[sampleid].shape
#    print scores_i.shape
#    print diff
    #newfd = _internal_backprojection_based_on_coefs(fpca, scores_i_r)
    newfd = _internal_backprojection_based_on_eval(fpca, scores_i)

    return newfd

def compare_with_original(data, newfd, sampleid, 
                          channel=0, is_new_evaluated=True):
    """ Plot the channel for the original data and the backprojected data 
    
    Parameters
    ----------        
    * data: rpy2.robjects.vectors.ListVector
    \tThe frame data
    
    * newfd: rpy2.robjects.vectors.ListVector or numpy.ndarray
    \tThe R representation of the new fd object or the sampled frames
    See is_new_evaluated
    
    * sampleid: int
    \tThe id of the sample in data
    
    * channel: int
    \tThe channel in the Framedata, e.g. 0 for x-Channel of rootjoint
    
    * is_new_evaluated: boolean
    \tIf this is True, the newfd is given as numpy array. If this is False,
    newfd is represented as fd R object
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

    channel = 0    
    
    plt.plot(newframes[:, channel], label='new')
    plt.plot(originalframes[:, channel], label='original')
    plt.legend()
    plt.show()
    
    
    
def main():
    """
    Main function to show the purpose of this module
    """
    
    path = clean_path(get_input_data_folder())
    datapath = os.sep.join((path, 'data', 'walk_leftStance_fd.RData'))
    data = load_r_data(datapath)

    result = fpca_with_pca_on_scores(data)

    sampleid = 0
    newfd = fpca_backprojection(result, sampleid=sampleid)
    
    compare_with_original(data, newfd, sampleid)
    
if __name__=='__main__':
    main()