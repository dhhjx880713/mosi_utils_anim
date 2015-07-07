#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 21.11.2014

@author: MAMAUER
'''
import json
import numpy as np
import os
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects

ROOT_DIR = os.sep.join([".."] * 4)
INPUT_DIR = os.sep.join((ROOT_DIR, 'data', '2 - PCA', 'temporal', 
                         '1 - z_t', 'experiments'))
OUTPUT_DIR = os.sep.join((ROOT_DIR, 'data', '2 - PCA', 'temporal', 
                         '2 - b_splines', 'experiments'))

def create_bsplines(data, numknots=8):
    ''' Compute the bspline in R and returns the R Object 
    
    Parameters
    ----------
    *data : np array
    \tThe (functional) data to be processed
    
    Returns
    -------
    *bspline : functional data object in R
    \tThe functional data generated from input data
    '''
    _data = np.array(data)
    _data = np.transpose(_data)
    
#     r_data = numpy2ri.numpy2ri(_data)
    robjects.conversion.py2ri = numpy2ri.numpy2ri
#    r_data = robjects.conversion.py2ri(_data)
    r_data = robjects.Matrix(np.array(_data))
    length = _data.shape[0]
    maxX = length - 1
        
    rcode = '''
        library(fda)
        basisobj = create.bspline.basis(c(0,{maxX}),{numknots})
        ys = smooth.basis(argvals=seq(0,{maxX},len={length}), 
                          y={data}, 
                          fdParobj=basisobj)
        xfd = ys$fd
        
    '''.format(data=r_data.r_repr(), maxX=maxX, 
               length=length, numknots=numknots)    
    robjects.r(rcode)

    bsplines = robjects.globalenv['xfd']
#    print bsplines    
    return bsplines

def b_splines(elementary_action, motion_primitive, numknots=8):
    """
    Generate functional data representation for motion data usnig fda library
    from R

    Parameters
    ----------
     * elementary_action: String
    \tElementary action of the motion primitive
     * motion_primitive: String
    \tSpecified motion primitive

    Return
    ------
    Save functional data representation of motion data as r objects
    """
    inputfilename = 'z_t_%s_%s.json' % (elementary_action, motion_primitive)
    inputfile = os.sep.join((INPUT_DIR, inputfilename))
    
    fp = open(inputfile, 'r')
    z_t = json.load(fp)
    fp.close()
    file_order = sorted(z_t.keys())
#    values = np.array(z_t.values())
    values = []
    for filename in file_order:
        values.append(z_t[filename])
    values = np.asarray(values)
    
    outputfilename = 'b_splines_%s_%s.rds' % (elementary_action, motion_primitive)
    outputfile = os.sep.join((OUTPUT_DIR, outputfilename))

    bsplines = create_bsplines(values, numknots=numknots)    
    
#    print np.asanyarray(bsplines[bsplines.names.index('coefs')]).shape
    
    saveRDS = robjects.r('saveRDS')
    saveRDS(bsplines, outputfile)
    
#    try:
#        shutil.copyfile('functionalData.RData', filename)
#        os.remove('functionalData.RData')
#    except:
#        raise IOError('no existing file or file path wrong')    


def __main__():
    n_basis = 8
    b_splines('place', 'first', numknots = n_basis)
    

if __name__=='__main__':
    __main__()
    