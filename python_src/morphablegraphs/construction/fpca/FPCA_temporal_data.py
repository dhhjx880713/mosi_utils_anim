# -*- coding: utf-8 -*-
"""
Created on Sun Aug 02 21:31:09 2015

@author: du
"""
import numpy as np
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import json


class FPCATemporalData(object):

    def __init__(self, temporal_data, n_basis, npc):
        """
        *temoral_data: dictionary
        \tDictionary contains filename and its warping index
        """
        self.temporal_data = temporal_data
        self.n_basis = n_basis
        self.npc = npc

    def _z_t_transform(self):
        self.z_t_transform_data = {}
        for filename in self.temporal_data:
            tmp = self._get_monotonic_indices(self.temporal_data[filename])
            assert self._is_strict_increasing(
                tmp), ("convert %s to monotonic indices failed" % filename)
            w_tmp = np.array(tmp)
            # add one to each entry, because we start with 0
            w_tmp = w_tmp + 1
            w_tmp = np.insert(w_tmp, 0, 0)  # set w(0) to zero

            w_diff = np.diff(w_tmp)
            z = np.log(w_diff)
            self.z_t_transform_data[filename] = z

    def _get_monotonic_indices(self, indices, epsilon=0.01, delta=0):
        """Return an ajusted set of Frameindices which is strictly monotonic

        Parameters
        ----------
        indices : list
        The Frameindices

        Returns
        -------
        A numpy-Float Array with indices similar to the provided list,
        but enforcing strict monotony
        """
        shifted_indices = np.array(indices, dtype=np.float)
        if shifted_indices[0] == shifted_indices[-1]:
            raise ValueError("First and Last element are equal")

        for i in xrange(1, len(shifted_indices) - 1):
            if shifted_indices[i] > shifted_indices[i - 1] + delta:
                continue

            while np.allclose(shifted_indices[i], shifted_indices[i - 1]) or \
                    shifted_indices[i] <= shifted_indices[i - 1] + delta:
                shifted_indices[i] = shifted_indices[i] + epsilon

        for i in xrange(len(indices) - 2, 0, -1):
            if shifted_indices[i] + delta < shifted_indices[i + 1]:
                break

            while np.allclose(shifted_indices[i], shifted_indices[i + 1]) or \
                    shifted_indices[i] + delta >= shifted_indices[i + 1]:
                shifted_indices[i] = shifted_indices[i] - epsilon

        return shifted_indices

    def _is_strict_increasing(self, indices):
        """ Check wether the indices are strictly increasing ore not

        Parameters
        ----------
        indices : list
        The Frameindices

        Returns
        -------
        boolean
        """
        for i in xrange(1, len(indices)):
            if np.allclose(indices[i], indices[i - 1]) or indices[i] < indices[i - 1]:
                return False
        return True

    def fpca_on_temporal_data(self):
        self._z_t_transform()
        file_order = sorted(self.z_t_transform_data.keys())
        timewarping_data = []
        for filename in file_order:
            timewarping_data.append(self.z_t_transform_data[filename])
        timewarping_data = np.transpose(np.asarray(timewarping_data))
        robjects.conversion.py2ri = numpy2ri.numpy2ri
        r_data = robjects.Matrix(np.array(timewarping_data))
        length = timewarping_data.shape[0]
        maxX = length - 1
        rcode = '''
            library(fda)
            basisobj = create.bspline.basis(c(0,{maxX}),{numknots})
            ys = smooth.basis(argvals=seq(0,{maxX},len={length}), 
                              y={data}, 
                              fdParobj=basisobj)
            pca = pca.fd(ys$fd, nharm={nharm})
            pcaVarmax <- varmx.pca.fd(pca)
            scores = pcaVarmax$scores            
        '''.format(data=r_data.r_repr(), maxX=maxX,
                   length=length, numknots=self.n_basis, nharm=self.npc)
        robjects.r(rcode)
        self.temporal_pcaobj = robjects.globalenv['pcaVarmax']
        # scores = np.asarray(
        #     self.temporal_pcaobj[self.temporal_pca.names.index('scores')])



