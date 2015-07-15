# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:40:25 2015

@author: mamauer
"""

import GPy
import numpy as np


class GPMulti(GPy.models.GPRegression):
    """ Gaussian Process Model for Multidimensional output

    Parameters
    ---------
    * X : numpy.ndarray
        input observations
    * Y : numpy.ndarray
        output observations
    * kern : (optional) GPy.kern.kern.kern
        The kernel, defaults to (ARD*ARD*ARD)**ARD

    """
    def __init__(self, X, Y, kern=None):
        self.X = X
        self.Y = Y
        _X, _Y = self._get_multidim_training_pairs(X, Y)
        self.original_outputdim = Y.shape[1]
        if kern is None:
            self.kern = (GPy.kern.RBF(input_dim=X.shape[1], ARD=True) *
                         GPy.kern.RBF(input_dim=X.shape[1], ARD=True) *
                         GPy.kern.RBF(input_dim=X.shape[1], ARD=True)) **\
                         GPy.kern.RBF(input_dim=Y.shape[1], ARD=True)

        super(GPMulti, self).__init__(_X, _Y[:, None], kernel=self.kern)

    def _get_multidim_training_pairs(self, X, Y):
        """Convert X to X_i

        Parameters
        ----------
        * X : numpy.ndarray
        input observations
        * Y : numpy.ndarray
        output observations

        Return
        ------
        _X_new, _Y_new : numpy.ndarrays of new observations
        """

        _X_new = []
        _Y_new = []
        ident = np.identity(Y.shape[1])

        for x_i, y_i in zip(X, Y):
            for i in xrange(Y.shape[1]):
                _X_new.append(x_i.tolist() + ident[i].tolist())
                _Y_new.append(y_i[i])
        return np.array(_X_new), np.array(_Y_new)

    def get_multidim_training_input(self, Xnew):
        """Returns list of x, i from a single x,
        similar to _get_multidim_training_pairs

        Parameters
        ----------
        Xnew : numpy.ndarray
        \tThe X to transform

        Returns
        -------

        _X_new : numpy.ndarray
        \tTransformed Xnew
        """
        _X_new = []
        ident = np.identity(self.original_outputdim)
        for extension in ident:
            _X_new.append(Xnew.tolist() + extension.tolist())
        return np.array(_X_new)

    def predict(self, Xnew, full_cov=True,
                **likelihood_args):
        """ Yields the prediction of the GP at Xnew

        Parameters
        ----------
        * Xnew : numpy.ndarray
        \tThe X to predict at (original dimension)
        * full_cov : (optional) Boolean
        \tWhether a full covariance matrix should be returned. default to True

        Returns
        -------
        * mu, cov : numpy.ndarrays
        \tThe mean and covariance matrix of the predicted gaussian distribution
        """

        _Xnew = self.get_multidim_training_input(np.ravel(Xnew))
        return super(GPMulti, self).predict(_Xnew, full_cov, **likelihood_args)


def main():
    raise NotImplementedError

if __name__ == '__main__':
    main()
