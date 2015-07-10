# -*- coding: utf-8 -*-
"""
Created on Mon Feb 09 10:15:22 2015

@author: mamauer
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import multivariate_normal


class Gaussian(object):
    """ Small Gaussian representation with several functions

    Supports the following functions:
    __mul__ : multiply two Gaussians
    plot: plot the Gaussian

    Parameters
    ----------
    mu: float or numpy.ndarray
    \tThe mu of the Gaussian. If multivariate, this is a numpy.ndarray \
    The default is 0
    sigma: float or numpy.ndarray
    \tThe sigma of the Gaussian. If multivariate, this is a numpy.ndarray \
    The default is 1

    Attributes
    ----------
    mu: float or numpy.ndarray
    \tThe mu of the Gaussian. If multivariate, this is a numpy.ndarray
    sigma: float or numpy.ndarray
    \tThe sigma of the Gaussian. If multivariate, this is a numpy.ndarray
    multi: bool
    \tIndicates whether this gaussian is multivariate or univariate
    """
    def __init__(self, mu=0, sigma=1):
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)
        try:
            self.multi = len(mu) > 1
        except TypeError:
            self.multi = False

    def __mul__(self, other):
        """ Multiply this gaussian with another gaussian """
        if not isinstance(other, Gaussian):
            raise ValueError("Don't know how to multiply Gaussian and %s"
                             % type(other))
        if self.multi and other.multi:
            return self._mul_multivariate(other)
        elif not self.multi and not other.multi:
            return self._mul_univariate(other)
        raise ValueError("Cannot multiply multivariate Gaussian "
                         "with Univariate")

    def _mul_univariate(self, other):
        """ Multiply two univariate gaussians """
        mu = (self.mu * other.sigma**2) + (other.mu * self.sigma**2)
        mu /= (self.sigma**2 + other.sigma**2)
        sigma = 1.0 / ((1.0/self.sigma**2) + (1.0/other.sigma**2))
        return Gaussian(mu, sigma)

    def _mul_multivariate(self, other):
        """ Multiply two multivariate gaussians """
        inv_sigma1 = np.linalg.inv(self.sigma)
        inv_sigma2 = np.linalg.inv(other.sigma)

        sigma = np.linalg.inv(inv_sigma1 + inv_sigma2)
        mu = np.dot(np.dot(sigma, inv_sigma1), self.mu) + \
            np.dot(np.dot(sigma, inv_sigma2), other.mu)
        return Gaussian(mu, sigma)

    def plot(self, dim=None, newfigure=True, **kwargs):
        """ Plot the guassian distribution

        Parameters
        ----------
        * dim: None or tuple
        \tThe dimensions to be plotted
        * newfigure: bool
        \tWhether to create a new figure or not
        * args: list
        \tFurther arguments passed to plot
        """
        if self.multi and len(self.mu) > 2 and (dim is None or len(dim) > 2):
            raise ValueError('Specify at maximum 2 dimensions for multivariate'
                             ' distributions')

        if self.multi and len(self.mu) == 2 and dim is None :
            dim = (0, 1)

        if newfigure:
            plt.figure()

        if self.multi and len(dim) == 2:
            mux = self.mu[dim[0]]
            muy = self.mu[dim[1]]
            sigmax = self.sigma[dim[0], dim[0]]
            sigmay = self.sigma[dim[1], dim[1]]
            sigmaxy = self.sigma[dim[0], dim[1]]

            delta_x = sigmax * 0.1
            delta_y = sigmay * 0.1

            x = np.arange(mux - 50*sigmax, mux + 50*sigmax, delta_x)
            y = np.arange(muy - 50*sigmay, muy + 50*sigmay, delta_y)

            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X; pos[:, :, 1] = Y
            rv = multivariate_normal([mux, muy], [[sigmax, sigmaxy],
                                                  [sigmaxy, sigmay]])

            Z = rv.pdf(pos)
            plt.axis([mux - 50*sigmax, mux + 50*sigmax,
                      muy - 50*sigmay, muy + 50*sigmay])
            C = plt.contour(X, Y, Z, **kwargs)
            plt.clabel(C, inline=1, fontsize=10)
            #plt.legend()


        if not self.multi or len(dim) == 1:
            def distribution(x, mu, sig):
                return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

            if not self.multi:
                mu = self.mu
                sigma = self.sigma
            else:
                mu = self.mu[dim[0]]
                sigma = self.sigma[dim[0], dim[0]]

            x_start = mu - 5 * sigma
            x_end = mu + 5 * sigma
            x = np.linspace(x_start, x_end, num=250)
            plt.plot(x, distribution(x, mu, sigma), **kwargs)
            plt.legend()

    def __str__(self):
        rep = "Gaussian Distribution with %s Mean and %s Varianze" % \
            (str(self.mu), str(self.sigma))
        rep = rep.replace('\n', ' ')
        return rep


def main():
    """ Demonstration function """
    g1 = Gaussian([1, 1], [[10, 2], [2, 1]])
    #g2 = Gaussian([2, 2, 2], [[1, 0], [0, 1]])

    g1.plot(label='Dim 0, 1')
#    g1.plot(dim=(0,), label='Dim 0')
#    g1.plot(dim=(1,), label='Dim 1')
    return

    g3 = g1 * g2
    print g3


    g1.plot(dim=(0,1), label='g1')
    g2.plot(dim=(0,1), newfigure=False, label='g2')
    g3.plot(dim=(0,1), newfigure=False, label='g3')

if __name__ == '__main__':
    main()