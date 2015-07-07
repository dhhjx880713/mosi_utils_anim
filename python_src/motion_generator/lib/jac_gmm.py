# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:40:15 2015

@author: hadu01

Provide Jacobian for GMM
"""
import numpy as np
from sklearn.mixture.gmm import _log_multivariate_normal_density_full
from motion_primitive import MotionPrimitive
import os
from scipy.optimize import minimize
import numdifftools as nd
#from scipy.optimize.optimize import approx_fprime
ROOT_DIR = os.sep.join(['..'] * 3)

def error_func(x0, data):
    """Objective function for optimization
    """
    # for this test, only likelihood is taken into consideration
    motion_primitive, constraints = data
    naturalness = -motion_primitive.gmm.score([x0,])[0]
    if naturalness > 100:
        naturalness = 100
    print 'error is %f.' % (naturalness)
    return naturalness

def jac(x0, data):
    """Jacobian of error function
    """
    tmp = np.reshape(x0, (1, len(x0)))
    motion_primitive, constraints = data 
    logLikelihoods = _log_multivariate_normal_density_full(tmp,
                                                           motion_primitive.gmm.means_, 
                                                           motion_primitive.gmm.covars_)
    logLikelihoods = np.ravel(logLikelihoods)

    numerator = 0

    n_models = len(motion_primitive.gmm.weights_)
    for i in xrange(n_models):
        numerator += np.exp(logLikelihoods[i]) * motion_primitive.gmm.weights_[i] * np.dot(np.linalg.inv(motion_primitive.gmm.covars_[i]), (x0 - motion_primitive.gmm.means_[i]))
#    numerator += logLikelihoods[i] + np.log(motion_primitive.gmm.weights_[i] * np.dot(np.linalg.inv(motion_primitive.gmm.covars_[i]), (x0 - motion_primitive.gmm.means_[i]))    
#    numerator = np.log(numerator)
#    print numerator
    denominator = np.exp(motion_primitive.gmm.score([x0])[0])
#    denominator = motion_primitive.gmm.score([x0])[0]
#    print 'denominator: '
#    print denominator
    logLikelihood_jac = numerator / denominator
#    return np.exp(logLikelihood_jac)
    return logLikelihood_jac

def jac1(x0, data):
    test_jac = approx_fprime(x0, error_func, 1e-7, data)
    return test_jac

def eval_func(data):
    x = data[0]
    y = data[1]

    fx = 2*x **2 + 3*y**3
#    fy = 3*x + 3* y**2
    return fx

def eval_func1(data):
    x = data[0]
    y = data[1]

#    fx = 2*x **2 + 3*y**3
    fy = 3*x + 3* y**2
    return fy

def approx_fprime(xk, f, epsilon, *args):
    """Finite-difference approximation of the gradient of a scalar function.

    Parameters
    ----------
    xk : array_like
        The coordinate vector at which to determine the gradient of `f`.
    f : callable
        The function of which to determine the gradient (partial derivatives).
        Should take `xk` as first argument, other arguments to `f` can be
        supplied in ``*args``.  Should return a scalar, the value of the
        function at `xk`.
    epsilon : array_like
        Increment to `xk` to use for determining the function gradient.
        If a scalar, uses the same finite difference delta for all partial
        derivatives.  If an array, should contain one value per element of
        `xk`.
    \*args : args, optional
        Any other arguments that are to be passed to `f`.

    Returns
    -------
    grad : ndarray
        The partial derivatives of `f` to `xk`.

    See Also
    --------
    check_grad : Check correctness of gradient function against approx_fprime.

    Notes
    -----
    The function gradient is determined by the forward finite difference
    formula::

                 f(xk[i] + epsilon[i]) - f(xk[i])
        f'[i] = ---------------------------------
                            epsilon[i]

    The main use of `approx_fprime` is in scalar function optimizers like
    `fmin_bfgs`, to determine numerically the Jacobian of a function.

    Examples
    --------
    >>> from scipy import optimize
    >>> def func(x, c0, c1):
    ...     "Coordinate vector `x` should be an array of size two."
    ...     return c0 * x[0]**2 + c1*x[1]**2

    >>> x = np.ones(2)
    >>> c0, c1 = (1, 200)
    >>> eps = np.sqrt(np.finfo(np.float).eps)
    >>> optimize.approx_fprime(x, func, [eps, np.sqrt(200) * eps], c0, c1)
    array([   2.        ,  400.00004198])

    """
    f0 = f(*((xk,) + args))
    print 'testing'
    print args
    print f0
    grad = np.zeros((len(xk),), float)
    ei = np.zeros((len(xk),), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0

    return grad

def test_approx_fprime():
    x0 = np.array([1,1])
    y = eval_func(x0)
    print y
    jac = approx_fprime(x0, eval_func, 1e-7)
    print jac
    jac1 = approx_fprime(x0, eval_func1, 1e-7)
    print jac1
#    jac = nd.Jacobian(eval_func)
#    result = jac(x0)
#    print result
#    return 


def get_motion_primitive(elementary_action,
                         primitive_type):
    """
    Return path to store result without trailing os.sep
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    morphable_model_type = "motion_primitives_quaternion_PCA95"
    mm_dir = os.sep.join([ROOT_DIR,
                          data_dir_name,
                          process_step_dir_name,
                          morphable_model_type,
                          'elementary_action_' + elementary_action,
                          '_'.join([elementary_action,
                                    primitive_type,
                                    'quaternion',
                                    'mm.json'])])
    return mm_dir

def test_jac_gmm():
    elementary_action = 'pick'
    primitive_type = 'first'
    motion_primitive = get_motion_primitive(elementary_action,
                                            primitive_type)
    model = MotionPrimitive(motion_primitive)
    constraints = []
    data = (model, constraints)
    tol = 0.001#0.01
    method = 'BFGS'
#    method = 'Nelder-Mead'
    max_iterations= 50
    initial_guess = model.sample(return_lowdimvector=True)
    options = {'maxiter': max_iterations, 'disp' : True}
#    print jac(initial_guess, data)
#    print approx_fprime(initial_guess, error_func, 1e-7, data)
    result = minimize(error_func, 
                      initial_guess, 
                      args=(data,), 
                      tol=tol,
                      method=method,
                      jac = jac,
                      options = options) 
    synthesized_motion = model.back_project(result.x)   
    synthesized_motion.save_motion_vector('synthesized_motion.bvh') 

def test_jac_gmm1():
    elementary_action = 'pick'
    primitive_type = 'first'
    motion_primitive = get_motion_primitive(elementary_action,
                                            primitive_type)
    model = MotionPrimitive(motion_primitive)
    constraints = []
    initial_guess = model.sample(return_lowdimvector=True)
    data = (model, constraints)
    tol = 0.001#0.01
    method = 'BFGS'
#    method = 'Nelder-Mead'
    max_iterations= 50
    options = {'maxiter': max_iterations, 'disp' : True}
    print scipy.misc.derivative(error_func, initial_guess, args=(data,))
#    print np.gradient(error_func(initial_guess, data), 0.1)
#    print jac(initial_guess, data)
#    test_jac = nd.Jacobian(error_func)
#    print test_jac(data)
#    result = minimize(error_func, 
#                      initial_guess, 
#                      args=(data,), 
#                      tol=tol,
#                      method=method,
#                      jac = jac,
#                      options = options) 
#    synthesized_motion = model.back_project(result.x)   
#    synthesized_motion.save_motio

#def err_func_kinematic(x0, data):                                    

if __name__ == "__main__":
    test_jac_gmm()
#    test_approx_fprime()
    