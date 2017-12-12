__author__ = 'erhe01'

import time
from .optimizer_base import OptimizerBase
from scipy.optimize import leastsq


class LeastSquares(OptimizerBase):
    """ A wrapper class for the the scipy wrapper of the Levenberg-Marquardt algorithm implemented in Minpack.
        For more details on this method, see
        http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.leastsq.html
        http://devernay.free.fr/hacks/cminpack/index.html

    """
    def run(self, initial_guess):
        """ Runs the optimization for a single motion primitive and a list of constraints
        Returns
        -------
        * x : np.ndarray
              optimal low dimensional motion parameter vector
        """
        if self._objective_function is not None and initial_guess is not None:
            if self.verbose:
                start = time.clock()
                print("Start optimization using LeastSquares")#, self.optimization_settings["method"]
            try:
                result = leastsq(self._objective_function,
                                 initial_guess,
                                 args=(self._error_func_params,),
                                 maxfev=int(self.optimization_settings["max_iterations"]))
                #result = pylevmar.ddif(self._objective_function,initial_guess, measurements,
                #                       self.optimization_settings["max_iterations"],
                #                       data=self._error_func_params)

            except ValueError as e:
                print("Warning: error in LeastSquares.run", e)
                return initial_guess

            if self.verbose:
                print("Finished optimization in ", time.clock()-start, "seconds")
            return result[0]
        else:
            print("Error: No objective function set. Return initial guess instead.")
            return initial_guess
