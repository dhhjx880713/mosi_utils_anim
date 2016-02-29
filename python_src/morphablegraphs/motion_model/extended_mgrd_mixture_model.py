import numpy as np
import scipy
try:
    from mgrd import MixtureModel as MGRDMixtureModel

    class ExtendedMGRDMixtureModel(MGRDMixtureModel):

        @staticmethod
        def load_from_json(json_data):
            return ExtendedMGRDMixtureModel(
                np.asarray(json_data['covars']),
                np.asarray(json_data['means']),
                np.asarray(json_data['weights'])
            )

        def _log_multivariate_normal_density(self, sample, min_covar=1.e-7):
            """
            Compute loglikelihood of each Gaussian
            The code is from sklean.mixture.gmm
            :param samples: numpy.multiarray
            :param min_covar: scalar
            :return log_prob: numpy.array
            """
            n_samples = 1
            n_dim = sample.shape[0]

            nmix = len(self.means)
            log_prob = np.empty((n_samples, nmix))
            for c, (mu, cv) in enumerate(zip(self.means, self.covars)):
                try:
                    cv_chol = scipy.linalg.cholesky(cv, lower=True)
                except scipy.linalg.LinAlgError:
                    # The model is most probably stuck in a component with too
                    # few observations, we need to reinitialize this components
                    cv_chol = scipy.linalg.cholesky(cv + min_covar * np.eye(n_dim),lower=True)
                cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
                cv_sol = scipy.linalg.solve_triangular(cv_chol, (sample - mu).T, lower=True).T
                #TODO check if it is correct to change the axis to 0
                log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=0) +
                                         n_dim * np.log(2 * np.pi) + cv_log_det)

            return log_prob

        @staticmethod
        def logsumexp(arr, axis=0):
            """Computes the sum of arr assuming arr is in the log domain.
                The code is from sklean.utils.extmath
            """
            arr = np.rollaxis(arr, axis)
            # Use the max to normalize, as with the log this is what accumulates
            # the less errors
            vmax = arr.max(axis=0)
            out = np.log(np.sum(np.exp(arr - vmax), axis=0))
            out += vmax
            return out

        def score(self, sample):
            """ For use with Maximum A Posteriori method used in INTERACT
            """
            if sample.shape[0] != self.means.shape[1]:
                raise ValueError('The length of sample is not correct!')
            log_gaussian_probabilities = (self._log_multivariate_normal_density(sample) + np.log(self.weights))
            logprob = self.logsumexp(log_gaussian_probabilities, axis=1)
            return logprob

except ImportError:
    pass
