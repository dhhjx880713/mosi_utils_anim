import numpy as np
import scipy.interpolate as si
from mgrd import TimeSpline as MGRDTimeSpline


class LegacyTemporalSplineModel(object):
    def __init__(self, data):
        self.eigen_vectors = np.array(data['eigen_vectors_time'])
        self.mean_vector = np.array(data['mean_time_vector'])
        self.n_basis = int(data['n_basis_time'])
        self.n_dim = 1
        self.n_components = len(self.eigen_vectors.T)
        self.knots = np.asarray(data['b_spline_knots_time'])
        self.eigen_coefs = zip(* self.eigen_vectors)
        self.n_canonical_frames = data['n_canonical_frames']
        self.canonical_time_range = np.arange(0, self.n_canonical_frames)

        self.frame_time = 0.013889
        self.semantic_labels = []
        self.motion_primitive = None  # TODO improve: this has to be set currently from the outside because the motion primitive model is constructed after the spline model

    def get_n_components(self):
        return self.n_components

    def create_spline(self, gamma, labels = None):
        sample_time_function = self._back_transform_gamma_to_canonical_time_function(gamma)
        knots, time_coeffs, degree = si.splrep(self.canonical_time_range, sample_time_function, w=None, k=3)
        return MGRDTimeSpline(time_coeffs, knots, 1, degree, self.semantic_labels, self)

    def _mean_temporal(self):
        """Evaluates the mean time b-spline for the canonical time range.
        Returns
        -------
        * mean_t: np.ndarray
            Discretized mean time function.
        """
        mean_tck = (self.knots, self.mean_vector, 3)
        return si.splev(self.canonical_time_range, mean_tck)

    def back_project_time_function(self, gamma):
        """ Backtransform a lowdimensional vector gamma to the timewarping
        function t(t') and inverse it to t'(t).

        Parameters
        ----------
        * gamma: numpy.ndarray
        \tThe lowdimensional vector

        Returns
        -------
        * time_function: numpy.ndarray
        \tThe indices of the timewarping function t'(t)
        """
        canonical_time_function = self._back_transform_gamma_to_canonical_time_function(gamma)
        sample_time_function = self._invert_canonical_to_sample_time_function(canonical_time_function)
        if self.smooth_time_parameters:
            return self._smooth_time_function(sample_time_function)
        else:
            return sample_time_function

    def _back_transform_gamma_to_canonical_time_function(self, gamma):
        """backtransform gamma to a discrete timefunction reconstruct t by evaluating the harmonics and the mean
        """
        mean_t = self._mean_temporal()
        n_latent_dim = len(self.eigen_coefs)
        t_eigen_spline = [(self.knots, self.eigen_coefs[i], 3) for i in xrange(n_latent_dim)]
        t_eigen_discrete = np.array([si.splev(self.canonical_time_range, spline_definition) for spline_definition in t_eigen_spline]).T
        canonical_time_function = [0]
        for i in xrange(self.n_canonical_frames):
            canonical_time_function.append(canonical_time_function[-1] + np.exp(mean_t[i] + np.dot(t_eigen_discrete[i], gamma)))
        # undo step from timeVarinaces.transform_timefunction during alignment
        canonical_time_function = np.array(canonical_time_function[1:])
        canonical_time_function -= 1.0
        return canonical_time_function

    def _invert_canonical_to_sample_time_function(self, canonical_time_function):
        """ calculate inverse spline and then sample that inverse spline
            # i.e. calculate t'(t) from t(t')
        """
        # 1 get a valid inverse spline
        x_sample = np.arange(self.n_canonical_frames)
        sample_time_spline = si.splrep(canonical_time_function, x_sample, w=None, k=3)
        # 2 sample discrete data from inverse spline
        # canonical_time_function gets inverted to map from sample to canonical time
        frames = np.linspace(1, stop=canonical_time_function[-2], num=np.round(canonical_time_function[-2]))
        sample_time_function = si.splev(frames, sample_time_spline)
        sample_time_function = np.insert(sample_time_function, 0, 0)
        sample_time_function = np.insert(sample_time_function, len(sample_time_function), self.n_canonical_frames-1)
        return sample_time_function