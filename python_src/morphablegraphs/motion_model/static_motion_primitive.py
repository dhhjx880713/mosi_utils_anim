import numpy as np
from sklearn.mixture.gaussian_mixture import GaussianMixture
from .motion_spline import MotionSpline


class StaticMotionPrimitive(object):
    """ Implements the interface of a motion primitive but always returns the same motion example
    """
    def __init__(self):
        self.motion_spline = None
        self.name = ""
        self.has_time_parameters = False
        self.has_semantic_parameters = False
        self.n_canonical_frames = 0

    def _initialize_from_json(self, data):
        spatial_coefs = data["spatial_coeffs"]
        knots = data["knots"]
        self.n_canonical_frames = data["n_canonical_frames"]
        self.name = data["name"]
        self.time_function = np.array(list(range(self.n_canonical_frames)))
        self.motion_spline = MotionSpline([0], spatial_coefs, self.time_function, knots, None)
        self.gmm = GaussianMixture(n_components=1, covariance_type='full')
        self.gmm.fit([0])

    def sample(self, use_time_parameters=True):
        return [0]

    def back_project(self, s, use_time_parameters=True, speed=1.0):
        return self.motion_spline

    def back_project_time_function(self, gamma, speed=1.0):
        return self.time_function

    def get_n_spatial_components(self):
        return 1

    def get_n_time_components(self):
        return 0

    def get_gaussian_mixture_model(self):
        return self.gmm

    def get_n_canonical_frames(self):
        return self.n_canonical_frames
