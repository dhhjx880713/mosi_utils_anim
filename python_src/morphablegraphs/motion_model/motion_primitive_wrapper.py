import numpy as np
from motion_primitive import MotionPrimitive as MGMotionPrimitive
from motion_spline import MotionSpline
from mgrd import MotionPrimitiveModel as MGRDMotionPrimitive
from ..utilities import load_json_file


class MixtureModelWrapper(object):
    def __init__(self, mgrd_mixture_model):
        self.mixture_model = mgrd_mixture_model

    def sample(self):
        return self.mixture_model.sample(1)[0]

class MotionPrimitiveWrapper(object):
    def __init__(self):
        self.motion_primitive=None
        self.use_mgrd = False

    def _load_from_file(self, mgrd_skeleton, file_name):
        data = load_json_file(file_name)
        if data is not None:
            self._initialize_from_json(mgrd_skeleton, data)

    def _initialize_from_json(self, mgrd_skeleton, data):
        if "version" in data.keys():
            self.motion_primitive = MGRDMotionPrimitive.load_from_dict(mgrd_skeleton, data)
            self.use_mgrd = True
        else:
            self.motion_primitive = MGMotionPrimitive(None)
            self.motion_primitive._initialize_from_json(data)

    def sample(self, use_time=True):
        if self.use_mgrd:
            s_vec = self.sample_low_dimensional_vector()
            quat_spline = self.back_project(s_vec)
            if use_time:
                time_spline = self.motion_primitive.create_time_spline(s_vec)
                quat_spline = time_spline.warp(quat_spline, time_spline.model.frame_time)
            return quat_spline
        else:
            return self.motion_primitive.sample(use_time)

    def sample_low_dimensional_vector(self):
        if self.use_mgrd:
            return self.motion_primitive.get_random_samples(1)[0]
        else:
            return self.motion_primitive.sample_low_dimensional_vector()

    def back_project(self, s_vec, use_time_parameters=True):
        if self.use_mgrd:
            return self.motion_primitive.create_spatial_spline(s_vec)
        else:
            return self.motion_primitive.back_project(s_vec, use_time_parameters)

    def back_project_time_function(self, parameters):
        if self.use_mgrd:
            return np.asarray(self.motion_primitive.create_time_spline(parameters, labels=[]).evaluate_domain(step_size=1.0))#[:,0]
        else:

            time_parameters = parameters[self.get_n_spatial_components():]
            #print time_parameters, step.n_spatial_components, step.n_time_components, len(step.parameters)
            return self.motion_primitive.back_project_time_function(time_parameters)

    def get_n_canonical_frames(self):
        if self.use_mgrd:
            return self.motion_primitive.time.n_canonical_frames-1
        else:
            return self.motion_primitive.n_canonical_frames

    def get_n_spatial_components(self):
        if self.use_mgrd:
            return self.motion_primitive.spatial.fpca.n_components
        else:
            return self.motion_primitive.get_n_spatial_components()

    def get_n_time_components(self):
        if self.use_mgrd:
            return self.motion_primitive.time.fpca.n_components
        else:
            return self.motion_primitive.get_n_time_components()

    def get_gaussian_mixture_model(self):
        if self.use_mgrd:
            return MixtureModelWrapper(self.motion_primitive.mixture)
        else:
            return self.motion_primitive.gaussian_mixture_model