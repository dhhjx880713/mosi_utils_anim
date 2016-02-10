import numpy as np
from motion_primitive import MotionPrimitive as MGMotionPrimitive
from motion_spline import MotionSpline
from mgrd import MotionPrimitiveModel as MGRDMotionPrimitive
from ..utilities import load_json_file


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
        return self.motion_primitive.sample(use_time)

    def sample_low_dimensional_vector(self):
        if self.use_mgrd:
            return self.motion_primitive.get_random_samples(1)[0]
        else:
            return self.motion_primitive.sample_low_dimensional_vector()

    def back_project(self, s_vec, use_time_parameters=True):
        if self.use_mgrd:
            return self.motion_primitive.create_spatial_spline(s_vec)#.get_motion_vector()
        else:
            return self.motion_primitive.back_project(s_vec, use_time_parameters)

    def get_n_canonical_frames(self):
        if self.use_mgrd:
            return self.motion_primitive.time.n_canonical_frames
        else:
            return self.motion_primitive.n_canonical_frames

    def get_n_spatial_components(self):
        return self.motion_primitive.get_n_spatial_components()

    def get_n_time_components(self):
        return self.motion_primitive.get_n_spatial_components()

    def back_project_spatial_function(self, parameters):
        return self.motion_primitive.back_project_spatial_function(parameters)

    def back_project_time_function(self, parameters):
        return self.motion_primitive.back_project_time_function(parameters)
