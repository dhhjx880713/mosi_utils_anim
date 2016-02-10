from motion_primitive import MotionPrimitive as MGMotionPrimitive
from mgrd import MotionPrimitiveModel as MGRDMotionPrimitive
from ..utilities import load_json_file


class MotionPrimitiveWrapper(object):
    def __init__(self):
        self.motion_primitive=None

    def _load_from_file(self, file_name):
        data = load_json_file(file_name)
        if data is not None:
            self._initialize_from_json(data)

    def _initialize_from_json(self, data):
        if "version" in data.keys():
            self.motion_primitive = MGRDMotionPrimitive()
        else:
            self.motion_primitive = MGMotionPrimitive(None)
            self.motion_primitive._initialize_from_json(data)

    def sample(self, use_time_parameters=True):
        return self.motion_primitive.sample(use_time_parameters)

    def sample_low_dimensional_vector(self):
        return self.motion_primitive.sample_low_dimensional_vector()

    def back_project(self, low_dimensional_vector, use_time_parameters=True):
        return self.motion_primitive.back_project(low_dimensional_vector, use_time_parameters)

    def get_n_canonical_frames(self):
        return self.motion_primitive.n_canonical_frames

    def get_n_spatial_components(self):
        return self.motion_primitive.get_n_spatial_components()

    def get_n_time_components(self):
        return self.motion_primitive.get_n_spatial_components()

    def back_project_spatial_function(self, parameters):
        return self.motion_primitive.back_project_spatial_function(parameters)

    def back_project_time_function(self, parameters):
        return self.motion_primitive.back_project_time_function(parameters)
