from motion_primitive import MotionPrimitive as MGMotionPrimitive
from mgrd import MotionPrimitiveModel as MGRDMotionPrimitive

class MotionPrimitiveWrapper(object):
    def __init__(self):
        self.motion_primitive=None

    def _initialize_from_json(self, data):
        self.motion_primitive = MGMotionPrimitive(None)
        self.motion_primitive._initialize_from_json(data)

    def sample(self, use_time_parameters=True):
        return self.motion_primitive.sample(use_time_parameters)

    def sample_low_dimensional_vector(self, use_time_parameters=True):
        return self.motion_primitive.sample_low_dimensional_vector(self, use_time_parameters)

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
