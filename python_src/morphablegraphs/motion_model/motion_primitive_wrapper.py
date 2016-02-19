import numpy as np
from motion_primitive import MotionPrimitive as MGMotionPrimitive
from motion_spline import MotionSpline
from .extended_mgrd_mixture_model import ExtendedMGRDMixtureModel
from mgrd import MotionPrimitiveModel as MGRDMotionPrimitiveModel
from mgrd import QuaternionSplineModel as MGRDQuaternionSplineModel
from mgrd import TemporalSplineModel as MGRDTemporalSplineModel
from ..utilities import load_json_file
from legacy_temporal_spline_model import LegacyTemporalSplineModel


class MotionPrimitiveModelWrapper(object):
    """ Class that wraps the MGRD MotionPrimitiveModel

    """
    def __init__(self):
        self.motion_primitive = None

    def _load_from_file(self, mgrd_skeleton, file_name):
        data = load_json_file(file_name)
        if data is not None:
            self._initialize_from_json(mgrd_skeleton, data)

    def _initialize_from_json(self, mgrd_skeleton, data):
        if "semantic_annotation" in data.keys():
            self.motion_primitive = MotionPrimitiveModelWrapper.load_model_from_json(mgrd_skeleton, data)
        else:
            mm = ExtendedMGRDMixtureModel.load_from_json({'covars': data['gmm_covars'], 'means': data['gmm_means'],
                                                          'weights': data['gmm_weights'] })

            tspm = LegacyTemporalSplineModel(data)
            #animated_joints = []
            #for node in mgrd_skeleton.skeleton_nodes:
            #    if node.get_type != "end" and  node.get_type != "fixed-joint":
            #        animated_joints.append(node.name)
            animated_joints = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]
            #print np.asarray(data['eigen_vectors_spatial']).shape, np.asarray(data['mean_spatial_vector']).shape
            sspm = MGRDQuaternionSplineModel.load_from_json({
                                                        'eigen': np.asarray(data['eigen_vectors_spatial']).T,
                                                        'mean': np.asarray(data['mean_spatial_vector']),
                                                        'n_coeffs': data['n_basis_spatial'],
                                                        'n_dims': data['n_dim_spatial'],
                                                        'knots': np.asarray(data['b_spline_knots_spatial']),
                                                        'degree': 3,
                                                        'translation_maxima': np.asarray(data['translation_maxima']),
                                                        'animated_joints': animated_joints
                                                    })
            self.motion_primitive = MGRDMotionPrimitiveModel(mgrd_skeleton, sspm, tspm, mm)
            #self.motion_primitive = MGMotionPrimitive(None)
            #self.motion_primitive._initialize_from_json(data)


    @staticmethod
    def load_model_from_json(skeleton, mm_data):

        #N_DIM_TIME = 1
        SPLINE_DEGREE = 3

        # the eigen vectors for spatial spline is stored column major
        mm_data['eigen_vectors_spatial'] = np.ascontiguousarray(np.asarray(mm_data['eigen_vectors_spatial']).transpose())
        mm_data['eigen_vectors_temporal_semantic'] = np.ascontiguousarray(np.asarray(mm_data['eigen_vectors_temporal_semantic']).transpose())
        # TODO: serialize as objects to avoid mapping names
        sspm = MGRDQuaternionSplineModel.load_from_json({
            'eigen': mm_data['eigen_vectors_spatial'],
            'mean': mm_data['mean_spatial_vector'],
            'n_coeffs': mm_data['n_basis_spatial'],
            'n_dims': mm_data['n_dim_spatial'],
            'knots': mm_data['b_spline_knots_spatial'],
            'degree': SPLINE_DEGREE,
            'translation_maxima': mm_data['translation_maxima'],
            'animated_joints': mm_data['animated_joints']
        })
        tspm = MGRDTemporalSplineModel.load_from_json({
            'eigen': mm_data['eigen_vectors_temporal_semantic'],
            'mean': mm_data['mean_temporal_semantic_vector'],
            'n_coeffs': mm_data['n_basis_temporal_semantic'],
            'n_dims': mm_data['n_dim_temporal_semantic'],
            'knots': mm_data['b_spline_knots_temporal_semantic'],
            'degree': SPLINE_DEGREE,
            'n_canonical_frames': mm_data['n_canonical_frames'],
            'semantic_annotation': mm_data['semantic_annotation'],
            'frame_time' : mm_data['frame_time'],
            'semantic_labels': mm_data['semantic_annotation']
        })
        # TODO: serialize each cluster separately
        # TODO: store eigen and mean for each cluster to avoid prepare_eigen_vectors()
        n_components = len(np.array(mm_data['gmm_weights']))
        #mm = GMM(n_components, covariance_type='full')
        #mm.weights_ = numpy.array(mm_data['gmm_weights'])
        #mm.means_ = numpy.array(mm_data['gmm_means'])
        #mm.converged_ = True
        #mm.covars_ = numpy.array(mm_data['gmm_covars'])
        mm = ExtendedMGRDMixtureModel.load_from_json({
            'covars': mm_data['gmm_covars'],
            'means': mm_data['gmm_means'],
            'weights': mm_data['gmm_weights']
        })

        return MGRDMotionPrimitiveModel(skeleton, sspm, tspm, mm)

    def sample(self, use_time=True):
        #if self.is_mgrd:
        s_vec = self.sample_low_dimensional_vector()
        quat_spline = self.back_project(s_vec)
        if use_time:
            time_spline = self.motion_primitive.create_time_spline(s_vec)
            quat_spline = time_spline.warp(quat_spline, time_spline.model.frame_time)
        return quat_spline
        #else:
        #    return self.motion_primitive.sample(use_time)

    def sample_low_dimensional_vector(self):
        #if self.is_mgrd:
        return self.motion_primitive.get_random_samples(1)[0]
        #else:
        #    return self.motion_primitive.sample_low_dimensional_vector()

    def back_project(self, s_vec, use_time_parameters=True):
        """ Returns a QuaternionSpline

        Parameters
            s_vec (np.ndarray):

        Returns
            mgrd.QuatSpline

        """
        #if self.is_mgrd:
        quat_spline = self.motion_primitive.create_spatial_spline(s_vec)
        if use_time_parameters:
            time_spline = self.motion_primitive.create_time_spline(s_vec)
            quat_spline = time_spline.warp(quat_spline)
        return quat_spline
        #else:
         #   return self.motion_primitive.back_project(s_vec, use_time_parameters)

    def back_project_time_function(self, s_vec):
        """

        Parameters
            s_vec (np.ndarray):

        Returns
            np.ndarray

        """
        #if self.is_mgrd:
        return np.asarray(self.motion_primitive.create_time_spline(s_vec, labels=[]).evaluate_domain(step_size=1.0))#[:,0]
        #else:
         #   #print time_parameters, step.n_spatial_components, step.n_time_components, len(step.parameters)
         #   return self.motion_primitive.back_project_time_function(s_vec[self.get_n_spatial_components():])

    def get_n_canonical_frames(self):
        #if self.is_mgrd:
        return self.motion_primitive.time.n_canonical_frames-1
        #else:
        #    return self.motion_primitive.n_canonical_frames

    def get_n_spatial_components(self):
        #if self.is_mgrd:
        return self.motion_primitive.spatial.get_n_components()
        #else:
        #    return self.motion_primitive.get_n_spatial_components()

    def get_n_time_components(self):
        #if self.is_mgrd:
        return self.motion_primitive.time.get_n_components()
        #else:
        #    return self.motion_primitive.get_n_time_components()

    def get_gaussian_mixture_model(self):
        #if self.is_mgrd:
        return self.motion_primitive.mixture
        #else:
        #    return self.motion_primitive.gaussian_mixture_model