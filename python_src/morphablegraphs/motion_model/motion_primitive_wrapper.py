import numpy as np
from motion_primitive import MotionPrimitive as MGMotionPrimitive
try:
    from .extended_mgrd_mixture_model import ExtendedMGRDMixtureModel
    from mgrd import MotionPrimitiveModel as MGRDMotionPrimitiveModel
    from mgrd import QuaternionSplineModel as MGRDQuaternionSplineModel
    from mgrd import TemporalSplineModel as MGRDTemporalSplineModel
    from legacy_temporal_spline_model import LegacyTemporalSplineModel
    has_mgrd = True
except ImportError:
    pass
    has_mgrd = False

from ..utilities import load_json_file, write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR
from sklearn.mixture.gaussian_mixture import GaussianMixture, _compute_precision_cholesky


class MotionPrimitiveModelWrapper(object):
    """ Class that wraps the MGRD MotionPrimitiveModel

    """
    SPLINE_DEGREE = 3

    def __init__(self):
        self.motion_primitive = None

    def _load_from_file(self, mgrd_skeleton, file_name, animated_joints=None, use_mgrd_mixture_model=False):
        data = load_json_file(file_name)
        if data is not None:
            self._initialize_from_json(mgrd_skeleton, data, animated_joints, use_mgrd_mixture_model)

    def _initialize_from_json(self, mgrd_skeleton, data, animated_joints=None, use_mgrd_mixture_model=False):
        if has_mgrd and "tspm" in data.keys():
            write_message_to_log("Init motion primitive model with semantic annotation", LOG_MODE_DEBUG)
            self.motion_primitive = MotionPrimitiveModelWrapper.load_model_from_json(mgrd_skeleton, data, use_mgrd_mixture_model)
        elif has_mgrd and animated_joints is not None:
            write_message_to_log("Init motion primitive model without semantic annotation", LOG_MODE_DEBUG)
            mm = MotionPrimitiveModelWrapper.load_mixture_model(data, use_mgrd_mixture_model)
            tspm = LegacyTemporalSplineModel(data)
            sspm = MGRDQuaternionSplineModel.load_from_json(mgrd_skeleton,{
                                                        'eigen': np.asarray(data['eigen_vectors_spatial']),
                                                        'mean': np.asarray(data['mean_spatial_vector']),
                                                        'n_coeffs': data['n_basis_spatial'],
                                                        'n_dims': data['n_dim_spatial'],
                                                        'knots': np.asarray(data['b_spline_knots_spatial']),
                                                        'degree': self.SPLINE_DEGREE,
                                                        'translation_maxima': np.asarray(data['translation_maxima']),
                                                        'animated_joints': animated_joints
                                                    })
            self._pre_scale_root_translation(sspm, data['translation_maxima'])
            self.motion_primitive = MGRDMotionPrimitiveModel(mgrd_skeleton, sspm, tspm, mm)
        else:
            write_message_to_log("Init legacy motion primitive model", LOG_MODE_DEBUG)
            self.motion_primitive = MGMotionPrimitive(None)
            self.motion_primitive._initialize_from_json(data)

    def _pre_scale_root_translation(self, sspm, translation_maxima):
        """ undo the scaling of the root translation parameters of the principal
        components that was done during offline training

        """
        root_columns = []
        for coeff_idx in range(sspm.n_coeffs):
            coeff_start = coeff_idx * sspm.n_dims
            root_columns += np.arange(coeff_start,coeff_start+3).tolist()

        indices_range = range(len(root_columns))
        x_indices = [root_columns[i] for i in indices_range if i % 3 == 0]
        y_indices = [root_columns[i] for i in indices_range if i % 3 == 1]
        z_indices = [root_columns[i] for i in indices_range if i % 3 == 2]
        sspm.fpca.eigen[:, x_indices] *= translation_maxima[0]
        sspm.fpca.eigen[:, y_indices] *= translation_maxima[1]
        sspm.fpca.eigen[:, z_indices] *= translation_maxima[2]
        sspm.fpca.mean[x_indices] *= translation_maxima[0]
        sspm.fpca.mean[y_indices] *= translation_maxima[1]
        sspm.fpca.mean[z_indices] *= translation_maxima[2]
        print "Prescale root translation"

    def sample_legacy(self, use_time=True):
        return self.motion_primitive.sample(use_time)

    def sample_mgrd(self, use_time=True):
        s_vec = self.sample_low_dimensional_vector()
        quat_spline = self.back_project(s_vec, use_time)
        return quat_spline
    sample = sample_mgrd if has_mgrd else sample_legacy

    def sample_vector_legacy(self):
        return self.motion_primitive.sample_low_dimensional_vector(1)

    def sample_vector_mgrd(self):
        return self.motion_primitive.get_random_samples(1)[0]
    sample_low_dimensional_vector = sample_vector_mgrd if has_mgrd else sample_vector_legacy

    def sample_vectors_legacy(self, n_samples=1):
        return self.motion_primitive.sample_low_dimensional_vector(n_samples)

    def sample_vectors_mgrd(self, n_samples=1):
        return self.motion_primitive.get_random_samples(n_samples)
    sample_low_dimensional_vectors = sample_vectors_mgrd if has_mgrd else sample_vectors_legacy

    def back_project_legacy(self, s_vec, use_time_parameters=True):
        return self.motion_primitive.back_project(s_vec, use_time_parameters)

    def back_project_mgrd(self, s_vec, use_time_parameters=True):
        if len(s_vec.shape) == 2:
            s_vec = np.ravel(s_vec)
        quat_spline = self.motion_primitive.create_spatial_spline(s_vec)
        if use_time_parameters:
            time_spline = self.motion_primitive.create_time_spline(s_vec)
            quat_spline = time_spline.warp(quat_spline)
        return quat_spline
    back_project = back_project_mgrd if has_mgrd else back_project_legacy

    def back_project_time_function_legacy(self, s_vec):
        return self.motion_primitive._back_transform_gamma_to_canonical_time_function(s_vec[self.get_n_spatial_components():])

    def back_project_time_function_mgrd(self, s_vec):
        time_spline = self.motion_primitive.create_time_spline(s_vec, labels=[])
        return np.asarray(time_spline.evaluate_domain(step_size=1.0))[:,0]
    back_project_time_function = back_project_time_function_mgrd if has_mgrd else back_project_time_function_legacy

    def get_n_canonical_frames_legacy(self):
        return self.motion_primitive.n_canonical_frames

    def get_n_canonical_frames_mgrd(self):
        #print max(self.motion_primitive.time.knots)+1, self.motion_primitive.time.n_canonical_frames
        return max(self.motion_primitive.time.knots)+1#.n_canonical_frames
    get_n_canonical_frames = get_n_canonical_frames_mgrd if has_mgrd else get_n_canonical_frames_legacy

    def get_n_spatial_components_legacy(self):
        return self.motion_primitive.get_n_spatial_components()

    def get_n_spatial_components_mgrd(self):
        return self.motion_primitive.spatial.get_n_components()
    get_n_spatial_components = get_n_spatial_components_mgrd if has_mgrd else get_n_spatial_components_legacy

    def get_n_time_components_legacy(self):
        return self.motion_primitive.get_n_time_components()

    def get_n_time_components_mgrd(self):
        return self.motion_primitive.time.get_n_components()
    get_n_time_components = get_n_time_components_mgrd if has_mgrd else get_n_time_components_legacy

    def get_gaussian_mixture_model_legacy(self):
        return self.motion_primitive.gaussian_mixture_model

    def get_gaussian_mixture_model_mgrd(self):
        return self.motion_primitive.mixture
    get_gaussian_mixture_model = get_gaussian_mixture_model_mgrd if has_mgrd else get_gaussian_mixture_model_legacy

    def get_time_eigen_vector_matrix_mgrd(self):
        return self.motion_primitive.time.fpca.eigen

    def get_time_eigen_vector_matrix_legacy(self):
        return self.motion_primitive.t_pca["eigen_vectors"]

    get_time_eigen_vector_matrix = get_time_eigen_vector_matrix_mgrd if has_mgrd else get_time_eigen_vector_matrix_legacy

    def get_spatial_eigen_vector_matrix_legacy(self, joints=None, frame_idx=-1):
        return self.motion_primitive.s_pca["eigen_vectors"].T

    def get_spatial_eigen_vector_matrix_mgrd(self, joints=None, frame_idx=-1):
        LEN_TRANSLATION = 3
        LEN_QUATERNION = 4
        n_frames = self.get_n_canonical_frames()
        # spatial = self.motion_primitive.spatial.clone(None)
        eigen = self.motion_primitive.spatial.fpca.eigen.T
        if joints is None:
            return eigen
        else:
            n_params = 3 + len(self.motion_primitive.spatial.animated_joints) * LEN_QUATERNION
            frame_idx = (len(eigen) / n_params) - 1

            modeled_joints = [j.name for j in self.motion_primitive.spatial.animated_joints]
            joint_indices = []
            if frame_idx > -1:
                frame_range = [frame_idx]
            else:
                frame_range = xrange(n_frames)
            for frame in frame_range:
                frame_offset = frame * n_params
                joint_indices += [frame_offset + i * LEN_QUATERNION + LEN_TRANSLATION for i, j in
                                  enumerate(modeled_joints) if j in joints]

            coeff_indices = []
            for i in joint_indices:
                coeff_indices += list(range(i, i + LEN_QUATERNION))
            return eigen[coeff_indices]

    get_spatial_eigen_vectors = get_spatial_eigen_vector_matrix_mgrd if has_mgrd else get_spatial_eigen_vector_matrix_legacy
    @staticmethod
    def load_model_from_json(skeleton, mm_data, use_mgrd_mixture_model=True):
        sspm = MGRDQuaternionSplineModel.load_from_json(skeleton, mm_data['sspm'])
        tspm = MGRDTemporalSplineModel.load_from_json(mm_data['tspm'])
        mixture_model = MotionPrimitiveModelWrapper.load_mixture_model({
                'gmm_covars': mm_data['gmm']['covars'],
                'gmm_means': mm_data['gmm']['means'],
                'gmm_weights': mm_data['gmm']['weights']
            }, use_mgrd_mixture_model)
        return MGRDMotionPrimitiveModel(skeleton, sspm, tspm, mixture_model)

    @staticmethod
    def load_mixture_model(data, use_mgrd=True):
         if use_mgrd:
             mm = ExtendedMGRDMixtureModel.load_from_json({'covars': data['gmm_covars'], 'means': data['gmm_means'], 'weights': data['gmm_weights']})
         else:
             n_components =len(np.array(data['gmm_weights']))
             mm = GaussianMixture(n_components=n_components, covariance_type='full')#weights_init=np.array(data['gmm_weights']),
            #reg_covar=np.array(data['gmm_covars']),
            #means_init=np.array(data['gmm_means']), covariance_type='full')
             mm.weights_ = np.array(data['gmm_weights'])
             mm.means_ = np.array(data['gmm_means'])
             mm.converged_ = True
             mm.covariances_ = np.array(data['gmm_covars'])
             mm.precisions_cholesky_ = _compute_precision_cholesky(mm.covariances_, covariance_type='full')
             mm.n_dims =len(mm.means_[0])
             # if 'gmm_precisions_cholesky' not in data.keys():

             write_message_to_log("Initialize scipy GMM", LOG_MODE_DEBUG)
         return mm

