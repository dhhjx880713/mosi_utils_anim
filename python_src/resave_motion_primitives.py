__author__ = 'erhe01'

import numpy as np
import os
import json
from morphablegraphs.motion_model.motion_primitive import MotionPrimitive
from rpy2 import robjects

class MotionPrimitiveExporter(MotionPrimitive):


   def _initialize_from_json(self,data):
        robjects.r('library("fda")')  #initialize fda for later operations
        super(MotionPrimitiveExporter, self)._initialize_from_json(data)

    def _init_spatial_parameters_from_json(self, data):
        """  Set the parameters for the _inverse_spatial_pca function.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the spatial pca.

        """
        self.translation_maxima = np.array(data['translation_maxima'])
        self.s_pca = {}
        self.s_pca["eigen_vectors"] = np.array(data['eigen_vectors_spatial'])
        self.s_pca["mean_vector"] = np.array(data['mean_spatial_vector'])
        self.s_pca["n_basis"] = int(data['n_basis_spatial'])
        self.s_pca["n_dim"] = int(data['n_dim_spatial'])
        self.s_pca["n_components"] = len(self.s_pca["eigen_vectors"])
        rcode = """
            n_basis = %d
            n_frames = %d
            basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        """% ( self.s_pca["n_basis"],self.n_canonical_frames)
        robjects.r(rcode)
        self.s_pca["basis_function"] = robjects.globalenv['basisobj']
        self.s_pca["knots"] = np.asarray(robjects.r['knots'](self.s_pca["basis_function"],False))


    def _init_time_parameters_from_json(self, data):
        """  Set the parameters for the _inverse_temporal_pca function.

        Parameters
        ----------
        * data: dictionary
        \tThe dictionary must contain all parameters for the spatial pca.

        """
        self.t_pca = {}
        self.t_pca["eigen_vectors"] = np.array(data['eigen_vectors_time'])
        self.t_pca["mean_vector"]= np.array(data['mean_time_vector'])
        self.t_pca["n_basis"]= int(data['n_basis_time'])
        self.t_pca["n_dim"] = 1
        self.t_pca["n_components"]= len(self.t_pca["eigen_vectors"].T)
        rcode ="""
             n_basis = %d
             n_frames = %d
             basisobj = create.bspline.basis(c(0, n_frames - 1), nbasis = n_basis)
        """% ( self.t_pca["n_basis"],self.n_canonical_frames)
        robjects.r(rcode)
        self.t_pca["basis_function"] = robjects.globalenv['basisobj']
        self.t_pca["knots"] = np.asarray(robjects.r['knots'](self.t_pca["basis_function"], False))
        self.t_pca["eigen_coefs"] =zip(* self.t_pca["eigen_vectors"])

    def save_to_file(self, filename):
        data = {'name': self.name,
                'gmm_weights': self.gaussian_mixture_model.weights_.tolist(),
                'gmm_means': self.gaussian_mixture_model.means_.tolist(),
                'gmm_covars': self.gaussian_mixture_model.covars_.tolist(),
                'eigen_vectors_spatial': self.s_pca["eigen_vectors"].tolist(),
                'mean_spatial_vector':  self.s_pca["mean_vector"].tolist(),
                'n_canonical_frames': self.n_canonical_frames,
                'translation_maxima': self.translation_maxima.tolist(),
                'n_basis_spatial': self.s_pca["n_basis"],
                'eigen_vectors_time': self.t_pca["eigen_vectors"].tolist(),
                'mean_time_vector': self.t_pca["mean_vector"].tolist(),
                'n_dim_spatial': self.s_pca["n_dim"],
                'n_basis_time': self.t_pca["n_basis"],
                'b_spline_knots_spatial': self.s_pca["knots"].tolist(),
                'b_spline_knots_time': self.t_pca["knots"].tolist()}
        with open(filename, 'wb') as outfile:
            json.dump(data, outfile)
        return

def add_property_to_elementary_action(directory):
      for root, dirs, files in os.walk(directory):
        for file_name in files:#for each morphable model
            if file_name.endswith("mm.json"):
                print "found motion primitive",file_name
                motion_primitive_file_name = directory+os.sep+file_name
                motion_primitive = MotionPrimitiveExporter(motion_primitive_file_name)
                motion_primitive.save_to_file(motion_primitive_file_name)

def run_over_data_dir():
    root = "E:\\projects\\INTERACT\\repository\\data\\3 - Motion primitives\\motion_primitives_quaternion_PCA95\\"
    for key in next(os.walk(root))[1]:
        elementary_action_dir = root+os.sep+key
        add_property_to_elementary_action(elementary_action_dir)
    return

def resave_file(filename):
    motion_primitive = MotionPrimitiveExporter(filename)
    motion_primitive.save_to_file(filename)

if __name__ == "__main__":
    #run_over_data_dir()
    filename = "E:\\projects\\INTERACT\\git-repos\\morphablegraphs\\test_data\\walk_leftStance_quaternion_mm.json"#motion_model\\motion_sample_test.json"
    resave_file(filename)
