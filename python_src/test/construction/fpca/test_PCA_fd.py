
import os
import numpy as np
import json
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-4]) + os.sep
from morphablegraphs.construction.fpca.pca_functional_data import PCAFunctionalData
from ...libtest import params, pytest_generate_tests

TEST_DATA_PATH = ROOTDIR + os.sep + r'../test_data/constrction/fpca'


class TestPCA_fd(object):

    def setup_class(self):
        test_data_file = TEST_DATA_PATH + os.sep + 'test_data_pca_fd.json'
        with open(test_data_file, 'rb') as infile:
            test_data = json.load(infile)
        print(type(test_data))
        print(np.asarray(test_data).shape)
        self.pca_fd = PCAFunctionalData(test_data, 7, 0.95)

    param_convert_to_fd = [{'res': (7, 29, 79)}]

    @params(param_convert_to_fd)
    def test_convert_to_fd(self, res):
        assert self.pca_fd.fd.shape == res

    param_reshape_fd = [{'res': (29, 553)}]

    @params(param_reshape_fd)
    def test_reshape_fd(self, res):
        assert  self.pca_fd.reshaped_fd.shape == res
        reshaped_fd, shape = self.pca_fd.reshape_fd(self.pca_fd.fd)
        for i in range(79):
            assert round(reshaped_fd[0,i], 5) == round(self.pca_fd.fd[0, 0, i], 5)


    param_project_data = [{'res':[-1.02687138, 1.02924354, 0.13219202, -2.48039802, 0.36659024, -0.57723445,
                                  0.01560552, -0.17724689]}]

    @params(param_project_data)
    def test_project_data(self, res):
        for i in range(len(res)):
            assert  round(self.pca_fd.low_vecs[0][i], 3) == round(res[i], 3)
