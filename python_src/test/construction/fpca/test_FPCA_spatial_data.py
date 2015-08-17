__author__ = 'hadu01'
import os
import json
from ....morphablegraphs.construction.fpca.FPCA_spatial_data import  FPCASpatialData
from ...libtest import params, pytest_generate_tests
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-4]) + os.sep
TEST_DATA_PATH = ROOTDIR + os.sep + r'../test_data/constrction/fpca'


class TestFPCASpatialData(object):

    def setup_class(self):
        motion_data_file = TEST_DATA_PATH + os.sep + 'test_data_spatial_fpca.json'
        with open(motion_data_file, 'rb') as infile:
            motion_data = json.load(infile)
        self.fpca_spatial_data = FPCASpatialData(motion_data, 7, 0.95)

    param_convert_data_for_fpca = [{'res': (103, 29, 79)}]

    @params(param_convert_data_for_fpca)
    def test_convert_data_for_fpca(self, res):
        self.fpca_spatial_data.convert_data_for_fpca()
        assert  self.fpca_spatial_data.reshaped_data.shape == res