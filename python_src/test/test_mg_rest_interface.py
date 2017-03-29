# encoding: UTF-8
import sys
import os
import json
import pytest
ROOTDIR = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2]) + os.sep
sys.path.append(ROOTDIR)
from mg_rest_interface import MGRESTInterface
import time, threading
import tornado.ioloop
import requests
TESTPATH = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]) + os.sep + 'test_data'
SERVICE_CONFIG_FILE = os.path.join(ROOTDIR, 'config', 'local_service.config')
from morphablegraphs.utilities import load_json_file


class TestMGRESTInterface(object):

    def setup_class(self):

        self.mg_service = MGRESTInterface(SERVICE_CONFIG_FILE)
        threading.Thread(target=self.mg_service.start).start()

    @pytest.fixture(scope='class')
    def load_unity_skeleton(self):
        skeleton_file = os.path.join(TESTPATH, 'skeleton_unity.json')
        return load_json_file(skeleton_file)

    @pytest.mark.parametrize("constraint_file", [
        (os.sep.join([TESTPATH, "mg_input_unity.json"]))
    ])
    def test_generate_motion(self, constraint_file):
        mg_server_url = 'http://localhost:8888/generate_motion'
        input_file = open(constraint_file, "rb")
        mg_input = json.load(input_file)
        data = json.dumps(mg_input)
        r = requests.post(mg_server_url, data)
        result = r.json()
        assert result is not None

    def test_get_skeleton(self, load_unity_skeleton):
        mg_server_url = 'http://localhost:8888/get_skeleton'
        r = requests.post(mg_server_url)
        skeleton_data = r.json()
        assert skeleton_data == load_unity_skeleton

    def teardown_class(self):
        print("The server will self destruct in 15 seconds.")
        time.sleep(15)
        tornado.ioloop.IOLoop.instance().stop()
