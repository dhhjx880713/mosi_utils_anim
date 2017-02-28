# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:05:40 2015
REST interface for the MorphableGraphs algorithm based on the Tornado library.
Implemented according to the following tutorial:
http://www.drdobbs.com/open-source/building-restful-apis-with-tornado/240160382?pgno=1
@author: erhe01
"""
import os
# change working directory to the script file directory
file_dir_name, file_name = os.path.split(os.path.abspath(__file__))
os.chdir(file_dir_name)
import urllib2
import socket
import tornado.escape
import tornado.ioloop
import tornado.web
import json
import time
from datetime import datetime
from morphablegraphs import MotionGenerator, AlgorithmConfigurationBuilder, load_json_file, write_to_json_file
from morphablegraphs.utilities.io_helper_functions import get_bvh_writer
from morphablegraphs.motion_generator.annotated_motion_vector import AnnotatedMotionVector
from morphablegraphs.animation_data import BVHReader, Skeleton
SERVICE_CONFIG_FILE = "config" + os.sep + "service.config"
ALGORITHM_CONFIG_FILE = "config" + os.sep + "accuracy_algorithm.config"


class GetMotionHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Starts the morphable graphs algorithm if an input file
        is detected in the request body.
    """

    def __init__(self, application, request, **kwargs):
        tornado.web.RequestHandler.__init__(
            self, application, request, **kwargs)
        self.application = application

    def get(self):
        error_string = "GET request not implemented. Use POST instead."
        print error_string
        self.write(error_string)

    def post(self):
        result_object = self.application.motion_vector.to_unity_local()
        self.write(json.dumps(result_object))


class GenerateMotionHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Starts the morphable graphs algorithm if an input file
        is detected in the request body.
    """

    def __init__(self, application, request, **kwargs):
        tornado.web.RequestHandler.__init__(
            self, application, request, **kwargs)
        self.application = application

    def get(self):
        error_string = "GET request not implemented. Use POST instead."
        print error_string
        self.write(error_string)

    def post(self):
        try:  # try to decode message body
            mg_input = json.loads(self.request.body)
        except:
            self.write("Could not decode input")
            return
        mg_input = self._set_orientation_to_null(mg_input)
        #print mg_input
        #write_to_json_file("output.path", mg_input)
        motion_vector = self.application.generate_motion(mg_input, complete_motion_vector=False)
        #motion_vector.export(".", "unity_test", add_time_stamp=False, export_details=False)
        result_object = motion_vector.to_unity_local()
        self.write(json.dumps(result_object))

    def _set_orientation_to_null(self, mg_input):
        print mg_input
        mg_input["startPose"]["orientation"] = [None, None, None, None]
        for action in mg_input["elementaryActions"]:
            for constraint in action["constraints"]:
                print constraint
                for p in constraint["trajectoryConstraints"]:
                    p["orientation"] = [None, None, None, None]
        return mg_input

class GetSkeletonHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Starts the morphable graphs algorithm if an input file
        is detected in the request body.
    """

    def __init__(self, application, request, **kwargs):
        tornado.web.RequestHandler.__init__(
            self, application, request, **kwargs)
        self.application = application

    def get(self):
        error_string = "GET request not implemented. Use POST instead."
        print error_string
        self.write(error_string)

    def post(self):
        result_object = self.application.skeleton.to_unity_json()
        self.write(json.dumps(result_object))


class UnityRESTApplication(tornado.web.Application):
    """ Extends the Application class with a MotionGenerator instance and algorithm options.
        This allows access to the data in the MGInputHandler class
    """
    def __init__(self, bvh_path, handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__(self, handlers, default_host, transforms)
        bvh_reader = BVHReader(bvh_path)
        self.skeleton = Skeleton()
        self.skeleton.load_from_bvh(bvh_reader)
        self.algorithm_config = None
        with open(ALGORITHM_CONFIG_FILE, "rt") as in_file:
            self.algorithm_config = json.load(in_file)
        with open(SERVICE_CONFIG_FILE, "rt") as in_file:
            self.service_config = json.load(in_file)
        self.motion_vector = AnnotatedMotionVector(self.algorithm_config)
        self.motion_vector.skeleton = self.skeleton
        self.motion_vector.from_bvh_reader(bvh_reader)
        self.activate_joint_map = True
        self.activate_coordinate_transform = False
        start = time.clock()
        self.motion_generator = MotionGenerator(self.service_config, self.algorithm_config)
        print "Finished construction from file in", time.clock() - start, "seconds"


    def generate_motion(self, mg_input, complete_motion_vector=True):
        return self.motion_generator.generate_motion(mg_input, activate_joint_map=self.activate_joint_map,
                                                     activate_coordinate_transform=self.activate_coordinate_transform,
                                                     complete_motion_vector=complete_motion_vector)


class UnityRESTInterface(object):
    """Implements a RESTful interface for MorphableGraphs.

    Parameters:
    ----------
    * service_config_file : String
        Path to service settings
    * output_mode : String
        Can be either "answer_request" or "file_output".
        answer_request: send result to HTTP client
        file_output: save result to files in preconfigured paths.

    How to use from client side:
    ----------------------------
    send POST request to 'http://localhost:port/run_morphablegraphs' with JSON
    formatted input as body.
    Example with urllib2 when output_mode is answer_request:
    request = urllib2.Request(mg_server_url, mg_input_data)
    handler = urllib2.urlopen(request)
    bvh_string, annotations, actions = json.loads(handler.read())

    configuration can be changed by sending the data to the URL
    'http://localhost:port/configmorphablegraphs'
    """

    def __init__(self, bvh_path, port=8888):

        #  Construct morphable graph from files
        self.application = UnityRESTApplication(bvh_path,
                                             [(r"/get_motion", GetMotionHandler),
                                              (r"/generate_motion",GenerateMotionHandler),
                                              (r"/get_skeleton", GetSkeletonHandler)
                                              ])

        self.port = port

    def start(self):
        """ Start the web server loop
        """
        self.application.listen(self.port)
        tornado.ioloop.IOLoop.instance().start()


def main():
    bvh_path = "skeleton.bvh"
    if os.path.isfile(bvh_path):
        mg_service = UnityRESTInterface(bvh_path)
        mg_service.start()
    else:
        print "Error: could not open service or algorithm configuration file"


if __name__ == "__main__":
    main()
