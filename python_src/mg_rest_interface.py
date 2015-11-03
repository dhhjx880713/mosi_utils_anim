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
import tornado.escape
import tornado.ioloop
import tornado.web
import json
import time
from morphablegraphs.motion_generator.graph_walk_generator import GraphWalkGenerator
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.utilities.io_helper_functions import load_json_file, get_bvh_writer


ALGORITHM_CONFIG_FILE = "config" + os.sep + "algorithm.json"
SERVICE_CONFIG_FILE = "config" + os.sep + "service.json"


class MGInputHandler(tornado.web.RequestHandler):
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
        #  try to decode message body
        try:
            mg_input = json.loads(self.request.body)
        except:
            error_string = "Error: Could not decode request body as JSON."
            self.write(error_string)
            return

        # start algorithm if predefined keys were found
        if "elementaryActions" in mg_input.keys() or "tasks" in mg_input.keys():
            graph_walk = self.application.generate_graph_walk(mg_input)

            self._handle_result(graph_walk)
        else:
            print mg_input
            error_string = "Error: Did not find expected keys in the input data."
            self.write(error_string)

    def _handle_result(self, graph_walk):
        """Sends the result back as an answer to a post request.
        """
        if graph_walk.motion_vector.has_frames():
            if self.application.use_file_output_mode:
                graph_walk.export_motion(self.application.service_config["output_dir"],
                              self.application.service_config[
                                  "output_filename"],
                              add_time_stamp=False, export_details=False)
                self.write("succcess")
            else:
                print "answer request", not self.application.use_file_output_mode
                graph_walk.convert_to_motion()
                bvh_writer = get_bvh_writer(
                    self.application.graph_walk_generator.motion_primitive_graph.skeleton,
                    graph_walk.get_quat_frames())
                result_object = {
                    "bvh": bvh_writer.generate_bvh_string(),
                    "annotation": graph_walk.frame_annotation,
                    "event_list": graph_walk.keyframe_events_dict}
                self.write(json.dumps(result_object))  # send result back
        else:
            error_string = "Error: Failed to generate motion data."
            print error_string
            self.write(error_string)


class MGConfiguratiohHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Sets the configuration of the morphable graphs algorithm
        if an input file is detected in the request body.
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
        #  try to decode message body
        try:
            algorithm_config = json.loads(self.request.body)
        except:
            error_string = "Error: Could not decode request body as JSON."
            self.write(error_string)
            return
        if "use_constraints" in algorithm_config.keys():
            self.application.set_algorithm_config(algorithm_config)
        else:
            print algorithm_config
            error_string = "Error: Did not find expected keys in the input data."
            self.write(error_string)


class MGRestApplication(tornado.web.Application):
    '''Extends the Application class with a MorphableGraph instance and options.
        This allows access to the data in the MGInputHandler class
    '''

    def __init__(self, service_config, algorithm_config,
                 handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__(
            self, handlers, default_host, transforms)
        start = time.clock()
        self.graph_walk_generator = GraphWalkGenerator(
            service_config, algorithm_config)
        print "finished construction from file in", time.clock() - start, "seconds"
        self.algorithm_config = algorithm_config
        self.service_config = service_config
        self.use_file_output_mode = (service_config["output_mode"] == "file_output")
        self.activate_joint_map = service_config["activate_joint_map"]
        print "send result as answer to the request", not self.use_file_output_mode

    def generate_graph_walk(self, mg_input):
        return self.graph_walk_generator.generate_graph_walk(mg_input, export=False, activate_joint_map=self.activate_joint_map)

    def set_algorithm_config(self, algorithm_config):
        self.graph_walk_generator.set_algorithm_config(algorithm_config)


class MorphableGraphsRESTfulInterface(object):
    """Implements a RESTful interface for MorphableGraphs.

    Parameters:
    ----------
    * service_config_file : String
        Path to service settings
    * algorithm_config_file : String
        Path to algorithm settings
    * output_mode : String
        Can be either "answer_request" or "file_output".
        answer_request: send result to HTTP client
        file_output: save result to files in preconfigured paths.

    How to use from client side:
    ----------------------------
    send POST request to 'http://localhost:port/runmorphablegraphs' with JSON
    formatted input as body.
    Example with urllib2 when output_mode is answer_request:
    request = urllib2.Request(mg_server_url, mg_input_data)
    handler = urllib2.urlopen(request)
    bvh_string, annotations, actions = json.loads(handler.read())

    configuration can be changed by sending the data to the URL
    'http://localhost:port/configmorphablegraphs'
    """

    def __init__(self, service_config_file, algorithm_config_file):

        #  Load configurtation files
        service_config = load_json_file(service_config_file)
        algorithm_config_builder = AlgorithmConfigurationBuilder()
        if os.path.isfile(algorithm_config_file):
            algorithm_config_builder.from_json(algorithm_config_file)
        algorithm_config = algorithm_config_builder.build()

        #  Construct morphable graph from files
        self.application = MGRestApplication(service_config, algorithm_config,
                                             [(r"/run_morphablegraphs", MGInputHandler),
                                              (r"/config_morphablegraphs",
                                               MGConfiguratiohHandler)
                                              ])

        self.port = service_config["port"]

    def start(self):
        """ Start the web server loop
        """
        self.application.listen(self.port)
        tornado.ioloop.IOLoop.instance().start()


def main():
    if os.path.isfile(SERVICE_CONFIG_FILE) and os.path.isfile(
            ALGORITHM_CONFIG_FILE):
        mg_service = MorphableGraphsRESTfulInterface(
            SERVICE_CONFIG_FILE, ALGORITHM_CONFIG_FILE)
        mg_service.start()
    else:
        print "Error: could not open service or algorithm configuration file"
    return


if __name__ == "__main__":
    main()
