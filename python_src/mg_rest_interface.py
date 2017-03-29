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
from morphablegraphs.utilities import write_message_to_log, LOG_MODE_DEBUG, LOG_MODE_INFO, LOG_MODE_ERROR, set_log_mode
import argparse

SERVICE_CONFIG_FILE = "config" + os.sep + "service.config"

ROCKETBOX_TO_GAME_ENGINE_MAP = dict()
ROCKETBOX_TO_GAME_ENGINE_MAP["Hips"] = "pelvis"
ROCKETBOX_TO_GAME_ENGINE_MAP["Spine"] = "spine_01"
ROCKETBOX_TO_GAME_ENGINE_MAP["Spine_1"] = "spine_02"
ROCKETBOX_TO_GAME_ENGINE_MAP["Neck"] = "neck_01"
ROCKETBOX_TO_GAME_ENGINE_MAP["Head"] = "head"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftShoulder"] = "clavicle_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightShoulder"] = "clavicle_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftArm"] = "upperarm_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightArm"] = "upperarm_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftForeArm"] = "lowerarm_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightForeArm"] = "lowerarm_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftHand"] = "hand_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightHand"] = "hand_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftUpLeg"] = "thigh_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightUpLeg"] = "thigh_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftLeg"] = "calf_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightLeg"] = "calf_r"
ROCKETBOX_TO_GAME_ENGINE_MAP["LeftFoot"] = "foot_l"
ROCKETBOX_TO_GAME_ENGINE_MAP["RightFoot"] = "foot_r"


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
        try:
            mg_input = json.loads(self.request.body)
        except:
            error_string = "Error: Could not decode request body as JSON."
            self.write(error_string)
            return

        # start algorithm if predefined keys were found
        if "elementaryActions" in mg_input.keys() or "tasks" in mg_input.keys():

            if mg_input["outputMode"] == "Unity":
                mg_input = self._set_orientation_to_null(mg_input)
                motion_vector = self.application.generate_motion(mg_input, complete_motion_vector=False)
            else:
                motion_vector = self.application.generate_motion(mg_input)

            self._handle_result(mg_input, motion_vector)

        else:
            error_string = "Error: Did not find expected keys in the input data."
            write_message_to_log(error_string, LOG_MODE_ERROR)
            self.write(error_string)

    def _handle_result(self, mg_input, motion_vector):
        """Sends the result back as an answer to a post request.
        """
        if motion_vector.has_frames():
            if mg_input["outputMode"] == "Unity":
                result_object = motion_vector.to_unity_format()
                if self.application.export_motion_to_file:
                    motion_vector.export(self.application.service_config["output_dir"], self.application.service_config["output_filename"],
                                         add_time_stamp=False, export_details=False)
            else:
                result_object = self.convert_to_interact_format(motion_vector)

            self.write(json.dumps(result_object))

        else:
            error_string = "Error: Failed to generate motion data."
            write_message_to_log(error_string, LOG_MODE_ERROR)
            self.write(error_string)

    def convert_to_interact_format(self, motion_vector):
        write_message_to_log("Converting the motion into the BVH format...", LOG_MODE_DEBUG)
        start = time.time()
        bvh_writer = get_bvh_writer(motion_vector.skeleton, motion_vector.frames)
        bvh_string = bvh_writer.generate_bvh_string()
        result_object = {
            "bvh": bvh_string,
            "annotation": motion_vector.keyframe_event_list.frame_annotation,
            "event_list": motion_vector.keyframe_event_list.keyframe_events_dict}
        message = "Finished converting the motion to a BVH string in " + str(time.time() - start) + " seconds"
        write_message_to_log(message, LOG_MODE_INFO)
        if self.application.export_motion_to_file:
            self._export_motion_to_file(bvh_string, motion_vector)
        return result_object

    def _export_motion_to_file(self, bvh_string, motion_vector):
        bvh_filename = self.application.service_config["output_dir"] + os.sep + self.application.service_config["output_filename"]
        if self.application.add_timestamp_to_filename:
            bvh_filename += "_"+unicode(datetime.now().strftime("%d%m%y_%H%M%S"))
        write_message_to_log("export motion to file " + bvh_filename, LOG_MODE_DEBUG)
        with open(bvh_filename+".bvh", "wb") as out_file:
            out_file.write(bvh_string)
        if motion_vector.mg_input is not None:
            write_to_json_file(bvh_filename+ "_input.json", motion_vector.mg_input.mg_input_file)
        if motion_vector.keyframe_event_list is not None:
            motion_vector.keyframe_event_list.export_to_file(bvh_filename)

    def _set_orientation_to_null(self, mg_input):
        if "setOrientationFromTrajectory" in mg_input.keys() and mg_input["setOrientationFromTrajectory"]:
            mg_input["startPose"]["orientation"] = [None, None, None]
        for action in mg_input["elementaryActions"]:
            for constraint in action["constraints"]:
                for p in constraint["trajectoryConstraints"]:
                    p["orientation"] = [None, None, None]
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
        write_message_to_log(error_string, LOG_MODE_ERROR)
        self.write(error_string)

    def post(self):
        result_object = self.application.get_skeleton().to_unity_format(joint_name_map=ROCKETBOX_TO_GAME_ENGINE_MAP)
        self.write(json.dumps(result_object))


class SetConfigurationHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Sets the configuration of the morphable graphs algorithm
        if an input file is detected in the request body.
    """

    def __init__(self, application, request, **kwargs):
        tornado.web.RequestHandler.__init__(
            self, application, request, **kwargs)
        self.application = application

    def get(self):
        error_string = "GET request is not implemented. Use POST instead."
        write_message_to_log(error_string, LOG_MODE_ERROR)
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
            print "Set algorithm config to", algorithm_config
        else:
            error_string = "Error: Did not find expected keys in the input data.", algorithm_config
            self.write(error_string)


class MGRestApplication(tornado.web.Application):
    """ Extends the Application class with a MotionGenerator instance and algorithm options.
        This allows access to the data in the MGInputHandler class
    """
    def __init__(self, service_config, algorithm_config, handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__(self, handlers, default_host, transforms)

        self.algorithm_config = algorithm_config
        self.service_config = service_config
        self.export_motion_to_file = False
        self.activate_joint_map = False
        self.apply_coordinate_transform = True
        self.add_timestamp_to_filename = True

        if "export_motion_to_file" in self.service_config.keys():
            self.export_motion_to_file = service_config["export_motion_to_file"]
        if "add_time_stamp" in self.service_config.keys():
            self.add_timestamp_to_filename = service_config["add_time_stamp"]
        if "activate_joint_map" in self.service_config.keys():
            self.activate_joint_map = service_config["activate_joint_map"]
        if "activate_coordinate_transform" in self.service_config.keys():
            self.activate_coordinate_transform = service_config["activate_coordinate_transform"]

        if self.export_motion_to_file:
            write_message_to_log("Motions are written as BVH file to the directory" + self.service_config["output_dir"], LOG_MODE_INFO)
        else:
            write_message_to_log("Motions are returned as answer to the HTTP POST request", LOG_MODE_INFO)

        if not service_config["activate_collision_avoidance"] or not self._test_ca_interface(service_config):
            service_config["collision_avoidance_service_url"] = None

        start = time.clock()
        self.motion_generator = MotionGenerator(self.service_config, self.algorithm_config)
        message = "Finished construction from file in " + str(time.clock() - start) + " seconds"
        write_message_to_log(message, LOG_MODE_INFO)

    def generate_motion(self, mg_input, complete_motion_vector=True):
        return self.motion_generator.generate_motion(mg_input, activate_joint_map=self.activate_joint_map,
                                                     activate_coordinate_transform=self.activate_coordinate_transform,
                                                     complete_motion_vector=complete_motion_vector)

    def is_initiated(self):
        return self.motion_generator.motion_state_graph.skeleton is not None \
                and len(self.motion_generator.motion_state_graph.nodes) > 0

    def get_skeleton(self):
        return self.motion_generator.get_skeleton()

    def set_algorithm_config(self, algorithm_config):
        self.motion_generator.set_algorithm_config(algorithm_config)

    def _test_ca_interface(self, service_config):
        if "collision_avoidance_service_url" in service_config.keys() and "collision_avoidance_service_port" in service_config.keys():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            address = (service_config["collision_avoidance_service_url"], service_config["collision_avoidance_service_port"])
            try:
                write_message_to_log("Try to connect to CA interface using address " + str(address), LOG_MODE_DEBUG)
                s.connect(address)
                s.close()
                return True
            except Exception as e:
                write_message_to_log("Could not create connection" + str(e.message), LOG_MODE_ERROR)
        write_message_to_log("Warning: Could not open collision avoidance service URL " +
                             service_config["collision_avoidance_service_url"] +
                             "\nCollision avoidance will be disabled", LOG_MODE_INFO)
        service_config["collision_avoidance_service_url"] = None
        return False




class MGRESTInterface(object):
    """Implements a REST interface for MorphableGraphs.

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
    send POST request to 'http://localhost:port/generate_motion' with JSON
    formatted input as body.
    Example with urllib2 when output_mode is answer_request:
    request = urllib2.Request(mg_server_url, mg_input_data)
    handler = urllib2.urlopen(request)
    bvh_string, annotations, actions = json.loads(handler.read())

    configuration can be changed by sending the data to the URL
    'http://localhost:port/configmorphablegraphs'
    """

    def __init__(self, args, service_config_file):

        #  Load configuration files
        service_config = load_json_file(service_config_file)
        if args.model_data is not None:
            service_config["model_data"] = args.model_data
        if args.port is not None:
            service_config["port"] = args.port
        if args.log_level is not None:
            service_config["log_level"] = args.log_level

        if "log_level" in service_config.keys():
            set_log_mode(service_config["log_level"])

        algorithm_config_builder = AlgorithmConfigurationBuilder()
        algorithm_config_file = "config" + os.sep + service_config["algorithm_settings"] + "_algorithm.config"
        if os.path.isfile(algorithm_config_file):
            write_message_to_log("Load algorithm configuration from " + algorithm_config_file, LOG_MODE_INFO)
            algorithm_config_builder.from_json(algorithm_config_file)
        else:
            write_message_to_log("Did not find algorithm configuration file " + algorithm_config_file, LOG_MODE_INFO)

        algorithm_config = algorithm_config_builder.build()

        #  Construct morphable graph from files
        self.application = MGRestApplication(service_config, algorithm_config,
                                             [(r"/run_morphablegraphs", GenerateMotionHandler),#legacy
                                              (r"/config_morphablegraphs", SetConfigurationHandler),
                                              (r"/generate_motion", GenerateMotionHandler),
                                               (r"/get_skeleton", GetSkeletonHandler)
                                              ])

        self.port = service_config["port"]

    def start(self):
        """ Start the web server loop
        """
        if self.application.is_initiated():
            self.application.listen(self.port)
            tornado.ioloop.IOLoop.instance().start()
        else:
            write_message_to_log("Error: Could not initiate MG REST service", LOG_MODE_ERROR)

    def stop(self):
        tornado.ioloop.IOLoop.instance().stop()

def parse_commandline_args():
    parser = argparse.ArgumentParser(description="Start the MorphableGraphs REST-interface")
    parser.add_argument("-m", "--model_data", nargs='?', default=None,
                        help="Path to the motion primitive model file.")
    parser.add_argument("-p", "--port", nargs='?', default=None,
                        help="Port of the REST service.")
    parser.add_argument("-l", "--log_level", nargs='?', default=None,
                        help="Set log level. Possible values: 1=info, 2=debug")
    args = parser.parse_args()
    return args


def main(args):
    if os.path.isfile(SERVICE_CONFIG_FILE):
        mg_service = MGRESTInterface(args, SERVICE_CONFIG_FILE)
        mg_service.start()
    else:
        write_message_to_log("Error: could not open service or algorithm configuration file", LOG_MODE_ERROR)


if __name__ == "__main__":
    args = parse_commandline_args()
    main(args)

