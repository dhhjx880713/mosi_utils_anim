# -*- coding: utf-8 -*-

import os
# change working directory to the script file directory
file_dir_name, file_name = os.path.split(os.path.abspath(__file__))
os.chdir(file_dir_name)
import tornado.escape
import tornado.ioloop
import tornado.web
import json
from .morphablegraphs.motion_generator.annotated_motion_vector import AnnotatedMotionVector
from .morphablegraphs.animation_data import BVHReader, SkeletonBuilder
SERVICE_CONFIG_FILE = "config" + os.sep + "service.config"
ALGORITHM_CONFIG_FILE = "config" + os.sep + "accuracy_algorithm.config"

GAME_ENGINE_TO_GAME_ENGINE_MAP = dict()
GAME_ENGINE_TO_GAME_ENGINE_MAP["pelvis"] = "pelvis"
GAME_ENGINE_TO_GAME_ENGINE_MAP["spine_01"] = "spine_01"
GAME_ENGINE_TO_GAME_ENGINE_MAP["spine_02"] = "spine_02"
GAME_ENGINE_TO_GAME_ENGINE_MAP["neck_01"] = "neck_01"
GAME_ENGINE_TO_GAME_ENGINE_MAP["head"] = "head"
GAME_ENGINE_TO_GAME_ENGINE_MAP["clavicle_l"] = "clavicle_l"
GAME_ENGINE_TO_GAME_ENGINE_MAP["clavicle_r"] = "clavicle_r"
GAME_ENGINE_TO_GAME_ENGINE_MAP["upperarm_l"] = "upperarm_l"
GAME_ENGINE_TO_GAME_ENGINE_MAP["upperarm_r"] = "upperarm_r"
GAME_ENGINE_TO_GAME_ENGINE_MAP["lowerarm_l"] = "lowerarm_l"
GAME_ENGINE_TO_GAME_ENGINE_MAP["lowerarm_r"] = "lowerarm_r"
GAME_ENGINE_TO_GAME_ENGINE_MAP["hand_l"] = "hand_l"
GAME_ENGINE_TO_GAME_ENGINE_MAP["hand_r"] = "hand_r"
GAME_ENGINE_TO_GAME_ENGINE_MAP["thigh_l"] = "thigh_l"
GAME_ENGINE_TO_GAME_ENGINE_MAP["thigh_r"] = "thigh_r"
GAME_ENGINE_TO_GAME_ENGINE_MAP["calf_l"] = "calf_l"
GAME_ENGINE_TO_GAME_ENGINE_MAP["calf_r"] = "calf_r"
GAME_ENGINE_TO_GAME_ENGINE_MAP["foot_l"] = "foot_l"
GAME_ENGINE_TO_GAME_ENGINE_MAP["foot_l"] = "foot_l"

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
        print(error_string)
        self.write(error_string)

    def post(self):
        result_object = self.application.motion_vector.to_unity_format(scale=10)
        self.write(json.dumps(result_object))


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
        print(error_string)
        self.write(error_string)

    def post(self):
        joint_name_map = {key: key for key in self.application.skeleton.animated_joints}
        result_object = self.application.skeleton.to_unity_format(joint_name_map=joint_name_map, scale=10)
        self.write(json.dumps(result_object))


class UnityRESTApplication(tornado.web.Application):
    """ Extends the Application class with a MotionGenerator instance and algorithm options.
        This allows access to the data in the MGInputHandler class
    """
    def __init__(self, bvh_path, handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__(self, handlers, default_host, transforms)
        bvh_reader = BVHReader(bvh_path)
        animated_joints = list(bvh_reader.get_animated_joints())
        print(animated_joints)
        self.skeleton = SkeletonBuilder().load_from_bvh(bvh_reader, animated_joints=animated_joints, add_tool_joints=False)
        self.algorithm_config = None
        self.motion_vector = AnnotatedMotionVector(self.algorithm_config)
        self.motion_vector.skeleton = self.skeleton
        self.motion_vector.from_bvh_reader(bvh_reader, filter_joints=False)



class UnityAnimationPlayerInterface(object):
    """Implements the REST Interface used by the CustomAnimationPlayerUI """

    def __init__(self, bvh_path, port=8888):
        self.application = UnityRESTApplication(bvh_path,
                                             [(r"/get_motion", GetMotionHandler),
                                              (r"/get_skeleton", GetSkeletonHandler)
                                              ])
        self.port = port

    def start(self):
        """ Start the web server loop
        """
        self.application.listen(self.port)
        tornado.ioloop.IOLoop.instance().start()


def main():
    bvh_path = "game_engine.bvh"
    if os.path.isfile(bvh_path):
        mg_service = UnityAnimationPlayerInterface(bvh_path)
        mg_service.start()
    else:
        print("Error: could not open service or algorithm configuration file")


if __name__ == "__main__":
    main()
