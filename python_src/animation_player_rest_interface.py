# -*- coding: utf-8 -*-

import os
# change working directory to the script file directory
file_dir_name, file_name = os.path.split(os.path.abspath(__file__))
os.chdir(file_dir_name)
import tornado.escape
import tornado.ioloop
import tornado.web
import json
from morphablegraphs.motion_generator.annotated_motion_vector import AnnotatedMotionVector
from morphablegraphs.animation_data import BVHReader, SkeletonBuilder, SKELETON_MODELS
from morphablegraphs.animation_data.retargeting import retarget_from_src_to_target
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
        print("get motion")
        motion_vector = self.application.get_motion_vector()
        result_object = motion_vector.to_unity_format(scale=1)
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
        print("get skeleton")
        skeleton = self.application.get_skeleton()
        joint_name_map = {key: key for key in skeleton.animated_joints}
        result_object = skeleton.to_unity_format(joint_name_map=joint_name_map, scale=10)
        self.write(json.dumps(result_object))


class UnityRESTApplication(tornado.web.Application):
    """ Extends the Application class with a MotionGenerator instance and algorithm options.
        This allows access to the data in the MGInputHandler class
    """
    def __init__(self, src_path, src_model, target_path=None, target_model=None, handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__(self, handlers, default_host, transforms)
        src = BVHReader(src_path)
        animated_joints = list(src.get_animated_joints())
        self.src_skeleton = SkeletonBuilder().load_from_bvh(src, animated_joints=animated_joints, add_tool_joints=False)
        self.src_skeleton.skeleton_model = src_model

        if target_path is not None:
            target = BVHReader(target_path)
            animated_joints = list(target.get_animated_joints())
            self.target_skeleton = SkeletonBuilder().load_from_bvh(target, animated_joints=animated_joints, add_tool_joints=False)
            self.target_skeleton.skeleton_model = target_model
        else:
            self.target_skeleton = None

        self.algorithm_config = None
        self.src_path = src_path

    def get_skeleton(self):
        if self.target_skeleton:
            return self.target_skeleton
        else:
            return self.src_skeleton

    def get_motion_vector(self):
        src = BVHReader(self.src_path)
        motion_vector = AnnotatedMotionVector(self.algorithm_config)
        motion_vector.skeleton = self.src_skeleton
        motion_vector.from_bvh_reader(src, filter_joints=False)
        scale_factor = 1.0
        frame_range = None
        additional_rotation_map = dict()
        additional_rotation_map["neck_01"] = [30,0,0]
        if self.target_skeleton:
            #print("src", self.src_skeleton.skeleton_model["joints"])
            #print("target", self.target_skeleton.skeleton_model["joints"], self.target_skeleton.animated_joints)
            motion_vector.frames = retarget_from_src_to_target(self.src_skeleton, self.target_skeleton, motion_vector.frames, additional_rotation_map=additional_rotation_map, scale_factor=scale_factor, frame_range=frame_range, place_on_ground=True)
            motion_vector.skeleton = self.target_skeleton

        return motion_vector



class UnityAnimationPlayerInterface(object):
    """Implements the REST Interface used by the CustomAnimationPlayerUI """

    def __init__(self, src_path, src_model, target_path=None, target_model=None, port=8888):
        self.application = UnityRESTApplication(src_path, src_model, target_path, target_model,
                                             [(r"/get_motion", GetMotionHandler),
                                              (r"/generate_motion", GetMotionHandler),
                                               (r"/get_skeleton", GetSkeletonHandler)
                                              ])
        self.port = port

    def start(self):
        """ Start the web server loop
        """
        self.application.listen(self.port)
        print("start")
        tornado.ioloop.IOLoop.instance().start()


def main():
    src_path = "game_engine.bvh"
    model_dir = r"E:\projects\model_data"
    data_dir = model_dir+ os.sep + r"hybrit\2_retargeting"
    src_path = data_dir + os.sep + r"filtered6\fix-screws-by-hand\17-11-20-Hybrit-VW_fix-screws-by-hand_002_snapPoseSkeleton.bvh"
    src_path = data_dir + os.sep +  r"walk\beginLeftStance\walk_001_1_beginleftStance_386_429_mirrored_from_beginrightStance.bvh"
    src_path = data_dir + os.sep + r"walk\beginLeftStance\walk_024_1_beginleftStance_320_369_mirrored_from_beginrightStance.bvh"
    #src_path = data_dir + os.sep + r"filtered6\fix-screws-by-hand\17-11-20-Hybrit-VW_fix-screws-by-hand_002_snapPoseSkeleton.bvh"
    #src_path = data_dir + os.sep + r"walk\beginLeftStance\test_motion.bvh"


    src_path = model_dir+ os.sep + r"hybrit\1_capturing\walk\beginLeftStance_game_engine_skeleton_smoothed_grounded\walk_001_1_beginleftStance_386_429_mirrored_from_beginrightStance.bvh"
    src_model = SKELETON_MODELS["game_engine"]
    target_path = model_dir + os.sep + r"hybrit\custom_target.bvh"
    target_model = SKELETON_MODELS["custom"]
    port = 8889
    if os.path.isfile(src_path):
        mg_service = UnityAnimationPlayerInterface(src_path, src_model, target_path, target_model, port)
        mg_service.start()
    else:
        print("Error: could not open service or algorithm configuration file")


if __name__ == "__main__":
    main()
