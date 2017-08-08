# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:11:52 2015

Simple Motion Graphs command line interface for pipeline tests.

@author: erhe01
"""


import os
 # change working directory to the script file directory
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)
import json
import glob
import time
from .morphablegraphs import MotionGenerator, AlgorithmConfigurationBuilder, load_json_file
SERVICE_CONFIG_FILE = "config" + os.sep + "service.config"


def get_newest_file_from_input_dir(service_config):
    input_file = glob.glob(service_config["input_dir"] + os.sep + "*.json")[-1]
    print("Loading constraints from file", input_file)
    return input_file


def replace_hand_joints_in_input_file(input_file_path):
    input_file = open(input_file_path)
    input_string = input_file.read()
   # input_file.close()
    #input_string = input_string.replace("RightHand", "RightToolEndSite")
    #input_string = input_string.replace("LeftHand", "LeftToolEndSite")
    return json.loads(input_string)


def run_pipeline(service_config):
    """Creates an instance of the morphable graph and runs the synthesis
       algorithm with the input_file and standard parameters.
    """

    input_file = get_newest_file_from_input_dir(service_config)

    algorithm_config_builder = AlgorithmConfigurationBuilder()

    algorithm_config_file = "config" + os.sep + service_config["algorithm_settings"] + "_algorithm.config"
    if os.path.isfile(algorithm_config_file):
        print("Load algorithm configuration from", algorithm_config_file)
        algorithm_config_builder.from_json(algorithm_config_file)
    else:
        print("Did not find algorithm configuration file", algorithm_config_file)
    algorithm_config = algorithm_config_builder.build()
    service_config["collision_avoidance_service_url"] = None  # disable collision avoidance

    start = time.clock()
    motion_generator = MotionGenerator(service_config, algorithm_config)
    print("Finished construction from file in", time.clock() - start, "seconds")

    mg_input = replace_hand_joints_in_input_file(input_file)
    motion_vector = motion_generator.generate_motion(mg_input, activate_joint_map=service_config["activate_joint_map"],
                                                     activate_coordinate_transform=service_config["activate_coordinate_transform"])
    motion_vector.export(service_config["output_dir"], service_config["output_filename"])



def main():
    """Loads the latest file added to the input directory specified in
        service_config.json and runs the algorithm.
    """
#    SEED_CONSTANT = 41
#    np.random.seed(SEED_CONSTANT)
    if os.path.isfile(SERVICE_CONFIG_FILE):
        service_config = load_json_file(SERVICE_CONFIG_FILE)
        run_pipeline(service_config)
    else:
        print("Error: Could not read service config file", SERVICE_CONFIG_FILE)


if __name__ == "__main__":
    """example call:
       mg_pipeline_interface.py
    """
    import warnings
    warnings.simplefilter("ignore")
    main()