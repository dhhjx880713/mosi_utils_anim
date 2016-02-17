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
from morphablegraphs import MotionGenerator, AlgorithmConfigurationBuilder, load_json_file
ALGORITHM_CONFIG_FILE = "config" + os.sep + "algorithm.json"
SERVICE_CONFIG_FILE = "config" + os.sep + "service.json"


def get_newest_file_from_input_directory(service_config):
    input_file = glob.glob(service_config["input_dir"] + os.sep + "*.json")[-1]
    return input_file


def replace_hand_joints_in_input_file(input_file_path):
    input_file = open(input_file_path)
    input_string = input_file.read()
   # input_file.close()
    #input_string = input_string.replace("RightHand", "RightToolEndSite")
    #input_string = input_string.replace("LeftHand", "LeftToolEndSite")
    return json.loads(input_string)


def run_pipeline(service_config, algorithm_config_file):
    """Creates an instance of the morphable graph and runs the synthesis
       algorithm with the input_file and standard parameters.
    """

    input_file = get_newest_file_from_input_directory(service_config)
    
    algorithm_config_builder = AlgorithmConfigurationBuilder()
    if os.path.isfile(algorithm_config_file):
        algorithm_config_builder.from_json(algorithm_config_file)
    algorithm_config = algorithm_config_builder.build()

    start = time.clock()
    motion_generator = MotionGenerator(service_config, algorithm_config)
    print "Finished construction from file in", time.clock() - start, "seconds"

    mg_input = replace_hand_joints_in_input_file(input_file)
    motion_vector = motion_generator.generate_motion(mg_input, export=False)
    motion_vector.export(service_config["output_dir"], service_config["output_filename"])



def main():
    """Loads the latest file added to the input directory specified in
        service_config.json and runs the algorithm.
    """
#    SEED_CONSTANT = 41
#    np.random.seed(SEED_CONSTANT)
    if os.path.isfile(SERVICE_CONFIG_FILE):
        service_config = load_json_file(SERVICE_CONFIG_FILE)

        run_pipeline(service_config, ALGORITHM_CONFIG_FILE)
    else:
        print "Error: Could not read service config file", SERVICE_CONFIG_FILE


if __name__ == "__main__":
    """example call:
       mg_pipeline_interface.py
    """
    import warnings
    warnings.simplefilter("ignore")
    main()