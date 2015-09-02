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
import glob
import time
from morphablegraphs.motion_generator.graph_walk_generator import GraphWalkGenerator
from morphablegraphs.motion_generator.algorithm_configuration import AlgorithmConfigurationBuilder
from morphablegraphs.utilities.io_helper_functions import load_json_file
ALGORITHM_CONFIG_FILE = "config" + os.sep + "algorithm.json"
SERVICE_CONFIG_FILE = "config" + os.sep + "service.json"

def get_newest_file_from_input_directory(service_config):
    input_file = glob.glob(service_config["input_dir"] + os.sep + "*.json")[-1]
    return input_file


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
    graph_walk_generator = GraphWalkGenerator(service_config, algorithm_config)
    print "Finished construction from file in", time.clock() - start, "seconds"

    graph_walk = graph_walk_generator.generate_graph_walk(input_file, export=False)

    if graph_walk.motion_vector.has_frames():  # checks for quat_frames in result_tuple
        graph_walk.export_motion(service_config["output_dir"], service_config["output_filename"])
    else:
        print "Error: Failed to generate motion data."


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