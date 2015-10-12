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
import urllib2
import json
from morphablegraphs.utilities.io_helper_functions import load_json_file
SERVICE_CONFIG_FILE = "config" + os.sep + "service.json"

def get_newest_file_from_input_directory(service_config):
    input_file = glob.glob(service_config["input_dir"] + os.sep + "*.json")[-1]
    return input_file


def run_pipeline(service_config):
    """Creates an instance of the morphable graph and runs the synthesis
       algorithm with the input_file and standard parameters.
    """
    input_file_path = get_newest_file_from_input_directory(service_config)
    input_file = open(input_file_path)
    input_string = input_file.read()
    input_string = input_string.replace("RightHand", "RightToolEndSite")
    input_string = input_string.replace("LeftHand", "LeftToolEndSite")
    mg_input = json.loads(input_string)
    #mg_input = load_json_file(input_file)
    data = json.dumps(mg_input)
    mg_server_url = 'http://localhost:8888/run_morphablegraphs'
    request = urllib2.Request(mg_server_url, data)
    
    print "send constraints and wait for motion generator..."
    handler = urllib2.urlopen(request)
    result = handler.read()
    print result


def main():
    """Loads the latest file added to the input directory specified in
        service_config.json and runs the algorithm.
    """

    if os.path.isfile(SERVICE_CONFIG_FILE):
        service_config = load_json_file(SERVICE_CONFIG_FILE)

        run_pipeline(service_config)
    else:
        print "Error: Could not read service config file", SERVICE_CONFIG_FILE


if __name__ == "__main__":
    """example call:
       mg_pipeline_interface.py
    """
    import warnings
    warnings.simplefilter("ignore")
    main()