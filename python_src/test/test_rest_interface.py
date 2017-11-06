# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:20:26 2015

@author: erhe01
"""
import os
import sys
sys.path.append("..")
import urllib.request, urllib.error, urllib.parse
import httplib2
import json
TESTPATH = os.sep.join([".."] * 2 + ["test_data"])


def test_rest_inteface():
    """ Based on the example from http://isbullsh.it/2012/06/Rest-api-in-python
    """
    mg_input_file = os.sep.join([TESTPATH, "mg_input_unity.json"])
    print(mg_input_file)
    mg_server_url = 'http://localhost:8888/generate_motion'
    input_file = open(mg_input_file, "rb")
    mg_input = json.load(input_file)
    input_file.close()
#    input_dict = {"output_mode": "file_output", "mg_input": mg_input}
    data = json.dumps(mg_input)
    request = urllib.request.Request(mg_server_url, data)
    print("send message and wait for answer...")
    handler = urllib.request.urlopen(request)
    result = handler.read()
    result_file_name = "mg_output.json"
    result_file = open(result_file_name, "wb")
    result_file.write(str(result))
    result_file.close()
    print("wrote result to file", result_file_name)
    #print result
    #assert str(result) == "success"


def test_rest_interface1():
    mg_input_file = os.sep.join([TESTPATH, "mg_input.json"])
    input_file = open(mg_input_file, "rb")
    mg_input = json.load(input_file)
    input_file.close()
#    input_dict = {"output_mode": "file_output", "mg_input": mg_input}
    data = json.dumps(mg_input)
    mg_server_url = 'http://localhost:8888/get_skeleton'
    conn = httplib2.HTTPConnectionWithTimeout(8888)
    conn.request('POST', mg_server_url, data)
    print("send message and wait for answer...")
    response = conn.getresponse()

if __name__ == "__main__":
    test_rest_inteface()
