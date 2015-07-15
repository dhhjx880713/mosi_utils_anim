# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:20:26 2015
http://isbullsh.it/2012/06/Rest-api-in-python/
@author: erhe01
"""
import os
import sys
sys.path.append("..")
import urllib2, json
from utilities.io_helper_functions import load_json_file
TESTPATH = os.sep.join([".."]*2+ ["test_data"])

def test_rest_inteface():
    mg_input_file = os.sep.join([TESTPATH,"mg_input.json"])
    print mg_input_file
    mg_server_url = 'http://localhost:8888/runmorphablegraphs'
    mg_input = load_json_file(mg_input_file)
#    input_dict = {"output_mode": "file_output", "mg_input": mg_input}
    data = json.dumps(mg_input)
    request = urllib2.Request(mg_server_url, data)
    print "send message and wait for answer..."
    handler = urllib2.urlopen(request)
    result = handler.read()
    print result
    assert str(result)=="success"
    
if __name__ == "__main__":
    test_rest_inteface()
    