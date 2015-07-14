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
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)
import tornado.escape
import tornado.ioloop
import tornado.web
import json
import threading
import time
from controllable_morphable_graph import load_morphable_graph, export_synthesis_result
from constrain_motion import generate_algorithm_settings
from utilities.io_helper_functions import load_json_file, get_bvh_writer


ALGORITHM_CONFIG_FILE = "algorithm_config.json"
SERVICE_CONFIG_FILE = "service_config.json"

class MGInputHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Starts the morphable graphs algorithm if an input file 
        is detected in the request body.
    """
    def __init__(self, application, request, **kwargs ):
        tornado.web.RequestHandler.__init__(self,application, request, **kwargs)
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
            if "elementaryActions" in mg_input.keys():
                motion = self.application.synthesize_motion(mg_input)
                self._handle_result(mg_input, motion, self.application.use_file_output_mode, self.application.service_config, self.application.morphable_graph.skeleton)
            else:
                print mg_input
                self.application.morphable_graph.print_information()
                error_string = "Error: Did not find expected keys in the input data."
                self.write(error_string)
   

 
    def _handle_result(self, mg_input, motion, use_file_output_mode, service_config, skeleton):
        """Sends the result back as an answer to a post request.
        """
        if motion.quat_frames is not None:  # checks for quat_frames in result_tuple
            if use_file_output_mode:
                export_synthesis_result(mg_input, service_config["output_dir"], service_config["output_filename"], \
                                        skeleton, motion.quat_frames, motion.frame_annotation, motion.action_list, add_time_stamp=False)
                self.write("succcess")
            else:

                bvh_writer = get_bvh_writer(skeleton, motion.quat_frames )
                bvh_string = bvh_writer.generate_bvh_string()
                result_list = [bvh_string, motion.frame_annotation, motion.action_list]
                self.write(json.dumps(result_list))#send result back
        else:
            error_string = "Error: Failed to generate motion data."
            print error_string
            self.write(error_string)
        
        
        

class MGRestApplication(tornado.web.Application):
    '''Extends the Application class with a MorphableGraph instance and options.
        This allows access to the data in the MGInputHandler class
    '''
        
    def __init__(self,morphable_graph, service_config, algorithm_config, handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__(self, handlers, default_host, transforms)
        self.morphable_graph = morphable_graph
        self.algorithm_config = algorithm_config
        self.service_config = service_config
        self.use_file_output_mode = (service_config["output_mode"] =="file_output")
       
    def synthesize_motion(self, mg_input):
        max_step = -1
        return self.morphable_graph.synthesize_motion(mg_input,algorithm_config=self.algorithm_config,
                                                          max_step=max_step,
                                                          output_dir=self.service_config["output_dir"],
                                                          output_filename=self.service_config["output_filename"],
                                                          export=False)
                                                          
        
class ServerThread(threading.Thread):
    '''Controls a WebSocketApplication by starting a tornado IOLoop instance in 
    a Thread
    
    Example usage:
    -----
    server = ServerThread(application,8888)
    server.start()
    while True:
        #do something else
        time.sleep(1)
    '''
    def __init__(self, web_application, port=8889):
        threading.Thread.__init__(self)
        self.web_application = web_application
        self.port = port
    
 
               
    def run(self):
        print "starting server"
        self.web_application.listen(self.port)       
        tornado.ioloop.IOLoop.instance().start() 

    def stop(self):
        print "stopping server"
        tornado.ioloop.IOLoop.instance().stop()



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
    """
    def __init__(self, service_config_file, algorithm_config_file):
  
        #  Load configurtation files
        service_config = load_json_file(SERVICE_CONFIG_FILE)    
        if os.path.isfile(algorithm_config_file):
            algorithm_config = load_json_file(algorithm_config_file)
        else:
            algorithm_config = generate_algorithm_settings()
            
        #  Construct morphable graph from files
        start = time.clock()
        morphable_graph = load_morphable_graph(service_config["data_root"])
        print "finished construction from file in", time.clock()-start, "seconds"
        self.application = MGRestApplication(morphable_graph, service_config, algorithm_config, 
                                             [(r"/runmorphablegraphs",MGInputHandler)
                                              ])

        #  Create server thread
        self.port = service_config["port"]
#        self.server = ServerThread(self.application, self.port)

        
    def start(self):
        self.application.listen(self.port)
        tornado.ioloop.IOLoop.instance().start() 
#        self.server.start()
#        while True:
#            time.sleep(1)
#            #print "run"
  
    
def main():
    if os.path.isfile(SERVICE_CONFIG_FILE) and os.path.isfile(ALGORITHM_CONFIG_FILE):  
        mg_service = MorphableGraphsRESTfulInterface(SERVICE_CONFIG_FILE, ALGORITHM_CONFIG_FILE)
        mg_service.start()
    else:
        print "Error: could not open service or algorithm configuration file"
    return

 
if __name__ == "__main__":
     main()