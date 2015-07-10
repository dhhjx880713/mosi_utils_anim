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
from lib.io_helper_functions import load_json_file, global_path_dict, convert_quat_frames_to_bvh_string

CONFIG_FILE = "config.json"


class MGInputHandler(tornado.web.RequestHandler):
    """Handles HTTP POST Requests to a registered server url.
        Starts the morphable graphs algorithm if an input file 
        is detected in the request body.
    """
    def __init__(self,application, request, **kwargs ):
        tornado.web.RequestHandler.__init__(self,application, request, **kwargs)
        self.application = application
        
    def get(self):
        error_string = "GET request not implemented. Use POST instead."
        print error_string
        self.write(error_string)
        
    def post(self):
            #  try to decode message body
            try:
                data = json.loads(self.request.body)
            except:
                error_string = "Error: Could not decode request body as JSON."
                self.write(error_string)
                return
                
            # start algorithm if predefined keys were found
            if "elementaryActions" in data.keys():
                self._run_algorithm(data,self.application.options)
            else:
                print data
                self.application.morphable_graph.print_information()
                error_string = "Error: Did not find expected keys in the input data."
                self.write(error_string)
   

 
    def _run_algorithm(self,mg_input,options):
        max_step = -1
        verbose = False
        result_tuple = self.application.morphable_graph.synthesize_motion(mg_input,options=options,
                                                          max_step=max_step,
                                                          verbose=verbose,
                                                          output_dir=global_path_dict["output_dir"],
                                                          output_filename=global_path_dict["output_filename"],
                                                          export=False)
                                                          
        

        if result_tuple[0] != None:  # checks for quat_frames in result_tuple
            if self.application.output_mode == "file_output":
                export_synthesis_result(mg_input, global_path_dict["output_dir"], global_path_dict["output_filename"], \
                                        self.application.morphable_graph.bvh_reader, \
                                        *result_tuple, add_time_stamp=False)
                self.write("succcess")
            else:
                quat_frames = result_tuple[0]
                bvh_string = convert_quat_frames_to_bvh_string(self.application.morphable_graph.bvh_reader, quat_frames)
                result_list = [bvh_string, result_tuple[1], result_tuple[2]]
                self.write(json.dumps(result_list))#send result back
        else:
            error_string = "Error: Failed to generate motion data."
            print error_string
            self.write(error_string)
        
        
        

class MGRestApplication(tornado.web.Application):
    '''Extends the Application class with a MorphableGraph instance and options.
        This allows access to the data in the MGInputHandler class
    '''
        
    def __init__(self,morphable_graph, options, output_mode, handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__( self, handlers, default_host, transforms)
        self.morphable_graph = morphable_graph
        self.options = options
        self.output_mode = output_mode

        
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
    def __init__(self, webApplication,port=8889):
        threading.Thread.__init__(self)
        self.webApplication = webApplication
        self.port = port
    
 
               
    def run(self):
        print "starting server"
        self.webApplication.listen(self.port)       
        tornado.ioloop.IOLoop.instance().start() 

    def stop(self):
        print "stopping server"
        tornado.ioloop.IOLoop.instance().stop()



class MorphableGraphsRESTfulInterface(object):
    """Implements a RESTful interface for MorphableGraphs.
    
    Parameters:
    ----------
    * config_file : String
        Path to algorithm settings
    * port : Integer
        Reserved port of the server process.
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
    bvh_string, annoations, actions = json.loads(handler.read())
    """
    def __init__(self, config_file, port=8888, output_mode="file_output"):
 
        start = time.clock()
        morphable_graph = load_morphable_graph()
        print "finished construction from file in", time.clock()-start, "seconds"
        if os.path.isfile(config_file):
            options = load_json_file(config_file)
        else:
            options = generate_algorithm_settings()

        self.application = MGRestApplication(morphable_graph, options, output_mode,
                                             [(r"/runmorphablegraphs",MGInputHandler)
                                              ])
        self.port = port
        self.server = ServerThread(self.application, self.port)

        
    def start(self):
        self.server.start()
        while True:
            time.sleep(1)
            #print "run"
  
    
def main():
    
    ##TODO place into server configuration file
    global_path_dict["data_root"] = "E:\\projects\\INTERACT\\repository\\"
    global_path_dict["output_dir"] = global_path_dict["data_root"] + r"BestFitPipeline\_Results"
    global_path_dict["output_filename"] = "MGresult"
    
    port = 8888
    mg_service = MorphableGraphsRESTfulInterface(CONFIG_FILE, port)
    mg_service.start()
    return

 
if __name__ == "__main__":
     main()