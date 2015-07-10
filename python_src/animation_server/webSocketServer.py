'''
Created on Jul 25, 2014
Based on the WebSocket example of the tornado framework
http://www.tornadoweb.org/en/branch2.1/websocket.html
@author: erhe01
'''

import tornado
from tornado import websocket
import threading
import time
import os
import json
import mimetypes

mimetypes.add_type("application/xhtml+xml", ".xhtml")
mimetypes.add_type("application/xml", ".xml")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


  
class AnimationWebSocketHandler(websocket.WebSocketHandler):
        '''
        extends the websocket.WebSocketHandler class to implement open, on_message,on_close
        additionally it adds itself to the list of connections of the web application
        '''
        def __init__(self, application, request, **kwargs):
            websocket.WebSocketHandler.__init__(self,application, request, **kwargs)
            self.id = -1
            self.application = application
                     
        def open(self):
            '''
            add the new connection to then connection list of the class that controls the animation engine thread
            '''
            #print  self.id
            print "WebSocket opened"
            self.id = self.application.addConnection(self)
            
             
#         def on_message(self, message):
#             self.write_message(u"You said: " + message)
            
        def on_message(self, message):
            data = json.loads(message)
            print "recieved message",message
            if 'event' in data.keys():
                if data['event'] == "start_animation":
                    self.application.startAnimation()
                elif data['event'] == "pause_animation":
                    self.application.pauseAnimation()
                elif data['event'] == "stop_animation":
                    self.application.stopAnimation()
        
            print data
            #self.write_message(json.dumps(actionMessage))  
             
        def on_close(self):
            print "WebSocket closed"
            self.application.removeConnection(self.id)

class WebSocketApplication(tornado.web.Application):
    '''
    extends the Application class by a list of connections to allow access from other classes
    '''
        
    def __init__(self,handlers=None, default_host="", transforms=None, **settings):
        tornado.web.Application.__init__( self, handlers, default_host, transforms)
        self.connections = {}
        self.activateBroadcast = False

        self.idCounter = 0
        
    def addConnection(self,connection):
        '''
        is called by the AnimationWebSocketHandler instance of a new connection
        '''
        id = self.idCounter
        self.idCounter+=1
        self.connections[id] = connection
        return id
    
    def removeConnection(self,id):
        '''
        is called by the AnimationWebSocketHandler instance before its destruction
        '''
        del self.connections[id]
        print "removed the connection"  
        
    def sendDataToId(self,id,data):
        '''
        @param message dictionary that is supposed to be send to one websocket client
        send message formated as a dictionary via JSON to all connections. Not efficient but normally only one connection is supposed to be established with another server
        '''
        json_data = json.dumps(data)
        if id in self.connections.keys():
            self.connections[id].write_message(json_data)
        return
        
    def sendData(self,data):
        '''
        @param message dictionary that is supposed to be send to the websocket clients
        send message formated as a dictionary via JSON to all connections. Not efficient but normally only one connection is supposed to be established with another server
        '''
        print "sending data to ",len(self.connections)," clients"
        json_data = json.dumps(data)
        for id in self.connections.keys():
            self.connections[id].write_message(json_data)
        return


    def toggleActiveBroadcast(self):
        
        self.activateBroadcast = not self.activateBroadcast
        if self.activateBroadcast:
            print "activated broadcast"
        else:
            print "deactivated broadcast"
        return
        

        
class IndexHandler(tornado.web.RequestHandler):
    """Regular HTTP handler to serve the web page
       Note there are two ways to display a web page using tornado: dynamic (self.render(filename)) and static (self.redirect(filename) + registered StaticFileHandler)
       Only the static method works for xml3d  without problems"""
       
    def get(self):
        #self.render('animation-player.xhtm')
        #self.redirect('test_scene.xhtml')
        self.redirect('pilotcase_elux.xhtml')
        #self.redirect('male.xhtml')


            
class AnimationServer(WebSocketApplication):
    '''
    creates WebSocketApplication that provides an animation scene and creates one AnimationWebSocketHandler instance for Web Socket connections 
    '''
    def __init__(self,controller):
        WebSocketApplication.__init__(self,[(r"/", IndexHandler),
                                            (r"/websocket",AnimationWebSocketHandler ),
                                             (r"/(.+)", tornado.web.StaticFileHandler, {"path": os.path.join(os.path.dirname(__file__), "public")})
                                             ])
        self.animationController = controller
    
    #     def toggleAnimation(self):
#         if self.animationController != None:
#             self.animationController.toggleAnimationPlayer()
      
 
      
    def startAnimation(self):
         if self.animationController != None:
             self.animationController.startAnimation()
    def pauseAnimation(self):
         if self.animationController != None: 
            self.animationController.pauseAnimation()        
    def stopAnimation(self):
         if self.animationController != None: 
            self.animationController.stopAnimation()
            

            
class ServerThread(threading.Thread):
    '''
    controls a WebSocketApplication by starting a tornado IOLoop instance
    '''
    def __init__(self, webApplication,port =8889):
        threading.Thread.__init__(self)
        self.webApplication = webApplication
        self.port = port
    
 
               
    def run(self):
        print "starting server"
        self.webApplication.listen(self.port)
#         from tornado.httpserver import HTTPServer
#         self.server = HTTPServer(self.webApplication)
#         self.server.listen(self.port)

        
        tornado.ioloop.IOLoop.instance().start() 

    def stop(self):
        print "stopping server"
        tornado.ioloop.IOLoop.instance().stop()

        
  
     
