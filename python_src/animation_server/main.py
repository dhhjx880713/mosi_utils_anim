'''
Created on Sep 5, 2014

@author: erhe01
'''
import os
import sys
import time
import json
from AnimationEngine.AnimationData import SkeletonAnimationData
from AnimationEngine.AnimationController import SkeletonAnimationController
from webSocketServer import IndexHandler,ServerThread,AnimationWebSocketHandler,WebSocketApplication
from Utilities.directory_observation import DirectoryObserverThread
from threading import Semaphore
import tornado

class AnimationServerController(WebSocketApplication):
    """ Creates an AnimationServer and sends the frames of the animation last added to an observed directory to XML3D clients
    """
    def __init__(self,bvh_path,port = 8889):
     
        WebSocketApplication.__init__(self,[(r"/", IndexHandler) ,(r"/websocket",AnimationWebSocketHandler ) , (r"/(.+)", tornado.web.StaticFileHandler, {"path": os.path.join(os.path.dirname(__file__), "public")}) ])
        self.lastFrameTicks = 0
        self.fps = 60.0
        self.frameTime = 1.0 / self.fps
        currentFrameTicks = time.clock()  
        self.nextUpdateTick  = currentFrameTicks 
        print "create observer for path",bvh_path
        self.observer_thread = DirectoryObserverThread(bvh_path,callback=self.addAnimationController)
        print "create animation server"
        ##initialize server
        self.port = port

        self.activateBroadcast =True
      
        self.server_thread = ServerThread(self, self.port)
        self.controllerIndex = -1
        self.animationControllers = []
        #synchronize animation and observer thread using semaphore
        self.sem = Semaphore()
        return
        
    def addConnection(self,connection):
        '''
        is called by the AnimationWebSocketHandler instance of a new connection
        '''
        id = WebSocketApplication.addConnection(self,connection)
        self._send_constraints(id)
        return id     
        
    def startAnimation(self):
        self.sem.acquire(1)
        if self.controllerIndex > -1:
            self.animationControllers[self.controllerIndex].startAnimation()
        self.sem.release()
        return
        
    def pauseAnimation(self):
        self.sem.acquire(1)
        if self.controllerIndex > -1:
            self.animationControllers[self.controllerIndex].pauseAnimation()
        self.sem.release()
        return
        
    def stopAnimation(self):
        self.sem.acquire(1)
        if self.controllerIndex > -1:
            self.animationControllers[self.controllerIndex].stopAnimation()
        self.sem.release()
        return
    
    def addAnimationController(self,path):
        """ Called by the observer thread if a new bvh is added to the observed directory
        """
        #load animation
        if os.path.isfile(path):
            self.sem.acquire(1)
            animation = SkeletonAnimationData()
            animation.buildFromBVHFile(path)            
            
            controller = SkeletonAnimationController(animation, visualize = False)
            controller.playAnimation = False #animation should be started by the client
            controller.loopAnimation = False
            actionsFilePath = animation.filePath[:-4]+"_actions.json"
            if os.path.isfile(actionsFilePath):
                actionsFile = open(actionsFilePath,"rb")
                actionDict = json.load(actionsFile)
                actionsFile.close()
                controller.setActions(actionDict)
            foundConstraints = False
            constraintsPath =  animation.filePath[:-4]+".json"
            if os.path.isfile(constraintsPath):
                controller.constraints["controlPoints"], controller.constraints["keyframes"] = \
                self._read_constraints(constraintsPath)
                foundConstraints = True
                
            controller.webApp = self
            self.controllerIndex = len(self.animationControllers)
            self.animationControllers.append(controller)
            self.sem.release()
            if foundConstraints:
                self._send_constraints()

    def start(self):
        """Starts a server thread and a observer thread and an endless loop
           that controls the animation
        """
        self.server_thread.start()
        self.observer_thread.start()
        try:
            while True:
                #control fps rate by sleeping
                self.nextUpdateTick += self.frameTime;
                sleepTime = self.nextUpdateTick - time.clock();
                #print "sleep time",sleepTime,self.frameTime
                if sleepTime >= 0:
                    time.sleep( sleepTime )
                
                #update delta time for animation
                currentFrameTicks = time.clock()
                dt = currentFrameTicks -self.lastFrameTicks
                self.lastFrameTicks = currentFrameTicks
                #print "update",dt
                self.sem.acquire(1)
                if self.controllerIndex > -1:
                    self.animationControllers[self.controllerIndex].update(dt)
                self.sem.release()
                
        except KeyboardInterrupt:
                self.observer_thread.observer.stop()
        self.observer_thread.join()
        self.server_thread.join()
        return
        
    def _send_constraints(self,client_id = -1):
        self.sem.acquire(1)
        if self.controllerIndex > -1:
            message = {
                        "messageType": "constraints",
                        "constraints": self.animationControllers[self.controllerIndex].constraints
                        }
            if client_id > -1:
                print 'send data to id',client_id
                self.sendDataToId(client_id,message)
            else:
                self.sendData(message)
        self.sem.release()

    def _read_constraints(self,path):

        infile = open(path)
        data = json.load(infile)
        infile.close()
        count = 0
        controlPointList = []
        keyframeConstraints = []
        for entry in data["elementaryActions"]:
            print "entry", count
            elementaryAction = str(entry["action"])
            constraints = entry["constraints"]
            for constraint in constraints:
                if "keyframeConstraints" in constraint.keys():
                    for c in constraint["keyframeConstraints"]:
                        if "position" in c.keys():
                            point = c["position"]
                            point = [p if p is not None else 0 for p in point]
                            keyframeConstraints.append(point)
                            print "add position keyframe constraint", point, elementaryAction 

                elif "trajectoryConstraints" in constraint.keys():
                    controlPoints = []
                    for c in constraint["trajectoryConstraints"]:
                        if "position" in c.keys():
                            point = c["position"]
                            point = [p if p is not None else 0 for p in point]
                            controlPoints.append(point)
                    controlPointList += controlPoints
                count += 1
        return controlPointList, keyframeConstraints


    
if __name__ == "__main__":

    #local_path = os.path.dirname(__file__)
    #bvh_path = local_path + r"\..\..\BestFitPipeline\_Results"
    bvh_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    controller = AnimationServerController(bvh_path,port=8889)
    controller.start()

    
   

   
            