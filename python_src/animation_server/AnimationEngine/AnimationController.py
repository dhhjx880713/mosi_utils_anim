# -*- coding: utf-8 -*-
#===============================================================================
# author: Erik Herrmann (DFKI GmbH, FB: Agenten und Simulierte Realität) general data structure, Han Du (Daimler AG/DFKI GmbH) additional functions
# last update: 25.4.2014
#===============================================================================

import collections
import numpy as np
from cgkit.cgtypes import *
from Utilities import CustomMath
from IO import AnimationWriter
from IO import AnimationReader
import AnimationData
import JointData


class AnimationController(object):
    def __init__(self):
        self.animationTime = 0.0
        self.currentFrameNumber = 0
        self.loopAnimation = True
        self.playAnimation = False
        self.frameTime = 0.0 #todo use the frame time of the animation controller instead of the animation data

     
    def startAnimation(self):
        self.playAnimation = True


    def pauseAnimation(self):
        self.playAnimation = False
        #print "stop animation"
        
    def stopAnimation(self):
        self.playAnimation = False
        self.resetAnimationTime()
        #print "stop animation"

    def resetAnimationTime(self):
        self.animationTime = 0.0
        self.currentFrameNumber = 0
        self.playAnimation = self.loopAnimation
        
    def toggleAnimationLoop(self):
        self.loopAnimation = not self.loopAnimation
        return


class JointFrameDataController(JointData.JointDescription):
    def __init__(self):
        JointData.JointDescription.__init__(self)
        #self.path  = Scene.PathObject()
        self.frameData = {}#dictionary[]#list of lists
        self.inversBindPose = None # used for skinning to store the inverse of the un-animated global transformation to bring a vertex in the local coordinate system of the joint: vertexGlobalTransformation = jointGlobalTransformation * joint.inversBindPose;
        self.idenitityMatrix = CustomMath.get4x4IdentityMatrix()
        return

    def constructControllerHierarchyFromAnimationFrameData(self,parent,jointFrameData, controllerList, animationId):
        controllerList[jointFrameData.name] = self
        self.level = jointFrameData.level
        self.name = jointFrameData.name
        self.offset = jointFrameData.offset
        self.offsetMatrix = jointFrameData.offsetMatrix
        self.type = jointFrameData.type
        self.rotOrder = jointFrameData.rotOrder
        self.posOrder = jointFrameData.posOrder
        self.numberOfChannels = jointFrameData.numberOfChannels
        self.parent = parent
     
        #todo copy only the frames
        if  len(jointFrameData.frames)  > 0 and animationId >=-1:#only copy the structure when there are frames
            #print self.name
            self.frameData[animationId]=jointFrameData

        for dataChild in jointFrameData.children:
            child = JointFrameDataController()
            child.constructControllerHierarchyFromAnimationFrameData(self,dataChild,controllerList,animationId)
            self.children.append(child)
        return



    def addChild(self,child):
        self.children.append(child)

    def addAnimationFrameData(self,animationId,frameData):
        self.frameData[animationId]=frameData
        childIndex = 0
        for child in self.children:
            child.addAnimationFrameData(animationId,frameData.children[childIndex])
            childIndex +=1

    def addAnimationToCurrentAnimationData(self, animationId, jointDataController):
        # add the joint frame data to current animation joint
#         print "testing"
#         print "current animation index is: " + str(animationIndex)
#         print "length of concatenated data: " + str(len(jointDataController.frameData))
        self.frameData[animationId].addAnimationToCurrentAnimationData(jointDataController.frameData[animationId])
        childIndex = 0
        for child in self.children:
            if child.type == JointData.TYPE_ROOT or child.type == JointData.TYPE_JOINT:
                child.addAnimationToCurrentAnimationData(animationId, jointDataController.children[childIndex])
            childIndex += 1

    def appendFramesToAnimation(self,animationId,frameData):
        if self.type == JointData.TYPE_JOINT or self.type == JointData.TYPE_ROOT:
            self.frameData[animationId].frames += frameData.frames
           # self.frameData[animationId].animationTime  = len(self.frameData[animationId].frames)*
            childIndex = 0
            for child in self.children:
                child.appendFramesToAnimation(animationId,frameData.children[childIndex])
                childIndex +=1



    def cutBeforeFrame(self,animationId,frameNumber):
        if (self.type == JointData.TYPE_JOINT or self.type == JointData.TYPE_ROOT)and animationId in self.frameData.keys():
            del self.frameData[animationId].frames[:frameNumber]#excluding
            for child in self.children:
                child.cutBeforeFrame(animationId,frameNumber)
            return len(self.frameData[animationId].frames)
        else:
            return -1

    def cutAfterFrame(self,animationId,frameNumber):
        if (self.type == JointData.TYPE_JOINT or self.type == JointData.TYPE_ROOT) and animationId in self.frameData.keys():
            del self.frameData[animationId].frames[(frameNumber):]#including
            for child in self.children:
                child.cutAfterFrame(animationId,frameNumber)
            return len(self.frameData[animationId].frames)
        else:
            return -1


    def deleteAnimation(self,animationId):
        if (self.type == JointData.TYPE_JOINT or self.type == JointData.TYPE_ROOT) and animationId in self.frameData.keys(): 
            del self.frameData[animationId]
            for child in self.children:
                child.deleteAnimation(animationId)
        return


    def transformAnimation( self,animationId,translation,eulerAngles):
        '''
        applies translation on offset and calls rotateAnimationFramesByquaternion
        '''
        
        if animationId in self.frameData.keys():
            self.offset +=translation
            self.offsetMatrix = CustomMath.getTranslationMatrix(self.offset)
            q = CustomMath.getQuaternionFromEuler(eulerAngles, self.rotOrder)
            self.rotateAnimationFramesByQuaternion(animationId,q)
       
        return
    #todo use frameData methods
    #note not really correct but temporarily used to aligned forward movements
    def rotateAnimationFramesByQuaternion(self,animationId,q):
        if animationId in self.frameData.keys():
            index = 0
            while index < len( self.frameData[animationId].frames):

                self.frameData[animationId].frames[index].rotateByQuaternion(q)
                #print "here"
#                 EulerAnglesDeg += degOffset
#                 self.frameData[animationId].frames[index].rotationQuaternion = q* self.frameData[animationId].frames[index].rotationQuaternion
#                 self.frameData[animationId].frames[index].translation  = q.rotateVec( self.frameData[animationId].frames[index].translation)

                index += 1

    #source: http://www.gamedev.net/topic/611614-get-relative-matrix-from-parent/
    def changeRelativeCoordinateSystem(self,animationId,oldAnimationData,frameNumber):
        oldTransformation = oldAnimationData.joints[self.name].getAbsoluteTransformation(frameNumber)
        newParentTransformation= self.parent.getAbsoluteTransformation(animationId,frameNumber)
        newRelativeTransformation = np.dot(oldTransformation,np.linalg.inv(newParentTransformation))
        #newRotationMatrix = np.dot(np.linalg.inv(self.offsetMatrix),newRelativeTransformation)
        q = quat()
        q.fromMat(CustomMath.numpyArrayToCGkit4x4(newRelativeTransformation))
        self.frameData[animationId].frames[frameNumber].rotationQuaternion = q
        for child in self.children:
            if child.type != JointData.TYPE_END:
                child.changeRelativeCoordinateSystem(animationId,oldAnimationData,frameNumber)



class SkeletonAnimationController(AnimationController):#SkeletonAnimationDataController
    '''
     @brief An instance of the class is constructed by copying hierarchy and frame data from an instance of the AnimationData class
     multiple animations can be added afterwards which can be treated as states or interpolated
    '''

    def __init__(self,animationData, visualize = True):


        self.globalRotationQuaternion = quat(1,0,0,0)#unit quaternion
        self.globalOffset = vec3(0,-15,0)#in the opengl coordinate system
        self.globalTranslationTransform = mat4([1, 0, 0, 0,
                                                 0, 0, -1, 0,
                                                 0,  1, 0, 0,
                                                 0, 0,  0,  1])#CustomMath.get4x4IdentityMatrix() 

        self.globalRotationTransform = mat4([1, 0, 0, 0,
                                             0, 0, -1, 0,
                                             0, 1 , 0, 0,
                                             0, 0,  0,  1])

        self.mainContext = 0
        #self.frameTime = 0.0
        self.currentAnimationId = -1

        AnimationController.__init__(self)
        self.loadedCorrectly = False
        self.hasVisualization = False
        self.webApp = None
        self.numberOfJoints = 0
        self.filePath = ""
        self.name = ""
        #SkeletonAnimationDataController.__init__(self,animationData)
        self.scaleFactor =1.0
        self.animationCounter = 0#is used to assign animation indices
        self.rootController = JointFrameDataController()
        self.jointControllers = collections.OrderedDict()
        #self.pose = SkeletonPose.SkeletonPose()
        self.constructControllerHierarchyFromAnimationFrameData(animationData, self.getNewAnimationId())
        self.actionBuffer = []
        self.actionDict = {}
        self.constraints = {}



    def getNewAnimationId(self):
        newId = self.animationCounter 
        self.animationCounter+=1
        return newId

    # todo maybe create hard coded hierarchy
    #copies structure data from the SkeletonAnimationData structure to the controller
    def constructControllerHierarchyFromAnimationFrameData(self,animationData, animationId = -1):
        self.scaleFactor= animationData.scaleFactor
        self.frameTime = animationData.frameTime
        self.name = animationData.name
        self.filePath = animationData.filePath
        self.numberOfJoints = len(animationData.joints)
        self.rootController = JointFrameDataController()
        self.jointControllers = collections.OrderedDict()
        self.currentAnimationId = animationId
        self.rootController.constructControllerHierarchyFromAnimationFrameData(None,animationData.root,self.jointControllers,animationId)
        #print "len",len(self.rootController.frameData)
        self.loadedCorrectly = len(self.rootController.frameData)>0#not loaded correctly
        #print "loaded",self.loadedCorrectly
        #self.pose.constructSkeletonHierarchyFromAnimationData(animationData)
        
        return



    def addAnimationFrameData(self,animationId,animationData):
        #if self.currentAnimationId >-1: # stupid error caused three bugs:  wrong default state, wrong transformation, broken graph walk
        self.rootController.addAnimationFrameData(animationId,animationData.root)



    def addAnimationToCurrentAnimationData(self, skeletonDataController):
        # add frames from skeletonDataController to current animation data
        self.rootController.addAnimationToCurrentAnimationData(self.currentAnimationId, skeletonDataController.rootController)



    def appendFramesToCurrentAnimation(self,animationData):
        if self.currentAnimationId >-1:
            self.rootController.appendFramesToAnimation(self.currentAnimationId,animationData.root)


    def getNumberOfFrames(self,animationId =-1):
        #print "ads"+str(animationId)+" "+str(len(self.rootController.frameData[animationId].frames))
        #print len(self.rootController.frameData[animationId].frames)
        if animationId > -1:
             return len(self.rootController.frameData[animationId].frames)
        else:
            #print "current animation index "+ str(self.currentAnimationId)
            #print "current number of frames"+  str(len(self.rootController.frameData[self.currentAnimationId].frames))
            if self.currentAnimationId > -1:
                #print "len",len(self.rootController.frameData),"name",self.rootController.name
                return len(self.rootController.frameData[self.currentAnimationId].frames)

            else:
                return 0

    def isLoadedCorrectly(self):
        #print self.loadedCorrectly
        return self.loadedCorrectly and (self.currentAnimationId in self.rootController.frameData.keys())
#
#     def hasVisualization(self):
#         return self.hasVisualization

    def getFrameTime(self):
        if self.isLoadedCorrectly():
            #print self.frameTime
            return self.frameTime
        else:
            return 0
    def update(self,dt):
        if self.isLoadedCorrectly():
            #calculate current frame based on dt without interpolation. #todo add interpolation
            if self.playAnimation:
                self.animationTime+=dt
                self.currentFrameNumber = int(self.animationTime/self.getFrameTime())       
                
                if len(self.actionBuffer)> 0:
                    if self.currentFrameNumber >= self.actionBuffer[0]:
                        frame = self.actionBuffer.pop(0)
                        self.sendFrameActions(frame)
                        
                #print self.currentFrameNumber
                #print self.animationTime/self.getFrameTime()
                if self.currentFrameNumber >= self.getNumberOfFrames():
                        #self.rootController.path.clear()
                        self.resetAnimationTime()       
                       
                        #print("reset animation")
                else:
               
                    #print "update skeleton",dt
                    #send frame to the web application
                    if self.webApp != None:
                        self.sendPoseFrame()
#                 else:
#                     newPoint = self.getRelativeRootPositionAtFrame(self.currentFrameNumber) #self.getScaledAbsoluteRootPositionAtFrame(self.currentFrameNumber,self.animationData.scaleFactor)
#                     self.rootController.path.addPoint(newPoint*self.scaleFactor)

            return 
        else:
            return None

    def sendPoseFrame(self):
        """ constructs a message and sends it to the client.
            Note the quaternion parameter order differs between cgkit and
            the xml3d implementation w x y z -> x y z w
        """
        if self.webApp != None and self.webApp.activateBroadcast:
            joints ={}
            for jointName in self.jointControllers.iterkeys():
                joint = self.jointControllers[jointName]
                translation = self.jointControllers[jointName].offset+joint.frameData[self.currentAnimationId].frames[self.currentFrameNumber].translation
				
                rotation = joint.frameData[self.currentAnimationId].frames[self.currentFrameNumber].rotationQuaternion
                if joint.type == JointData.TYPE_ROOT:
                    translation += self.globalOffset
                    translation =  self.globalTranslationTransform *translation# np.dot(self.globalTransformation.T,np.array([joint.translation[0],joint.translation[1],joint.translation[2],1]))[:3]#transform and remove 1
                    localRotationMatrix = rotation.toMat4();#convert to matrix
                    rotationMatrix = self.globalRotationTransform * localRotationMatrix#transform
                    rotation = quat().fromMat(rotationMatrix).normalize()#convert to quaternion

                    joints["Root"] ={"translation": [translation.x,translation.y,translation.z],"rotation":[rotation.x,rotation.y,rotation.z,rotation.w]}
                else:
                    #map Spine to Spine1 and Spine1 to Spine2
                    if joint.name == "Spine":
               
                        joints["Spine1"] ={"rotation":[rotation.x,rotation.y,rotation.z,rotation.w],"translation": [translation.x,translation.y,translation.z]}
                    elif joint.name == "Spine1":
                        joints["Spine2"] ={"rotation":[rotation.x,rotation.y,rotation.z,rotation.w],"translation": [translation.x,translation.y,translation.z]}
                    else: #the rest of the joints have the same names
                        joints[joint.name] ={"rotation":[rotation.x,rotation.y,rotation.z,rotation.w],"translation": [translation.x,translation.y,translation.z]}

            frameMessage = {"key": self.currentFrameNumber,
                            "messageType": "pose_frame",
                            "target": "male1",
                            "joints": joints
                            }
            self.webApp.sendData(frameMessage)
        return
        
    def sendFrameActions(self,frame):
        """ Constructs an action messeage and sends it to the clients.
            "messageType": "event",   
            "event": 
                    {
                    	 “type”: “attach”,
                         “parameters”:{ “target”:”W212_E55_899_Miko” ,“newparent”: ”LeftHand” }
                    }

        """
        for action in self.actionDict[str(frame)]:
            message = { "messageType": "event",
                       "event":
                            {
                           "type":action["event"],
                           "parameters": action["parameters"]
                           }
                       }
            print "action message",action,message
            self.webApp.sendData(message)
            
    def stopAnimation(self):
        self.playAnimation = False
        self.resetAnimationTime()
        self.sendPoseFrame()
        #print "stop animation"    

    def resetAnimationTime(self):
        """ Brings the controller back to the initial state"""
        
        AnimationController.resetAnimationTime(self)
        self.fillActionBuffer()

    def setActions(self,actionDict):
        """ Sets the actions corresponding to the animation keyframes
        """
        self.actionDict = actionDict
        self.fillActionBuffer()
        
    def fillActionBuffer(self):
        """ Fills a list with keys that are associated with
            actions.
        """
        self.actionBuffer  =[]
        for key in self.actionDict.keys():
            self.actionBuffer.append(int(key))
        self.actionBuffer.sort()
        print "actionBuffer",self.actionBuffer

