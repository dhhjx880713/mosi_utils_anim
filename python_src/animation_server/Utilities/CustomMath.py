# -*- coding: utf-8 -*-
#===============================================================================
# author: Erik Herrmann (DFKI GmbH, FB: Agenten und Simulierte Realität)
# last update: 13.1.2014
#===============================================================================
from __future__ import division 
import numpy as np
from math import *
from cgkit.cgtypes import *

class DualQuaternion():
    '''
    #copied from papers of Ben Kenwright:
    A Beginners Guide to Dual-Quaternions (http://wscg.zcu.cz/wscg2012/short/a29-full.pdf)
    Dual-QuaternionsFrom Classical Mechanics to Computer Graphics and Beyond (www.xbdev.net/misc_demos/demos/dual_quaternions_beyond/paper.pdf)
    dependent on cgkit quaternion
    '''
    def __init__(self,rotation = None,translation = None):
        '''
        :param rotation: rotation as quat
        :param translation: translation as vec3
        m_real contains rotation
        m_dual contains translation
        note cgkit takes #w x y z to initialize a quaternion
        '''
        if rotation != None and translation != None:
            self.m_real = rotation.normalize()
            self.m_dual = (quat(0,translation.x,translation.y,translation.z)*self.m_real)*0.5#
            self.normalize()
#             print self.m_real
#             print translation
#             print self.m_dual
        else:
            self.m_real = None
            self.m_dual = None

    def __mul__(self,rightHandSide):
        return DualQuaternion(self.m_real * rightHandSide.m_real,self.m_dual*rightHandSide.m_real +self.m_real*rightHandSide.m_dual)
    def __add__(self,rightHandSide):
        return DualQuaternion(self.m_real + rightHandSide.m_real, self.m_dual + rightHandSide.m_dual)
    def normalize(self):
        mag = self.m_real.dot(self.m_real)
        if mag > 0.000001:
            ret = self
            ret.m_real *= 1.0/mag
            ret.m_dual *= 1.0/mag
        return ret
    
    def getRotation(self):
        return self.m_real
    
    def getTranslation(self):
        translation_q = (self.m_dual*2.0) * self.m_real.conjugate()
        return vec3(translation_q.x,translation_q.y,translation_q.z)
    
    def toMatrix(self):
        #m = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        m = self.m_real.toMat4()
#         w = self.m_real.w
#         x = self.m_real.x
#         y = self.m_real.y
#         z = self.m_real.z
        #rotation
#       m = mat4()
#         m[0,0] = w*w * x*x -y*y -z*z
#         m[0,1] = 2*x*y + 2*w *z
#         m[0,2] = 2*x*z - 2*w*y 
#          
#         m[1,0] = 2*x*y - 2*w*z
#         m[1,1] = w*w + y*y - x*x - z*z
#         m[1,2] = 2*y*z + 2*w*x 
#          
#         m[2,0] = 2*x*z + 2*w *y
#         m[2,1] = 2*y*z -2*w*x
#         m[2,2] = w*w +z*z - x*x -y*y
        #translation
        t = self.getTranslation()
        m[3,0] = t.x
        m[3,1] = t.y
        m[3,2] = t.z
        return m
        
def get4x4ZeroMatrix():
    #Note vectors represent columns
    return np.array([[0.0, 0.0, 0.0, 0.0], 
                           [ 0.0, 0.0, 0.0, 0.0], 
                           [ 0.0, 0.0, 0.0, 0.0], 
                           [ 0.0, 0.0, 0.0, 0.0]], np.float32)
    
def get4x4IdentityMatrix():
    #Note vectors represent columns
    return np.array([[1.0, 0.0, 0.0, 0.0], 
                           [ 0.0, 1.0, 0.0, 0.0], 
                           [ 0.0, 0.0, 1.0, 0.0], 
                           [ 0.0, 0.0, 0.0, 1.0]], np.float32)
def get3x3IdentityMatrix():
    #Note vectors represent columns
    return np.array([[1,0,0],[0,1,0],[0,0,1]], np.float32)

def getRotationAroundXAxis(alpha):
    cx = cos(alpha)
    sx = sin(alpha)
    #Note vectors represent columns
    rotationAroundXAxis = np.array([[1.0 , 0.0, 0.0, 0.0],
                             [0.0, cx ,  sx,0.0],
                              [0.0, -sx,  cx,0.0],
                             [0.0,0.0,0.0,1.0]],np.float32)
    return rotationAroundXAxis


def getRotationAroundYAxis(beta):
    cy = cos(beta)
    sy = sin(beta)
    #Note vectors represent columns
    rotationAroundYAxis = np.array([[ cy,0.0,-sy ,0.0],
                                  [0.0,1.0,0.0,0.0],
                                  [ sy,0.0,cy,0.0],
                                   [0.0,0.0,0.0,1.0]],np.float32)
    return rotationAroundYAxis

def getRotationAroundZAxis(gamma):
    cz = cos(gamma)
    sz = sin(gamma)
    #Note vectors represent columns
    rotationAroundZAxis =  np.array([[ cz, sz,0.0,0.0],
                                  [ -sz, cz,0.0,0.0],
                                  [0.0,0.0,1.0,0.0],
                                   [0.0,0.0,0.0,1.0]],np.float32)
    return rotationAroundZAxis
    
#TODO add also a method that calculates the complete matrix instead of doing multiple matrix multiplications
def get3DRotationMatrix(eulerAnglesDeg,rotOrder):
    rotationMatrix = get4x4IdentityMatrix()
    currentAxis = 0
    while currentAxis < len(rotOrder):
        if rotOrder[currentAxis] == 'X':
            rotationMatrix = np.dot(getRotationAroundXAxis(radians(eulerAnglesDeg[0] )), rotationMatrix)
        elif rotOrder[currentAxis] == 'Y':
            rotationMatrix = np.dot(getRotationAroundYAxis(radians(eulerAnglesDeg[1])), rotationMatrix)
        elif rotOrder[currentAxis] == 'Z':
            rotationMatrix = np.dot(getRotationAroundZAxis(radians(eulerAnglesDeg[2])), rotationMatrix)
        currentAxis+=1

    return rotationMatrix   

#vectors represent columns
def getTranslationMatrix(translation):
    translationMatrix =np.array([[1.0, 0.0, 0.0, 0.0], 
                           [ 0.0, 1.0, 0.0, 0.0], 
                           [ 0.0, 0.0, 1.0, 0.0], 
                           [translation[0],translation[1], translation[2], 1.0]], np.float32)      
    return translationMatrix

 #vectors represent columns
def getScaleMatrix(scaleFactor):
    scaleMatrix =np.array([[scaleFactor, 0.0, 0.0, 0.0], 
                             [ 0.0, scaleFactor, 0.0, 0.0], 
                             [ 0.0, 0.0, scaleFactor, 0.0], 
                             [0.0,0.0, 0.0, 1.0]], np.float32) 
    return scaleMatrix


def get3DTransformationMatrixFromTranslationAndEulerAngles(translation,eulerAnglesDeg,rotationOrder):
    transformationMatrix= get3DRotationMatrix(eulerAnglesDeg,rotationOrder)
    transformationMatrix[3,:] = np.array([translation[0],translation[1],translation[2],1.0])
    return transformationMatrix
def extractRotationMatrixFromTransformation(transformation):
    return transformation[:3,:3]
def extractRotationMatrixAndTranslationFromTransformation(transformation):
    return transformation[:3,:3],vec3(transformation[3,0],transformation[3,1],transformation[3,2])

# 
# #http://web.mit.edu/2.05/www/Handout/HO2.PDF
# def get3DRotationMatrixZYX(eulerAnglesDeg):
#     rotationMatrix = get4x4IdentityMatrix()
# 
# 
#     return rotationMatrix  

#COPIED FROM http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
#order y,z,x heading,attidute, bank
def rotationMatrixToEulerYZX(rotationMatrix):
    " Assuming the angles are in radians."
    if rotationMatrix[1][0] > 0.998:# singularity at north pole
        heading = atan2(rotationMatrix[0][2],rotationMatrix[2][2])
        attitude = pi/2
        bank = 0;

    elif rotationMatrix[1][0] < -0.998: # singularity at south pole
        heading = atan2(rotationMatrix[0][2],rotationMatrix[2][2])
        attitude = -pi/2
        bank = 0

    else:
        heading = atan2(-rotationMatrix[2][0],rotationMatrix[0][0])
        bank = atan2(-rotationMatrix[1][2],rotationMatrix[1][1])
        attitude = asin(rotationMatrix[1][0])

    degBank = degrees(bank)
    degHeading = degrees(heading)
    degAttitude = degrees(attitude)
    return vec3(degBank,degHeading,degAttitude)#http://www.euclideanspace.com/maths/standards/index.htm


#http://planning.cs.uiuc.edu/node104.html
# order roll by gamma, pitch by beta, yaw by alpha and translate by x,y,u
def homogeneousTransformationMatrix3D(alpha,beta,gamma,x,y,z):
    m11 = cos(alpha)*cos(beta)
    m12 = cos(alpha)*cos(beta)*sin(gamma) - sin(alpha)*cos(gamma)
    m13 = cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*cos(gamma)
    m14 = x
    
    m21 = sin(alpha)*cos(beta)
    m22 = sin(alpha)*sin(beta)*sin(gamma) + cos(alpha) *cos(alpha)
    m23 = sin(alpha)*sin(beta)*cos(gamma) - cos(alpha) * sin(gamma)
    m24 = y
    
    m31 = -sin(beta)
    m32 = cos(beta) * sin(gamma)
    m33 = cos(beta) * cos(gamma)
    m34 = z
    
    m41 = 0
    m42 = 0
    m43 = 0
    m44 = 1
    matrix = np.array([m11,m12,m13,m14],[m21,m22,m23,m24],[m31,m32,m33,m34],[m41,m42,m43,m44])
    return matrix

def getQuaternionFromEuler(eulerAnglesDeg,rotOrder):
    currentAxis = 0
    quaternion= 0
    while currentAxis < len(rotOrder):
        if rotOrder[currentAxis] == 'X':
            tempQuaternion = quat()#CustomMath.Quaternion(0,0,0,0)
            tempQuaternion.fromAngleAxis(radians(eulerAnglesDeg[0]), vec3(1.0,0.0,0.0))
            if quaternion != 0:
                quaternion = quaternion*tempQuaternion #quaternion.multiply(tempQuaternion)
            else:
                quaternion = tempQuaternion
                #frame.rotationMatrix = np.dot(CustomMath.getRotationAroundXAxis(radians(deg)), frame.rotationMatrix)
        elif rotOrder[currentAxis] == 'Y':
            tempQuaternion = quat()#CustomMath.Quaternion(0,0,0,0)
            tempQuaternion.fromAngleAxis(radians(eulerAnglesDeg[1]), vec3(0.0,1.0,0.0))
            if quaternion != 0:
                    quaternion = quaternion*tempQuaternion #quaternion.multiply(tempQuaternion)
            else:
                    quaternion = tempQuaternion
                #frame.rotationMatrix = np.dot(CustomMath.getRotationAroundYAxis(radians(deg)), frame.rotationMatrix)
        elif rotOrder[currentAxis] == 'Z':
            tempQuaternion = quat()#CustomMath.Quaternion(0,0,0,0)
            tempQuaternion.fromAngleAxis(radians(eulerAnglesDeg[2]), vec3(0.0,0.0,1.0))
            if quaternion != 0:
                quaternion = quaternion*tempQuaternion #quaternion.multiply(tempQuaternion)
            else:
                quaternion = tempQuaternion 
        currentAxis +=1
    
    if quaternion != 0:
        return quaternion
    else:
        return quat(1,0,0,0)

def numpyArrayToCGkit4x4(array):
    return mat4(array.reshape(1,16).tolist()).transpose() 
def numpyArrayToCGkit3x3(array):
    return mat3(array.reshape(1,9).tolist()).transpose() 

def CGkitMatToNumpyArray3x3(matrix):
    return np.array(matrix.toList(), np.float32).reshape(3,3)

def CGkitMatToNumpyArray4x4(matrix):
    return np.array(matrix.toList(), np.float32).reshape(4,4)

def CGkitMat3x3ToNumpyArray4x4(matrix):
    tempMatrix = CGkitMat3ToCGKitMat4(matrix)#np.array(matrix.toList(), np.float32).reshape(3,3)
    return CGkitMatToNumpyArray4x4(tempMatrix)

def CGkitMat3ToCGKitMat4(matrix):
    # mat4(rotationMatrix[0,0],rotationMatrix[0,1],rotationMatrix[0,2],0.0,rotationMatrix[1,0],rotationMatrix[1,1],rotationMatrix[1,2],0.0,rotationMatrix[2,0],rotationMatrix[2,1],rotationMatrix[2,2],0.0, 0.0,0.0,0.0,1.0)
    return mat4(matrix[0,0],matrix[0,1],matrix[0,2],0.0,matrix[1,0],matrix[1,1],matrix[1,2],0.0,matrix[2,0],matrix[2,1],matrix[2,2],0.0,  0.0,0.0,0.0,1.0)

class Vector3D():
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        
    def fromCGKit(self,vector):
        self.x = vector.x
        self.y = vector.y
        self.z = vector.z    
    
    def __str__(self):
        return "Vector3D: "+str(self.x) +", "+ str(self.y) +", "+ str(self.z)
    
    def __add__(self, otherVector):
        tempVector = Vector3D()
        tempVector.x = self.x +otherVector.x
        tempVector.y = self.y +otherVector.y
        tempVector.z = self.z +otherVector.z
        return tempVector
    
    def __sub__(self,vector):
        tempVector = Vector3D()
        tempVector.x = self.x -vector.x
        tempVector.y = self.y -vector.y
        tempVector.z = self.z -vector.z
        return tempVector
       
       
    def __neg__(self):
        tempVector = Vector3D()
        tempVector.x = -self.x
        tempVector.y = -self.y
        tempVector.z = -self.z 
        return tempVector
         
    def __mul__(self,factor):
        tempVector = Vector3D()
        #typeOfObject = type(factor)
        if isinstance(factor,Vector3D):
          tempVector.x = self.x *factor.x
          tempVector.y = self.y *factor.y
          tempVector.z = self.z *factor.z   
            
        elif isinstance(factor,float)  or isinstance(factor,int):
          #print "test"
          tempVector.x = self.x *factor
          tempVector.y = self.y *factor
          tempVector.z = self.z *factor   
            
        return tempVector
    
    
    
    def matrixMultiplication(self, transformationMatrix):
        
        npVector = np.array([[self.x], 
                            [self.y], 
                            [self.z], 
                            [0.0]], np.float32)


        print("  ")
        print(npVector)
        print("to")
        transformedNPVector = np.dot(transformationMatrix,npVector)
       # print(transformedNPVector)
       # print("using")
        #print(transformationMatrix)
        print(transformedNPVector)
        print("  ")
        product = Vector3D()
        product.x = transformedNPVector[0][0]
        product.y = transformedNPVector[1][0]
        product.z = transformedNPVector[2][0]
        return product
        
    def normalize(self):
        magnitude = sqrt(self.x*self.x +self.y*self.y  +self.z*self.z)
        self.x = self.x/magnitude
        self.y = self.y/magnitude
        self.z = self.z/magnitude
        
    def cross(self,otherVector):
        crossProduct = Vector3D()
        crossProduct.x = self.y*otherVector.z - self.z* otherVector.y
        crossProduct.y = self.z*otherVector.x - self.x* otherVector.z
        crossProduct.z = self.x*otherVector.y - self.y* otherVector.x
        return crossProduct
    
    def getSquaredDistance(self,otherVector):
        return sqrt((pow((otherVector.x - self.x), 2)+pow((otherVector.y - self.y), 2)+pow((otherVector.z - self.z), 2)))

    def length(self):
        return  sqrt(self.x*self.x +self.y*self.y  +self.z*self.z) #sqrt(pow(self.x, 2)+pow(self.y, 2)+pow( self.z, 2))
  
  
  
  
      
#copied entirely from these websites and forum threads
#http://content.gpwiki.org/index.php/OpenGL:Tutorials:Using_Quaternions_to_represent_rotation
# http://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion 
#see http://run.usc.edu/cs520-s12/assign2/p245-shoemake.pdf and http://web.mit.edu/2.998/www/QuaternionReport1.pdf page 20 for the background of quaternion based rotation
#https://theory.org/software/qfa/writeup/node12.html
#http://en.wikipedia.org/wiki/Slerp
#and http://www.java-forum.org/spiele-multimedia-programmierung/121904-slerp-quaternion-berechnen.html
class Quaternion():
    
    #takes half radian of desired angle as w 
    def __init__(self, x,y,z,w):
        
        self.x = x #rotation vector x
        self.y = y #rotation vector y
        self.z = z #rotation vector z
        self.w = w #half rotation angle in radians
        
        
        self.TOLERANCE = 0.00001
        self.PIOVER180 = pi/180.0


        
        self.normalize()
      
    
    def __str__(self):
        return "Quaternion: "+str(self.x) +", "+ str(self.y) +", "+ str(self.z) +", "+str(self.w) 
      
    #note angle  in radians
    def fromAxisAngle(self,angle,vector):
        vector.normalize()
        theta = angle/2
        sintheta = sin(theta)
        
        self.x = sintheta*vector.x
        self.y = sintheta*vector.y
        self.z = sintheta*vector.z
        self.w = cos(theta)
        self.normalize()
        
    def toAxisAngle(self):
        scale = self.x*self.x+self.y+self.y+self.z+self.z
        axis = Vector3D()
        axis.x = self.x/scale
        axis.y = self.y/scale
        axis.z = self.x/scale
        angle = acos(self.w)*2.0 
        return axis,angle
    
    
    
    
    
    
    
    #does not work yet
    #takes radians
    #source http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm
    def fromEulerYawPitchRoll(self,yaw,pitch,roll):
                
        c1 = cos(yaw/2.0);
        s1 = sin(yaw/2.0);
        c2 = cos(pitch/2.0);
        s2 = sin(pitch/2.0);
        c3 = cos(roll/2.0);
        s3 = sin(roll/2.0);
        c1c2 = c1*c2;
        s1s2 = s1*s2;
    
        self.x =c1c2*s3 + s1s2*c3;
        self.y =s1*c2*c3 + c1*s2*s3;
        self.z =c1*s2*c3 - s1*c2*s3;
        self.w =c1c2*c3 - s1s2*s3;
        
        return
    
    #does not work yet
    #takes degrees
    #roll z pitch x, yaw y
    def fromEuler(self,pitch,yaw,roll):#,sequence
        

        p = pitch * self.PIOVER180 / 2.0;
        y = yaw * self.PIOVER180 / 2.0;
        r = roll * self.PIOVER180 / 2.0;
        sinp = sin(p)
        siny = sin(y)
        sinr = sin(r)
        cosp = cos(p)
        cosy = cos(y)
        cosr = cos(r)
        #there are six different options in the sequence
        
        self.x = sinr * cosp * cosy - cosr * sinp * siny
        self.y = cosr * sinp * cosy + sinr * cosp * siny
        self.z = cosr * cosp * siny - sinr * sinp * cosy
        self.w = cosr * cosp * cosy + sinr * sinp * siny
        self.normalize()
   
  
    #http://content.gpwiki.org/index.php/OpenGL:Tutorials:Using_Quaternions_to_represent_rotation
    def toMatrix(self):
        x2 = self.x*self.x
        y2 = self.y*self.y
        z2 = self.z*self.z
        xy = self.x*self.y
        xz = self.x*self.z
        yz = self.y*self.z
        wx= self.w*self.x
        wy= self.w*self.y
        wz= self.w*self.z
                        #note columns and rows are  switched
                        #first row, second row, third row, fourth row
        return np.array([[ 1.0-2.0*(y2+z2),   2.0*(xy+wz),      2.0*(xz-wy)     ,0.0],#first column
                         [2.0*(xy-wz),        1.0-2.0 *(x2+z2), 2.0*(yz+wx)     ,0.0],#second column
                         [ 2.0*(xz+wy),       2.0*(yz-wx),      1.0-2.0*(x2+y2)  ,0.0],#third column
                         [0.0,                0.0,              0.0,              1.0]],np.float32)#fourth column
    
     
    def length(self):
        return sqrt(self.w*self.w+self.x*self.x +self.y*self.y  +self.z*self.z) # sqrt(pow(self.w, 2)+pow(self.x, 2)+pow(self.y, 2)+pow( self.z, 2))
    
    def length2(self):
        return self.w*self.w+self.x*self.x +self.y*self.y  +self.z*self.z
    

    def conjugate(self):
        x = -self.x
        y = -self.y
        z = -self.z
        return Quaternion(x,y,z,self.w)
    
    # unit quaternion can be described as: |q|*|q| = 1 #http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    def normalize(self):
        mag2 = sqrt(self.length2())
        if mag2 > self.TOLERANCE and mag2-1.0 > self.TOLERANCE:
            mag = sqrt(mag2)
            self.w = self.w/mag
            self.x = self.x/mag
            self.y = self.y/mag
            self.z = self.z/mag
    
    
#and http://www.java-forum.org/spiele-multimedia-programmierung/121904-slerp-quaternion-berechnen.html
#     def multiply(self,q):
#         w1, x1, y1, z1 = self.w,self.x,self.y,self.z
#         w2, x2, y2, z2 = q.w,q.x,q.y,q.z
#         w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#         x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#         y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#         z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#         return Quaternion(x,y,z,w)
    
    #http://content.gpwiki.org/index.php/OpenGL:Tutorials:Using_Quaternions_to_represent_rotation
#     def multiply(self,rq):#x y z w
#         return Quaternion(
#                           self.w * rq.x + self.x * rq.w + self.y * rq.z - self.z * rq.y,
#                           self.w * rq.y + self.y * rq.w + self.z * rq.x - self.x * rq.z,
#                           self.w * rq.z + self.z * rq.w + self.x * rq.y - self.y * rq.x,
#                           self.w * rq.w - self.x * rq.x - self.y * rq.y - self.z * rq.z
#                           )
   #http://wiki.delphigl.com/index.php/Quaternion
    def multiply(self,rq):
        result = Quaternion(0,0,0,0)
        result.w = self.w * rq.w - self.x * rq.x  - self.y * rq.y - self.z * rq.z
        result.x = self.w * rq.x + self.x * rq.w  + self.y * rq.z   - self.z * rq.y
        result.y = self.w * rq.y - self.x * rq.z      + self.y * rq.w      + self.z * rq.x
        result.z = self.w * rq.z + self.x * rq.y      - self.y * rq.x+ self.z * rq.w
        return result
       
   
    def dot(self,q):
        return self.w+q.w+self.x*q.x+self.y*q.y+self.z*q.z
      
    #R(v) = q*[0,v]*q1^-1
    def rotate3DVector(self,vector):
        vector.normalize()
        q = Quaternion(vector.x,vector.y,vector.z,0.0)
        
        tempQ = self.multiply(q)
        tempQ = tempQ.multiply(self.conjugate())
        
        vector = Vector3D()
        vector.x = tempQ.x
        vector.y = tempQ.y
        vector.z = tempQ.z
        return vector
    
    #http://wiki.delphigl.com/index.php/Quaternion
    def apply(self,vector):
        result = Vector3D();
        v0 = vector.x
        v1 = vector.y
        v2 = vector.z
        a00 = self.w * self.w 
        a01 = self.w  * self.x
        a02 =self.w* self.y
        a03 =self.w * self.z
        a11 = self.x* self.x
        a12 = self.x * self.y
        a13 = self.x * self.z
        a22 = self.y* self.y
        a23 = self.y * self.z
        a33 = self.z* self.z
        result.x = v0 * (+a00 + a11 - a22 - a33)                + 2 * (a12 * v1 + a13 * v2 + a02 * v2 - a03 * v1);
        result.y = v1 * (+a00 - a11 + a22 - a33)                + 2 * (a12 * v0 + a23 * v2 + a03 * v0 - a01 * v2);
        result.z = v2 * (+a00 - a11 - a22 + a33)   + 2 * (a13 * v0 + a23 * v1 - a02 * v0 + a01 * v1);
        return result;
    
    #http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
    #https://theory.org/software/qfa/writeup/node12.html
    def slerp(self,q,t):
        theta = self.dot(q)
        stheta = sin(theta)
        result = (sin(1-t)*theta/stheta)*self + (sin(t*theta)/stheta)*q
        result.normalize()
        return result


#COPIED FROM http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
# q1 can be non-normalised quaternion #
def quaternionToEuler(q1):
    sqw = q1.w*q1.w
    sqx = q1.x*q1.x
    sqy = q1.y*q1.y
    sqz = q1.z*q1.z
    unit = sqx + sqy + sqz + sqw# if normalised is one, otherwise is correction factor
    test = q1.x*q1.y + q1.z*q1.w
    if test > 0.499*unit:#singularity at north pole
        heading = 2 * atan2(q1.x,q1.w)
        attitude = pi/2
        bank = 0;
    elif test < -0.499*unit:# singularity at south pole
        heading = -2 * atan2(q1.x,q1.w)
        attitude = -pi/2
        bank = 0;

    else:
        heading = atan2(2*q1.y*q1.w-2*q1.x*q1.z , sqx - sqy - sqz + sqw)
        attitude = asin(2*test/unit)
        bank = atan2(2*q1.x*q1.w-2*q1.y*q1.z , -sqx + sqy - sqz + sqw)  
    return vec3(degrees(bank),degrees(heading),degrees(attitude))#http://www.euclideanspace.com/maths/standards/index.htm

#     
#     test = q1.x*q1.y + q1.z*q1.w;
#     if test > 0.499 : #singularity at north pole
#         heading = 2 * atan2(q1.x,q1.w)
#         attitude = pi/2
#         bank = 0
# 
#     elif test < -0.499 : #singularity at south pole
#         heading = -2 * atan2(q1.x,q1.w)
#         attitude = - pi/2
#         bank = 0
#         return
#     else:
#         sqx = q1.x*q1.x
#         sqy = q1.y*q1.y
#         sqz = q1.z*q1.z
#         heading = atan2(2*q1.y*q1.w-2*q1.x*q1.z , 1 - 2*sqy - 2*sqz)
#         attitude = asin(2*test)
#         bank = atan2(2*q1.x*q1.w-2*q1.y*q1.z , 1 - 2*sqx - 2*sqz)
#     return vec3(degrees(bank),degrees(heading),degrees(attitude))#http://www.euclideanspace.com/maths/standards/index.htm
    

def quaternionToEuler2(q):
    rad,axis = q.toAngleAxis()
    print axis
    if axis.length() >0:
        axis = axis.normalize()
    radAngles = axisAngleToEuler(axis.x,axis.y,axis.z,rad)
    degAngles = vec3(degrees(radAngles.x),degrees(radAngles.y),degrees(radAngles.z))
    return degAngles

#http://cgkit.sourceforge.net/doc2/mat3.html
def quaternionToEulerZXY(q1):
    tempMatrix = q1.toMat3()
    vector = tempMatrix.toEulerZXY()
    print vector[0],vector[1],vector[2] 
    return vec3(degrees(vector[0]),degrees(vector[1]),degrees(vector[2]))

#http://cgkit.sourceforge.net/doc2/mat3.html
def matrixToEuler(matrix,rotOrder):
    '''
    returns the angles in a three dimensional vector where the x/y/z component of the vector represents the rotation around the x/y/z axis independent of the rotation order
    rotation order is needed to extract the correct angles
    when the angles are converted into a matrix or a quaternion or saved as a bvh format string the rotation order needs to be applied
    TODO use better and faster code by Ken Shoemake in Graphic Gems 4, p.222
    http://thehuwaldtfamily.org/jtrl/math/Shoemake,%20Euler%20Angle%20Conversion,%20Graphic%27s%20Gems%20IV.pdf
    https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf
    '''
    #result = vec3(0,0,0)
    if rotOrder[0] =='X':
          if rotOrder[1] =='Y':
              vector = matrix.toEulerXYZ()
              #result = vec3(degrees(vector[0]),degrees(vector[1]),degrees(vector[2])) 
          elif rotOrder[1] =='Z':
              vector = matrix.toEulerXZY()
              #result = vec3(degrees(vector[0]),degrees(vector[2]),degrees(vector[1])) 
    elif rotOrder[0] =='Y':
        if rotOrder[1] =='X':
             vector = matrix.toEulerYXZ()
             #result = vec3(degrees(vector[1]),degrees(vector[0]),degrees(vector[2])) 
        elif rotOrder[1] =='Z':
             vector = matrix.toEulerYZX()  
             #result = vec3(degrees(vector[2]),degrees(vector[0]),degrees(vector[1]))  
    elif rotOrder[0] =='Z': 
        if rotOrder[1] =='X':
            vector = matrix.toEulerZXY()  
            #result = vec3(degrees(vector[1]),degrees(vector[2]),degrees(vector[0]))  
        elif rotOrder[1] =='Y': 
            vector = matrix.toEulerZYX()  
            #result = vec3(degrees(vector[2]),degrees(vector[1]),degrees(vector[0])) 
    #print vector[0],vector[1],vector[2] 
    return vec3(degrees(vector[0]),degrees(vector[1]),degrees(vector[2])) # result 

def matrixToEulerOrdered(matrix,rotOrder):
    '''
    returns a vector of angles in the order in which they are supposed to be applied, in order to apply it on the correct axis the rotation order is needed
    '''
    result = vec3(0,0,0)
    if rotOrder[0] =='X':
          if rotOrder[1] =='Y':
              vector = matrix.toEulerXYZ()
              result = vec3(degrees(vector[0]),degrees(vector[1]),degrees(vector[2])) 
          elif rotOrder[1] =='Z':
              vector = matrix.toEulerXZY()
              result = vec3(degrees(vector[0]),degrees(vector[2]),degrees(vector[1])) 
    elif rotOrder[0] =='Y':
        if rotOrder[1] =='X':
             vector = matrix.toEulerYXZ()
             result = vec3(degrees(vector[1]),degrees(vector[0]),degrees(vector[2])) 
        elif rotOrder[1] =='Z':
             vector = matrix.toEulerYZX()  
             result = vec3(degrees(vector[2]),degrees(vector[0]),degrees(vector[1]))  
    elif rotOrder[0] =='Z': 
        if rotOrder[1] =='X':
            vector = matrix.toEulerZXY()  
            result = vec3(degrees(vector[1]),degrees(vector[2]),degrees(vector[0]))  
        elif rotOrder[1] =='Y': 
            vector = matrix.toEulerZYX()  
            result = vec3(degrees(vector[2]),degrees(vector[1]),degrees(vector[0])) 
    #print vector[0],vector[1],vector[2] 
    return result 
    
def quaternionToEuler3(q,rotOrder):
    tempMatrix = q.toMat3()
    return matrixToEuler(tempMatrix,rotOrder)

#http://cgkit.sourceforge.net/doc2/mat3.html
#uses runtime compilation so it is slow
def quaternionToEuler4(q,rotOrder):
    tempMatrix = q.toMat3()
    vector = eval("tempMatrix.toEuler"+rotOrder[0]+rotOrder[1]+rotOrder[2]+"()")
    print vector[0],vector[1],vector[2] 
    return vec3(degrees(vector[0]),degrees(vector[1]),degrees(vector[2]))

#COPIED FROM http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/
def axisAngleToEuler(x,y,z,rad):
    s=sin(rad)
    c=cos(rad)
    t=1.0-c
   
    if (x*y*t + z*s) > 0.998: # north pole singularity detected
        heading = 2.0*atan2(x*sin(rad/2),cos(rad/2))
        attitude = pi/2.0
        bank = 0.0
        #return
  
    elif (x*y*t + z*s) < -0.998: #south pole singularity detected
        heading = -2.0*atan2(x*sin(rad/2),cos(rad/2))
        attitude = -pi/2.0
        bank = 0.0
        #return
    else:
        heading = atan2(y * s- x * z * t , 1 - (y*y+ z*z ) * t)
        attitude = asin(x * y * t + z * s) 
        bank = atan2(x * s - y * z * t , 1 - (x*x + z*z) * t)
    return vec3(bank,heading,attitude)#http://www.euclideanspace.com/maths/standards/index.htm



def globalToLocalTransformation(globalTransformation,coordinateSystemTransformation):
    '''
    bring global transformation into local transformation
    see source: http://www.gamedev.net/topic/611614-get-relative-matrix-from-parent/
    '''
    localTransformation = np.dot(globalTransformation, np.linalg.inv(coordinateSystemTransformation))
    return localTransformation
      
# def getQuaternionFromEulerAngles(eulerAnglesDeg,rotOrder):
#     #rotationMatrix = get4x4IdentityMatrix()
#     if rotOrder[0] == 'X':
#         q = Quaternion(radians(eulerAnglesDeg.x/2.0),1.0,0.0,0.0)
#     elif rotOrder[0] == 'Y':
#         q = Quaternion(radians(eulerAnglesDeg.y/2.0),0.0,1.0,0.0)
#     elif rotOrder[0] == 'Z':
#         q = Quaternion(radians(eulerAnglesDeg.z/2.0),0.0,0.0,1.0)
#     currentAxis = 1
#     while currentAxis < len(rotOrder):
#         if rotOrder[currentAxis] == 'X':
#             q = q.multiply(Quaternion(radians(eulerAnglesDeg.x/2.0),1.0,0.0,0.0))# np.dot(getRotationAroundXAxis(radians(eulerAnglesDeg.x )), rotationMatrix)
#         elif rotOrder[currentAxis] == 'Y':
#             q = q.multiply(Quaternion(radians(eulerAnglesDeg.y/2.0),0.0,1.0,0.0))#rotationMatrix = np.dot(getRotationAroundYAxis(radians(eulerAnglesDeg.y)), rotationMatrix)
#         elif rotOrder[currentAxis] == 'Z':
#             q = q.multiply(Quaternion(radians(eulerAnglesDeg.z/2.0),0.0,0.0,1.0))#rotationMatrix = np.dot(getRotationAroundZAxis(radians(eulerAnglesDeg.z)), rotationMatrix)
#         currentAxis+=1
# 
#     return q   
                          
                          
                          

    
xAxis  =Vector3D()
xAxis.x = 1.0
xAxis.y = 0.0
xAxis.z = 0.0


yAxis  =Vector3D()
yAxis.x = 0.0
yAxis.y = 1.0
yAxis.z = 0.0

zAxis  =Vector3D()
zAxis.x = 0.0
zAxis.y = 0.0
zAxis.z = 1.0

identityMatrix = get4x4IdentityMatrix()
