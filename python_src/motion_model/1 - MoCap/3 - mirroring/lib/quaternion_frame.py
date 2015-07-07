# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:10:21 2014

@author: erhe01

"""
import collections
from math import radians

from cgkit.cgtypes import vec3,quat


class QuaternionFrame(collections.OrderedDict):
    """OrderedDict that contains data for a quaternion frame"""

    def __init__(self, bvh_reader, frame_number,filter_values = False):
        """Reads an animation frame from a BVH file and fills the list class
           with quaternions of the skeleton nodes

        Parameters
        ----------

         * bvh_reader: BVHReader
        \t BVH data structure read from a file
        * frame_number: Integer
        \t animation frame number that gets extracted
        * filter_values: Bool
        \t enforce a unique rotation representation
        
        """
        quaternions= \
            self._get_all_nodes_quaternion_representation(bvh_reader, frame_number,filter_values)
        collections.OrderedDict.__init__(self, quaternions)
          
    def _get_quaternion_from_euler(self,euler_angles,rotation_order, filter_values = True):
        '''Converts a rotation represented using euler angles into a 
        quaternion (w,x,y,z) using the cgkit library	
              
        Parameters
        ----------
    
         * euler_angles: Iteratable
        \t  an ordered list of length 3 with the rotation angles given in degrees  
        * rotation_order: Iteratable
        \t a list that specifies the rotation axis corresponding to the values in euler_angles 
        * filter_values: Bool
        \t enforce a unique rotation representation
        
        '''
        current_axis = 0
        quaternion= 0
        while current_axis < len(rotation_order):
            rad = radians(euler_angles[current_axis])
            if rotation_order[current_axis] == 'X':
                temp_quaternion = quat()
                temp_quaternion.fromAngleAxis(rad, vec3(1.0,0.0,0.0))
                if quaternion != 0:
                    quaternion = quaternion*temp_quaternion
                else:
                    quaternion = temp_quaternion
            elif rotation_order[current_axis] == 'Y':
                temp_quaternion = quat()
                temp_quaternion.fromAngleAxis(rad, vec3(0.0,1.0,0.0))
                if quaternion != 0:
                    quaternion = quaternion*temp_quaternion 
                else:
                    quaternion = temp_quaternion
            elif rotation_order[current_axis] == 'Z':
                temp_quaternion = quat()
                temp_quaternion.fromAngleAxis(rad, vec3(0.0,0.0,1.0))
                if quaternion != 0:
                    quaternion = quaternion*temp_quaternion 
                else:
                    quaternion = temp_quaternion 
            current_axis +=1
        if filter_values :# see http://physicsforgames.blogspot.de/2010/02/quaternions.html
            dot = quaternion.x + quaternion.y + quaternion.z + quaternion.w#dot product with [1,1,1,1]
            if dot < 0:#flip
                quaternion = -quaternion
        return quaternion.w,quaternion.x ,quaternion.y,quaternion.z
			
 
    def _get_quaternion_representation(self, bvh_reader, node_name, frame_number,filter_values = False):
        """Returns the rotation for one node at one frame of an animation as
           a quaternion

        Parameters
        ----------

        * node_name: String
        \tName of node
        * bvh_reader: BVHReader
        \t BVH data structure read from a file
        * frame_number: Integer
        \t animation frame number that gets extracted
        * filter_values: Bool
        \t enforce a unique rotation representation
        

        """        
        euler_angles = bvh_reader.get_angles(node_name)[frame_number]
        if node_name.startswith("Bip"):
           euler_angles = [0, 0, 0]     # Set Fingers to zero
          
        rotation_order =('X','Y','Z')# hard coded for now
        return self._get_quaternion_from_euler(euler_angles,rotation_order,filter_values)

    def _get_all_nodes_quaternion_representation(self, bvh_reader, frame_number,filter_values = False):
        """Returns dictionary of all quaternions for all nodes except leave nodes
           Note: bvh_reader.node_names may not include EndSites

        Parameters
        ----------

         * bvh_reader: BVHReader
        \t BVH data structure read from a file
        * frame_number: Integer
        \t animation frame number that gets extracted
        * filter_values: Bool
        \t enforce a unique rotation representation
        
        """
        
        for node_name in bvh_reader.node_names:
            # simple fix for ignoring finger joints.
            # NOTE: Done in _get_quaternion_representation(...) now
            #if not node_name.startswith("Bip"):
            yield node_name, self._get_quaternion_representation(bvh_reader,
                                                                 node_name,
                                                                 frame_number,
                                                                 filter_values)
def main():
   
    from bvh import BVHReader
    filepath = "test/walk_001_1_rightStance_86_128.bvh"
    bvh_reader = BVHReader(filepath)
    quat_frame = QuaternionFrame(bvh_reader,0,True)
    print quat_frame.values()

if __name__ == '__main__':
    main()

