import numpy as np
from ..utilities import load_json_file, write_to_json_file


class Panim(object):
    """ A customized animation data format for point cloud data. It is also used in Unity.
    """

    def __init__(self):
        self.motion_data = None
        self.has_skeleton = False
        self.skeleton = None
    
    def load(self, filename):
        """
        
        Arguments:
            filename {str} -- path to .panim file
        """
        json_data = load_json_file(filename)
        self.motion_data = json_data['motion_data']
        self.has_skeleton = json_data['has_skeleton']
        self.skeleton = json_data['skeleton']
    
    def save(self, saveFilename):
        """
        
        Arguments:
            saveFilename {str} -- save path
        """
        output_data = {'motion_data': self.motion_data,
                       'has_skeleton': self.has_skeleton,
                       'skeleton': self.skeleton
        }
        write_to_json_file(saveFilename, output_data)
    
    def setSkeleton(self, ske_description):
        """list of joints
        
        Arguments:
            ske_description {list} -- a list of dictionary for joint information
        """
        self.skeleton = ske_description
        self.has_skeleton = True

    def setMotionData(self, motion_data):
        """
        
        Arguments:
            motion_data {numpy.array3d} -- n_frames * n_joints * 3
        """
        self.motion_data = np.asarray(motion_data).tolist()
        print(type(self.motion_data))

    def convert_to_unity_format(self, scale=1.0):
        """ 
        Convert motion_data from numpy array to dictionary for Unity loading. The coordinate system is flipped because Unity use left-hand coordinate system
        """
        output_frames = []
        for frame in self.motion_data:
            world_pos = []
            for point in frame:
                world_pos.append({'x': -point[0] * scale,
                                  'y': point[1] * scale,
                                  'z': point[2] * scale})
                output_frame = {'WorldPos': world_pos}
        output_frames.append(output_frame)
        ## update motion_data
        self.motion_data = output_frames                     
