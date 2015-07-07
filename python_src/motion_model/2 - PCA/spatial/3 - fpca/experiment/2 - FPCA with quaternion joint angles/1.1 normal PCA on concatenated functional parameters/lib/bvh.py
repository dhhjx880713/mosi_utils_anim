# -*- coding: utf-8 -*-

"""
bvh.py
======

@author: mamauer

"""
from math import atan2, pi, degrees, asin
        
class BVHWriter(object):
    """Write BVH files 

    NOTE: Currently not saves any skeleton. Copy the default INTERACT 
    skeleton into the created file.
    
    """

    def __init__(self, data, filename, is_quaternion=True):
        if filename[-4:] == '.bvh':
            self.filename = filename
        else:
            self.filename = filename + '.bvh'
        self.data = data

        if not is_quaternion:
            self.frames = data
            self.write()
        else:
            self.frames = []
            for frame in data:
                euler_frame = frame[:3]     # copy root
                for i in xrange(3, len(frame), 4):
                    eulers = self.quaternionToEuler(frame[i:i+4])
                    for e in eulers:
                        euler_frame.append(e)
                self.frames.append(euler_frame)
            self.write()
            
    #COPIED FROM http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/
    # q1 can be non-normalised quaternion #
    def quaternionToEuler(self, q1):
        w, x, y, z = q1
        sqw = w*w
        sqx = x*x
        sqy = y*y
        sqz = z*z
        unit = sqx + sqy + sqz + sqw# if normalised is one, otherwise is correction factor
        test = x*y + z*w
        if test > 0.499*unit:#singularity at north pole
            heading = 2 * atan2(x,w)
            attitude = pi/2
            bank = 0;
        elif test < -0.499*unit:# singularity at south pole
            heading = -2 * atan2(x,w)
            attitude = -pi/2
            bank = 0;
    
        else:
            heading = atan2(2*y*w-2*x*z , sqx - sqy - sqz + sqw)
            attitude = asin(2*test/unit)
            bank = atan2(2*x*w-2*y*z , -sqx + sqy - sqz + sqw)  
        return [degrees(bank),degrees(heading),degrees(attitude)]#http://www.euclideanspace.com/maths/standards/index.htm
    
    
    def write(self):
        """ Write all frames to file """
        fp = open(self.filename, 'w+')
        fp.write("{SKELETON STRUCTUR}\n")
        fp.write("MOTION\n")
        fp.write("Frames: %d\n" % len(self.frames))
        fp.write("Frame Time: 0.013889\n")
        for frame in self.frames:
            fp.write(' '.join([str(i) for i in frame]))
            fp.write('\n')
        fp.close()
    
def main():
    testdata = [
        [1, 90, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 90, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [3, 90, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [4, 90, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [5, 90, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [6, 90, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [7, 90, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [8, 90, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],    
    ]
    
    f = 'test.bvh'
    BVHWriter(testdata, f)
    

if __name__=='__main__':
    main()