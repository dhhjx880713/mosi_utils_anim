Morphable Graphs implementation

Python Dependencies:
 - Python 2.7.6.4 64 bit - WinPython is recommended, which already includes most of the other dependencies:  http://sourceforge.net/projects/winpython/files/WinPython_2.7/2.7.6.4/
 - NumPy 1.8.1
 - SciPy 0.15.1
 - scikit-learn 0.14.1
 - cgkit 2.0.0
 - rpy2 2.4.4 (requires R 3.0 or higher)
 - GPy 0.6.1 - can be found in the Dependencies directory
 - fk3.py + BOOST_PYTHON-VC110-MT-1_55.DLL  (C++ implementation of analytic forward kinematics + jacobian by LMS) - can be found in the Dependencies directory

R-Dependencies
 - R3.1.2 (http://cran.r-project.org/bin/windows/base/)
 - fda 

Developers:
Han Du*, Markus Mauer°, Erik Herrmann*, Martin Manns°, Fabian Rupp°
°Daimler AG 
*DFKI GmbH

Instructions:

A simple command line interface to run the complete pipeline is found in 
mg_pipeline_interface.py. This script needs to be called from the command line
with three arguments
  - input filename
  - output directory
  - output filename

There are two versions of the algorithm that differ in the constraints generation

synthesize_motion.py -- First generates a graph walk with constraints for the complete input file 
                        that is sequentially converted into a motion
synthesize_motion_v2.py -- Sequentially converts the input file into a motion without generating 
                            the constraints on the fly based on the previous steps

Both modules contain the function run_pipeline that takes a json file as input as defined in the interface document. 
 Additionally, algorithm settings can be specified, debug output can be activated and a maximum number of motion primitives to be generated 
can be specified. It is recommended to use version 2. 

Both algorithms are called by the method synthesize_motion of the class 
ControllableMorphableGraph in controllable_morphable_graph.py, which is intended as interface
for further integration.

In the algorithm settings different steps of the algorithm can be deactivated for debugging, 
however all steps should be activated.
The result of the pipeline is a BVH file that is exported to the specified output directory 
named after the session value in the input file.

If the specified constraints are unreachable the motion synthesis is aborted,
 the result is however still exported as a BVH file so it is possible to inspect 
the result for debugging purposes.

Note: the Morphable Graph data structure is automatically constructed from the data 
found in the directories "data\3 - Motion primitives\" and "data\4 - Transition model\". 
If there are any problems running the script make sure that the directory structure from the repository 
is used and all directores are updated.

At least 16 GB of memory are recommended due to the size of the
transition models of up to 6 GB.

Current limitations:
 - Only one trajectory constraint can be applied


