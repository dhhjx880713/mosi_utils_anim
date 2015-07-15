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

There are two interfaces:

mg_pipeline_interface.py - A simple command line interface without arguments. It automatically loads the latest input file in the input directory
specified in config\service.json.

mg_rest_interface.py - Starts a web server that provides the REST following interface localhost:port/runmorphablegraphs. It can be called using a HTTP POST 
message with a string containing the input file as message body.


Current limitations:
 - Only one trajectory constraint can be applied


