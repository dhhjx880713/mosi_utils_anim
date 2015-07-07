Animation server implementation and XML3D client for the Daimler and Electrolux pilotcases

Developers:
Erik Herrmann (based on work done by Jan Sutter) (DFKI GmbH) 
The pilot case scenes are property of Daimler AG and Electrolux AB.
The scenes were converted with the help of LMS and Daimler
The final conversion was done using the XML3D exporter for Blender written by Kristian Sons (DFKI GmbH) 


Python Dependencies:
 - Python 2.7.6.4 64 bit - WinPython is recommended:  http://sourceforge.net/projects/winpython/files/WinPython_2.7/2.7.6.4/
 - NumPy 1.8.1
 - cgkit 2.0.0
 - watchdog 0.8.3
 - tornado 4.0.2
 - pathtools 0.1.2


Instructions:
To start the server run main.py with the path that will contain the output bvh files
Examples: 
python main.py "."
python main.py "C:\output"

Animations are loaded and displayed automatically when a new bvh file is moved to the observed directory.
The web server and the animation are accessible using port 8889. 

After the server has been started the XML3D scenes can be displayed by opening the URL "localhost:8889" with a web browser supporting WebGL. 
Chrome is recommended for speed but Firefox is also supported.
The Electrolux scene is located at "localhost:8889/pilotcase_elux.xhtml" 
The Daimler pilot case is located at "localhost:8889/pilotcase_daimler.xhtml". 


Troubleshooting:
- Make sure port 8889 is not already used by a different application. A different port cannot be used 
  because the port needs to be hard coded into the XML3D clients.
- Older versions of XML3D scripts will lead to problems. Make sure the cache of 
  your web browser is empty before loading the XML3D client for the first time. 
   
