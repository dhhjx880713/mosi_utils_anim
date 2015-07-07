For automated keyframe detection (currently implemented for walk / carry) use the segmentation.py script. 
The following steps have to be performed:

* Change action name in main - function of segmentation.py
* Change filter_tpose parameter in main - function of segmentation.py (whether the tpose has to be filtered automatically)
* (optional) Change the mask parameter if you only want to process a subset of the files (e.g. '*4.bvh' for actor 4 only)
* run the script

For manual keyframe segmentation, the following steps have to be performed:

* Create a new .txt file in the "manual_key" Folder. (See the other files for the data structure)
* Convert the .txt file to a json file. Therefor, perform the following steps:
	* change the "fileprefix" variable in "lib/keyframes.py" to the name of the .txt file (without the ending)
	* Run the script
* Copy the json file to the "1 - segmentation" Folder and rename it to keyframes.json
* Continue with the steps for automated keyframe detection