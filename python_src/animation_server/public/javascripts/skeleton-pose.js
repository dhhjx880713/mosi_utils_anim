//classes that form a wrapper around a skinning xlfow node
//Author: Erik Herrmann
(function() {

    var quat = XML3D.math.quat;
    var vec3 = XML3D.math.vec3;
    var vec4 = XML3D.math.vec4;
    var mat4 = XML3D.math.mat4;

	
	//######################################################################
	/**
	 * Skeleton joint class based on the Python classes but using XML3D.math classes for transformations
	 * translationData and rotationData need to refer to the corresponding values in a Skinning Xflow node 
	 **/
	var SkeletonJoint = function(){
	
		this.name = "";//has to be set after construction
		this.parent = null;
		this.parentName = "";
		this.children = new Array();//array of joint names
		this.boneIndex = -1;
		this.translationIndex = -1;//index in the translation input vector of the xflow node
		this.rotationIndex = -1;//index in the rotation input vector of the xflow node
		this.pose = null;
		this.translationData = null;
		this.rotationData = null;
		this.observers = new Array();	//transform ids callback new XML3DDataObserver(callback);
		
		//this.observers["a"] = "test";
		//delete this.observers["a"];

	},j = SkeletonJoint.prototype;
	
    
	j.addChild = function(name,joint){
		this.children[name] = joint;
	};
	
	j.addTransformObserver = function(id,callback){
		console.log("add observer",id);
		this.observers[id] = callback;//XML3D.base.resourceManager.getAdapter(node, XML3D.data);// node
		
	};
	j.removeTransformObserver = function(id){
		delete this.observers[id];
	
		
	};
	j.updateObservers = function(pose){
		/**
		 * sends the current transformation to all observers. is called by the SkeletonPose class after an update the the skinning.
		 * @param {pose} optional pose obtained from skinning observer of the skeleton pose class, if the pose is not provided the transformation is calculated recursively
		 **/
		if (pose == null){
			var translation = this.getAbsolutePosition();
			var quaternion = [0,0,0,1];
		}else{
			//transformationIndex = boneIndex*16;
			var jointPose = new XML3DMatrix();
			
			XML3DMatrix.prototype.set.apply(jointPose,pose.subarray(0+16*this.boneIndex , 16+16*this.boneIndex));
			var tVector= jointPose.translation();
			var rVector= jointPose.rotation();
			var translation = [tVector.x,tVector.y,tVector.z];
			var quaternion = [rVector.x,rVector.y,rVector.z,rVector.w];
		/*
			 * @param {Number} x X component
			 * @param {Number} y Y component
			 * @param {Number} z Z component
			 * @param {Number} w W component
			 * */
			//var quaternion = [1.4,1.0,0.2,0.1];
			//var translation = pose.subarray(0+16*boneIndex , 16+16*boneIndex);//[transformationIndex];
		}
		
		for (id in this.observers){
			//console.log("update transform "+this.name);
			//console.log("update this",id,translation);
			
			//console.log(translation);
			this.observers[id](translation,quaternion);//use callback
			//this.observers[id].translation.x =translation[0];
			//this.observers[id].translation.y =translation[1];
			//this.observers[id].translation.z =translation[2];
			
			
			//var transform = $('#'+this.observers[id])[0];
			//console.log(this.observers[id]);
			//this.observers[id].setAttribute("translation",translation[0]+" "+translation[1]+" "+translation[2]);
			
			//$("#t_walk_goal")[0].translation.x
			//this.observers.apply();
		}
		

			
		
	};
	
	j.getTranslationString = function(){
		var translationString= "";
		var translation = this.getTranslation();
		translationString+=translation[0] +" "+translation[1]+" "+translation[2]+" ";
		return translationString
		
	};
	j.getRotationString = function(){
		var rotationString= "";
		var rotation = this.getRotation();
		rotationString+=rotation[0] +" "+rotation[1]+" "+rotation[2]+" "+rotation[3]+" ";
		return rotationString
	};
	j.setTranslation = function(t){
		//translation x,y,z
		//var translationData =this.translationData.getValue();
		this.translationData[this.translationIndex]= t[0];
		this.translationData[this.translationIndex+1]= t[1];
		this.translationData[this.translationIndex+2]= t[2];
		//this.translationData.setValue(translationData)
		//console.log(translationData);
	};
	j.setRotation = function(q){
		//quaternion x,y,z,w
		//console.log(this.name);
		//console.log(this.rotationIndex);
		this.rotationData[this.rotationIndex] = q[0];
		this.rotationData[this.rotationIndex+1]= q[1];
		this.rotationData[this.rotationIndex+2]= q[2];
		this.rotationData[this.rotationIndex+3]= q[3];
		//console.log(q);
	};
	
	j.getTranslation = function(){
		var translation = vec3.fromValues(this.translationData[this.translationIndex],this.translationData[this.translationIndex+1],this.translationData[this.translationIndex+2]);
		return translation;
	};
	j.getRotation = function(){
		var rotation = quat.fromValues(this.rotationData[this.rotationIndex],this.rotationData[this.rotationIndex+1],this.rotationData[this.rotationIndex+2],this.rotationData[this.rotationIndex+3]);
		return rotation;
		
	};
	


	j.getRelativeTransformation =function(){
		var relativeJointTransformation = mat4.create();
        relativeJointTransformation	= mat4.fromRotationTranslation(relativeJointTransformation, this.getRotation(), this.getTranslation());
        return relativeJointTransformation;
	};
	 
	
	j.getParentTransformation = function(){
    	if (this.parent != null){
        	var parentTransformation = this.parent.getAbsoluteTransformation();
        } else{
        	var parentTransformation = mat4.create();
            parentTransformation =mat4.identity(parentTransformation);   
        }
    	return parentTransformation;
	};

    j.getAbsoluteTransformation=function(){
    	var absoluteJointTransformation = this.getParentTransformation();
        var absoluteJointTransformation = mat4.multiply( absoluteJointTransformation, absoluteJointTransformation, this.getRelativeTransformation()) ;
        return absoluteJointTransformation;
    };
    

    j.getAbsolutePosition = function(){
    	 var absoluteParentTransformation = this.getParentTransformation();
         //var relativePosition = vec3.create();
    	 var translation =  this.getTranslation();
         var relativePosition = vec3.fromValues(translation[0],translation[1],translation[2]);
         var absolutePosition = vec3.create();
         absolutePosition = vec3.transformMat4(absolutePosition,relativePosition,absoluteParentTransformation);
         return absolutePosition;
    };
        
	j.printStructure = function(){
		console.log(this.name);
		for (jointName in this.children){
			this.children[jointName].printStructure();
		};
	
	};
	
	//########################################################### 
	/**
	 * Skeleton pose class based on the Python classes using XML3D.math classes for transformations and the Xflow data structure as data storage 
	 * for the initialization of this class setXflowNode, buildFromJSONDescription and setJointOrder need to be called in that order otherwise it wont work
	 **/
	var SkeletonPose = function(){
		
		this.rootJointName = "";//joint name
		this.jointOrder = new Array();
		this.joints =new Array();//hash table
		this.translationInputElement  = null;
		this.rotationInputElement = null;
		this.skinningXflowNode = null;//skinning xlfow node
		this.rotationDataEntry =null;//Float32Array with length 4 * number of joints that stores a list of quaternions that is used as input to the skinning
	    this.translationDataEntry =null;//Float32Array with length 3 * number of joints that stores a list of translations that is used as input to the skinning
	    this.enableRootTransformation = true;
	    this.jointMapping = new Array();// is used in updateParametersFromJointList to map parameters to different joints
	    this.skinningObserver = null;//calls handleSkinningUpdate to allow access to the result of the xflow network
	    this.translationVector = null;
	    this.rotationVector = null;

		
		
	},p = SkeletonPose.prototype;

	/**
	 *  Extracts the input data entries from the Xflow Node that can then be manipulated based on a skeleton structure 
	 *  Note: the skeleton structure has to be constructed by calling buildFromJSONDescription and setJointOrder afterwards
	 * @param {node} DOM element that holds the Skinning Xflow 
	 **/
	p.setXflowNode = function(node){
	
		console.log("set xlfow node")
        //var node = $("#male-skinning")[0];
        this.skinningXflowNode = XML3D.base.resourceManager.getAdapter(node, XML3D.data).getXflowNode();
        console.log(this.skinningXflowNode);
        var inputSlots = this.skinningXflowNode._channelNode.inputSlots;
     
        console.log(inputSlots);
 
        this.rotationDataEntry =inputSlots["rotation;0"].dataEntry;//_value
        console.log(this.rotationDataEntry.getValue());
        this.translationDataEntry =inputSlots["translation;0"].dataEntry;//_value
        console.log(this.translationDataEntry.getValue());
        
		this.translationVector = new Float32Array(this.translationDataEntry.getLength());
		this.rotationVector = new Float32Array(this.rotationDataEntry.getLength());
		var iter = 0;
		while (iter < this.translationDataEntry.getLength()){
			this.translationVector[iter] = this.translationDataEntry._value[iter];
			iter+=1;
		}
		console.log("length");
		console.log(this.translationVector);
		//console.log(this.rotationDataEntry.getLength());
	    this.rotationInputElement = $("#male1_rotation")[0];
	    this.translationInputElement = $("#male1_translation")[0];
		
		
        var outputChannels = this.skinningXflowNode._channelNode.finalOutputChannels;
        console.log(outputChannels);
        //this.poseDataEntry  = this.skinningXflowNode._channelNode.finalOutputChannels.map;// .pose.channels[0].channel.entries[0].dataEntry _value
        //console.log("pose3");
        //console.log(this.skinningXflowNode._channelNode.finalOutputChannels);
        //console.log(this.skinningXflowNode._channelNode.finalOutputChannels.map['pose']);
        //console.log(this.skinningXflowNode._channelNode.finalOutputChannels.map['pose'].channels[0]);
        //console.log(this.poseDataEntry.getValue());
        this.skinningObserver = new XML3DDataObserver(this.handleSkinningUpdate.bind(this));
        this.skinningObserver.observe(node, {names: ["pose"]});
        this.jointObservers = new Array();
	};
	
	
	
	/**
	 * Creates the skeleton data structure using SkeletonJoint classes according to the given description
	 * @param {description} custom JSON format to describe skeletons
	 **/
	p.buildFromJSONDescription= function(description){
	
		console.log("add joints from JSON description")
		this.joints = new Array();

		
		this.rootJointName = description['skeleton']['root']
		var rootJoint =new SkeletonJoint();
		rootJoint.name = this.rootJointName;
		rootJoint.pose = this;
		rootJoint.translationData = this.translationDataEntry._value;//translationVector;//
		rootJoint.rotationData = this.rotationDataEntry._value;//rotationVector;//
		console.log("root name "+rootJoint.name);
		this.addJoint(rootJoint.name,rootJoint);
		
		//console.log(description['skeleton']['joints']);
		//create dictionary of joints from the parameters
		for (index in description['skeleton']['joints']){
			 console.log(description['skeleton']['joints'][index]);
			 var joint = new SkeletonJoint();
			 joint.name = description['skeleton']['joints'][index]["name"];
			 joint.pose = this;
			 joint.parentName = description['skeleton']['joints'][index]["parent"];
			 joint.translationData = this.translationDataEntry._value;//translationVector;//
			 joint.rotationData = this.rotationDataEntry._value;//rotationVector;//

			 this.addJoint(joint.name, joint)
		}
		
		//add hierarchy information
		console.log(this.joints)
		console.log("find children");
		for (jointName in this.joints){
			console.log(jointName);
			joint = this.joints[jointName];
			//console.log(joint.parentName);
			if (joint.parentName in this.joints){
				this.joints[joint.parentName].addChild(joint.name,joint);
				joint.parent = this.joints[joint.parentName];
			}
		}
		/*for (jointName in this.joints){
			this.jointMapping[jointName] = jointName;
			
		}*/
	};
	
	/**
	 * Sets the order in which parameters are written to the XflowNode input data arrays
	 * buildFromJSONDescription needs to be called first otherwise the function does not have an effect
	 *@param  {jointOrder} a list of joint names
	 **/
	p.setJointOrder = function(jointOrder){

		this.jointOrder = jointOrder;
		for (index in jointOrder){
			var jointName = jointOrder[index];
			if (jointName in this.joints){
				this.joints[jointName].boneIndex = index
				this.joints[jointName].translationIndex = index*3;
				this.joints[jointName].rotationIndex = index*4;
			}
		}
	};
	p.addJoint = function(name,joint){
		this.jointOrder.push(name);
		this.joints[name] = joint;
	};
	
	p.addJointTransformObserver = function(jointName){
        //var observer = new XML3DDataObserver(callback);
        //observer.observe(this.skinningXflowNode, {names: ["pose"]});
		var def = $("defs")[0];
		var rootNode = $("#observerGroup")[0]
		var transform = document.createElement("transform");
		var id = jointName+"observer"+index;
		transform.setAttribute("id",id+"_transform");
		transform.setAttribute("translation",0+" "+0+" "+0);
		transform.setAttribute("scale","10 10 10");
		def.appendChild(transform);
		//transform.translation.x = 100;
		var group = document.createElement("group");
		group.setAttribute("id", id);
		group.setAttribute("shader", "#Material_blue");
		group.setAttribute("transform","#"+id+"_transform");
		  
		var mesh = document.createElement("mesh");
		mesh.setAttribute("src","#Sphere-mesh");
		mesh.setAttribute("type","triangles");
		group.appendChild(mesh);
		rootNode.appendChild(group);
		
		//define callback that uses applies the translation to the newly created transform
		var callback = function(translation,quaternion){
			//console.log(translation);
			transform.translation.x =translation[0];
			transform.translation.y =translation[1];
			transform.translation.z =translation[2];
			
			// change color of shader here
			quaternion
			
			
		};
        this.joints[jointName].addTransformObserver("0",callback);
        
	};
	
	
	
	p.addTransformObserver = function(jointList,id,callback){
		console.log("add observer",id);
		this.jointObservers[id] = [jointList,callback];//XML3D.base.resourceManager.getAdapter(node, XML3D.data);// node
		
	};
	
	p.removeTransformObserver = function(id){
		delete this.jointObservers[id];

	};
	p.updateObservers = function(pose){
		for (id in this.jointObservers){
			//console.log("update observer"+id);
			var jointList = this.jointObservers[id][0];
			var parameterList = new Array();
			for (var i in jointList){
				//console.log(i);
				if (typeof pose == 'undefined'){
					var translation = this.joints[jointList[i]].getAbsolutePosition();
					var quaternion = [0,0,0,1];
				}else{
					//transformationIndex = boneIndex*16;
					var jointPose = new XML3DMatrix();
					var boneIndex = this.joints[jointList[i]].boneIndex
					XML3DMatrix.prototype.set.apply(jointPose,pose.subarray(0+16*boneIndex , 16+16*boneIndex));
					var tVector= jointPose.translation();
					var rVector= jointPose.rotation();
					var translation = [tVector.x,tVector.y,tVector.z];
					var quaternion = [rVector.x,rVector.y,rVector.z,rVector.w];
				}
				parameterList[jointList[i]] = [translation,quaternion];
			}
			
			this.jointObservers[id][1](parameterList);
			
		}
	};/**/
	

	p.translationToText = function(){
		var translationString = "";
		
		for (var i = 0; i< this.jointOrder.length;i++){//(jointName in this.joints)
			//console.log(this.jointOrder[i]);
			translationString += this.joints[this.jointOrder[i]].getTranslationString()
		
		}
		return  translationString;
	};
	p.rotationToText = function(){
		var rotationString = "";
		for (var i = 0; i< this.jointOrder.length;i++){//jointName in this.joints)
			rotationString += this.joints[this.jointOrder[i]].getRotationString()
		}
		return rotationString;
	};
	p.updateParametersFromJointList = function(joints){

		
		
		if (!this.enableRootTransformation && this.rootJointName in joints){
			joints[this.rootJointName].translation = [0,0,0];
		}
		//console.log(this.joints);
		for (jointName in joints){
			//console.log(jointName);
			 if (jointName in this.joints){
				 if (joints[jointName].translation){this.joints[jointName].setTranslation(joints[jointName].translation)};
				 if (joints[jointName].rotation){this.joints[jointName].setRotation(joints[jointName].rotation)};
				 
				// console.log("updated "+jointName)
			 }else if (jointName in this.jointMapping){
				 mappedJointName = this.jointMapping[jointName];
				 if (joints[jointName].translation){this.joints[mappedJointName].setTranslation(joints[jointName].translation)};
				 if (joints[jointName].rotation){this.joints[mappedJointName].setRotation(joints[jointName].rotation)};
			 }else{
				 //console.log("did not update "+jointName);
			 }
		}
		
		this.updateSkinning();
		
	
		
		//console.log(this.rotationDataEntry._value);
		
		//this.translationDataEntry.setValue(this.translationVector);
		//this.rotationDataEntry.setValue(this.rotationVector);
		//this.translationInputElement.setScriptValue(this.translationVector); //.textContent = this.translationToText();//
		//this.rotationInputElement.setScriptValue(this.rotationVector);//.textContent = this.rotationToText();//

		//position = this.joints['LeftHand'].getAbsolutePosition();
		//update object transformation
       	//console.log(position);
		//this.updateXflowInput();
		
		/*var f =function(result){
			console.log("request")
			console.log(result);
		}
		var r = new Xflow.ComputeRequest(this.skinningXflowNode,null,f);
		var dasd = r.getResult();*/
	};
	
	p.updateSkinning = function(){
		console.log("update skinning");
		this.translationInputElement.innerHTML = this.translationToText();
		this.rotationInputElement.innerHTML = this.rotationToText();
		//this.skinningXflowNode.compute()
		//this.translationDataEntry._notifyChanged();
		//this.rotationDataEntry._notifyChanged();
		//this.translationDataEntry.notifyChanged();
		//this.rotationDataEntry.notifyChanged();
		//this.skinningXflowNode._notifyChanged();
		
	};
	
	p.handleSkinningUpdate= function(records){
		
		/**/
		var pose = records[0].result.getValue("pose");
		//console.log("update skinning");//,pose
		this.updateObservers(pose);
		for (jointName in this.joints){
			this.joints[jointName].updateObservers(pose);//
		}

	};
	
	p.toggleTranslation = function(){
		
		this.enableRootTransformation = ! this.enableRootTransformation;
	};
	
	
	p.updateXflowInput = function(){
		
		if (this.translationDataEntry  != null && this.rotationDataEntry != null){
			
			this.writeTranslationToDataEntry(this.translationDataEntry);
			this.writeRotationToDataEntry(this.rotationDataEntry);
	
		}

	};
	
	p.writeTranslationToDataEntry = function(dataEntry){
		for (joinName in this.joints){
			this.joints[joinName].writeTranslationToDataEntry(dataEntry);
		}
			
		dataEntry._notifyChanged();

	};
	p.writeRotationToDataEntry = function(dataEntry){
		for (joinName in this.joints){
			this.joints[joinName].writeRotationToDataEntry(dataEntry);
		}
		dataEntry._notifyChanged();

	};/**/

	p.printStructure = function(){
		if (this.rootJointName != ""){
			this.joints[this.rootJointName].printStructure();		
		}
		
	};
	
	/*p.setXflowInput = function(){
	
	if (this.translationInputElement  != null && this.rotationInputElement != null){
		
		this.translationInputElement.textContent = this.translationToText();
		this.rotationInputElement.textContent = this.rotationToText();
	}

};*/
	 window.SkeletonPose = SkeletonPose;
	
}());