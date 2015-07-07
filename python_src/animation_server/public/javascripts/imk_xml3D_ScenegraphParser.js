var listOfObjects = [];	

function KBUpdate(connectionTarget)
{
	// clear object list
	listOfObjects = [];	
	// find root of scenegraph
	var root = document.getElementsByTagName("xml3d");
	// exit, if root was not found
	if (root == null || root.length == 0)
	{
		alert("XML 3D root node not found!");
		return;
	}
	// convert children list to array
	var sgNodesArray = Array.prototype.slice.call(root[0].children);	
	// iterate through nodes ...	
	for (n of sgNodesArray) { TraverseSG(n, ""); }
	// start recursive tree walk	
	//TraverseSG(root, "");
	// transform object list to JSON string

	fullJSON = JSON.stringify(listOfObjects);	
	// print JSON to screen (DEBUG)
	console.log(fullJSON);
	// send via websocket ...
	socket = new WebSocket(connectionTarget);
	socket.onopen = function(evt) { onOpen(evt) };	
}

function TraverseSG(node, parent)
{
	if (node.nodeName.toLowerCase() != "group")
		return;
	// get id / name
	var id = node.getAttribute("id");	
	if (id == null)
		id = listOfObjects.length
	var name = node.getAttribute("name");	
	if (name == null)
		name = id
	// get transformation matrix
	if (name == "male"){
		// construct the matrix for the virtual human model from root joint parameters 
		var translationString = $("#male1_translation")[0].textContent;
		var translationVector = $.trim(translationString).split(" ").map(parseFloat);
		maleTranslation = [];
		for(var i = 0 ; i < 3; i++){
			maleTranslation.push(translationVector[i]);
		}
		maleTranslation[2]=0;//set z axis to zero
		
		var rotationString = $("#male1_rotation")[0].textContent;
		var rotationVector =$.trim(rotationString).split(" ").map(parseFloat);
		maleOrientation = [];
		for(var i = 0 ; i < 4; i++){
			maleOrientation.push(rotationVector[i]);
		}
		maleOrientation_q = XML3D.math.quat.create();
		maleOrientation_q.set(maleOrientation);
		rotate_z = XML3D.math.quat.create();
		rotate_z.set ([-0, -0, 0.716172, 0.697923] ); 
		rotate_y = XML3D.math.quat.create();//inverse of the start transformation in the electrolux_scene
		rotate_y.set( [0, 0.716173, -0, 0.697924]);

	    maleOrientation = XML3D.math.quat.multiply(maleOrientation_q,maleOrientation_q, rotate_y);
		maleOrientation = XML3D.math.quat.multiply(maleOrientation_q,maleOrientation_q ,rotate_z);
		
		
		var matrixData = XML3D.math.mat4.create();
		var matrixData=  XML3D.math.mat4.fromRotationTranslation(matrixData,maleOrientation, maleTranslation);
		matrix = new XML3DMatrix(matrixData);
		console.log("transformation matrix of the human model:");
		console.log(matrix);
	}else{
		// extract matrix from XML3D node
		var matrix = node.getLocalMatrix();	
	}
	// get bounding box
	var bb = node.getBoundingBox();	
	// get mesh informations ...
	
	
	var meshNodes = node.getElementsByTagName("mesh");
	var modelNodes = node.getElementsByTagName("model");
	var meshArray = Array.prototype.slice.call(meshNodes);
	var modelArray = Array.prototype.slice.call(modelNodes);
	var arrayAll = meshArray.concat(modelArray);
	
	var listOfMeshes = [];
	for (m of arrayAll)
	{
		if (m.parentNode != node)
			continue;
		var msrc = m.getAttribute("src");
		if (msrc == null)
			msrc = "";
		listOfMeshes.push(msrc);
	}
	
	// wrap data in object and add to object list
	listOfObjects.push(
	{
		objID:id.toString(),
		objName:name.toString(),
		objParent:parent.toString(),
		objMatrix:Array.prototype.slice.call(matrix._data).map(String),
		objBoundingBox:Array.prototype.slice.call(bb._min._data).concat(Array.prototype.slice.call(bb._max._data)).map(String),
		objMeshes:Array.prototype.slice.call(listOfMeshes).map(String),
	});
	
	// convert children list to array
	var sgNodesArray = Array.prototype.slice.call(node.children);	
	// iterate through nodes ...	
	for (n of sgNodesArray) { TraverseSG(n, id); }
}

function onOpen(evt)
{
	socket.send(fullJSON);
	socket.close();
}


