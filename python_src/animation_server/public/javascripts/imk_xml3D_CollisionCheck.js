function CACheckCollision(connectionTarget)
{
	fullJSON = "{'command':'ExecuteCollision','bvh':'E:/Temp/carry_020_1_leftStance_130_172.bvh','json':'E:/Temp/geometry and skeleton.json'}";	
	// print JSON to screen (DEBUG)
	alert(fullJSON);	

	socket = new WebSocket(connectionTarget);
	socket.onopen = function (evt) { onOpen(evt) };
	socket.onerror = function (evt) {
	    alert(evt.error);
	};
	socket.onmessage = function (evt) { onMessage(evt) };
}

function KBConstraints(connectionTarget)
{
	fullJSON = "{'command':'KBConstraints'}";	
	// print JSON to screen (DEBUG)
	//alert(fullJSON);	

	socket = new WebSocket(connectionTarget);
	socket.onopen = function (evt) { onOpen(evt) };
	socket.onerror = function (evt) {
	    alert(evt.error);
	};
	socket.onmessage = function (evt) { onMessage2(evt) };
}

function onOpen(evt)
{
    socket.send(fullJSON);
    //do not close the socket, expect response!
}

function onMessage2(evt) {
    alert(evt.data);
    var msg = JSON.parse(evt.data);

	CreateCubes2(msg);
};

function onMessage(evt) {
    alert(evt.data);
    var msg = JSON.parse(evt.data);
    
    //socket.close();

    if (null == msg)
    {
        return;
    }

    var tmp = msg[0];
    var IsNameList = tmp.hasOwnProperty("name")

    if (IsNameList)
    {
        ShadeObjects(msg);
    }
    else
    {
        CreateCubes(msg);
    }
};

function CreateCubes(msg)
{
    var xml3dArray = document.getElementsByTagName("xml3d");
    var xml3d = xml3dArray[0]

    var defsArray = document.getElementsByTagName("defs");
    var defs = defsArray[0];

    var ns = XML3D.tools.creation;

    var shader = document.getElementById("imk_CollisionShader");

    if (!shader) {
        var s_Collision = ns.phongShader({
            id: "imk_CollisionShader", diffuseColor: "1.0 0 0",
            specularColor: "0 0 0", shininess: "40"
        });
        defs.appendChild(s_Collision);
    }
	
	var root = document.getElementById("xml3dView");
	
	// exit, if root was not found
	if (root == null || root.length == 0)
	{
		alert("XML 3D root node not found!");
		return;
	}
	var rOldGroup = document.getElementById("imkCollision");

	if (rOldGroup != null)
	{
		root.removeChild(rOldGroup);
	}

	var rootgroup = document.createElement("group");
		rootgroup.setAttribute("id", "imkCollision");
	root.appendChild(rootgroup); 
	
    for (var i = 0; i < msg.length; ++i) {
        var point = msg[i];

        	  var rGroup = document.getElementById("imkCollision");
              var group = document.createElement("group"); 
              group.setAttribute("id", "CollisionPoint" + i); 
              group.setAttribute("shader", "#imk_CollisionShader"); 
              group.setAttribute("style","transform: translate3d(" + point.x +"," + point.y + "," + point.z + ") scale3d(0.012, 0.012, 0.012);"); 
               
              var mesh = document.createElement("mesh"); 
              mesh.setAttribute("src","#Cube-mesh"); 
              mesh.setAttribute("type","triangles"); 
              group.appendChild(mesh); 
			  rGroup.appendChild(group);
    }

    
}

function CreateCubes2(msg)
{
    var xml3dArray = document.getElementsByTagName("xml3d");
    var xml3d = xml3dArray[0]

    var defsArray = document.getElementsByTagName("defs");
    var defs = defsArray[0];

    var ns = XML3D.tools.creation;

    var shader = document.getElementById("imk_ConstraintShader");

    if (!shader) {
        var s_Collision = ns.phongShader({
            id: "imk_ConstraintShader", diffuseColor: "0 1 0",
            specularColor: "0 0 0", shininess: "40"
        });
        defs.appendChild(s_Collision);
    }
	
	var root = document.getElementById("xml3dView");

	// exit, if root was not found
	if (root == null || root.length == 0)
	{
		alert("XML 3D root node not found!");
		return;
	}

	var rOldGroup = document.getElementById("imkConstraints");

	if (rOldGroup != null)
	{
		root.removeChild(rOldGroup);
	}

		var rootgroup = document.createElement("group");
		rootgroup.setAttribute("id", "imkConstraints");

	root.appendChild(rootgroup); 
	
    for (var i = 0; i < msg.length; ++i) {
        var point = msg[i];

        	  var rGroup = document.getElementById("imkConstraints");
              var group = document.createElement("group"); 
              group.setAttribute("id", "Constraint" + i); 
              group.setAttribute("shader", "#imk_ConstraintShader"); 
              group.setAttribute("style","transform: translate3d(" + point.x +"," + point.y + "," + point.z + ") scale3d(5, 5, 5);"); 
               
              var mesh = document.createElement("mesh"); 
              mesh.setAttribute("src","#Sphere-mesh"); 
              mesh.setAttribute("type","triangles"); 
              group.appendChild(mesh); 
			  rGroup.appendChild(group);
    }   
}

function ShadeObjects(msg)
{
    var xml3dArray = document.getElementsByTagName("xml3d");
    var xml3d = xml3dArray[0]

    var defsArray = document.getElementsByTagName("defs");
    var defs = defsArray[0];

    var ns = XML3D.tools.creation;

    var shader = document.getElementById("imk_CollisionShader");

    if (!shader) {
        var s_Collisioin = ns.phongShader({
            id: "imk_CollisionShader", diffuseColor: "1.0 0 0",
            specularColor: "0 0 0", shininess: "40"
        });
        defs.appendChild(s_Collisioin);
    }

    for (var i = 0; i < msg.length; ++i) {
        var obj = msg[i];
        var objtmp = document.getElementById("trolley02");
        

        var objModel = objtmp.getElementsByTagName("model");
		var extend = document.createElement("assetmesh"); 
              extend.setAttribute("includes","Mesh-006"); 
              extend.setAttribute("shader","imk_CollisionShader.xml#imkCollisionShader"); 
              objModel[0].appendChild(extend); 
    }

}
//function FindFirstShader(node)
//{
//    if (node.tagName.toUpperCase() != "GROUP" && node.tagName.toUpperCase != "MESH")
//        return;

//    var shadVar = node.getAttribute("shader");	
//    if (shadVar != null)
//    {
//        node.shader = "imk_Collision";
//        return;
//    }
//    else
//    {
//        for (var i=0; i< node.children.length; ++i)
//        {
//            FindFirstShader(node.children[i]);
//        }
//    }
//}