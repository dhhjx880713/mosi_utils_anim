(function() {
    "use strict";

    var quat = XML3D.math.quat;
    var vec3 = XML3D.math.vec3;
    var vec4 = XML3D.math.vec4;
    var mat4 = XML3D.math.mat4;

    var PIOVER180 = Math.PI / 180.0;

    function AvatarCamera(transform, view, xml3dRoot) {
        this._view = view;
        this._xml3dRoot = xml3dRoot;
	    this._transform = transform;
	    this._initialAngles = transform.rotation.toEulerAngles().toArray();
	    this._initialTranslation = transform.translation.toArray();
		this.reset();
    }

    AvatarCamera.prototype.rotateLocal = function (x, y, z) {
        var tmp = quat.create();
        var axis = vec3.create();

        vec3.set(axis, 0, 0, 1);
        quat.setAxisAngle(tmp, axis, -this._localRollAngle * PIOVER180);
        quat.multiply(this._orientation, this._orientation, tmp);

        vec3.set(axis, 1, 0, 0);
        quat.setAxisAngle(tmp, axis, -this._localPitchAngle * PIOVER180);
        quat.multiply(this._orientation, this._orientation, tmp);

        vec3.set(axis, 0, 1, 0);
        quat.setAxisAngle(tmp, axis, y * PIOVER180);
        quat.multiply(this._orientation, this._orientation, tmp);

        vec3.set(axis, 1, 0, 0);
        quat.setAxisAngle(tmp, axis, (this._localPitchAngle + x)  * PIOVER180);
        quat.multiply(this._orientation, this._orientation, tmp);

        vec3.set(axis, 0, 0, 1);
        quat.setAxisAngle(tmp, axis, (z + this._localRollAngle) * PIOVER180);
        quat.multiply(this._orientation, this._orientation, tmp);

        this._localRollAngle = (this._localRollAngle + z) % 360;
        this._localPitchAngle = (this._localPitchAngle + x) % 360;

        this._updateTransformation();
    };

    AvatarCamera.prototype.rotateGlobal = function (x, y, z) {
        var tmp = quat.create();
        var axis = vec3.create();

        vec3.set(axis, 1, 0, 0);
        quat.setAxisAngle(tmp, axis, x * PIOVER180);
        quat.multiply(this._orientation, tmp, this._orientation);

        vec3.set(axis, 0, 1, 0);
        quat.setAxisAngle(tmp, axis, y * PIOVER180);
        quat.multiply(this._orientation, tmp, this._orientation);

        vec3.set(axis, 0, 0, 1);
        quat.setAxisAngle(tmp, axis, z * PIOVER180);
        quat.multiply(this._orientation, tmp, this._orientation);

        this._updateTransformation();
    };

    AvatarCamera.prototype.translateLocal = function (x, y, z) {
        vec4.transformMat4(this._position, vec4.fromValues(x, y, z, 1), this._transformation);

        this._updateTransformation();
    };

    AvatarCamera.prototype.translateGlobal = function (x, y, z) {
        vec4.add(this._position, this._position, vec4.fromValues(x, y, z, 1));

        this._updateTransformation();
    };

    AvatarCamera.prototype.setPosition = function (x, y, z) {
        vec4.set(this._position, x, y, z, 1);

        this._updateTransformation();
    };

    AvatarCamera.prototype.orbit = function (center, x, y, z) {
        center = vec4.fromValues(center[0], center[1], center[2], 1);
        var distance = vec4.len(vec4.sub(vec4.create(), this._position, center));

        this.rotateLocal(x, 0, z);1
        this.rotateGlobal(0, y, 0);
        this.setPosition(center[0], center[1], center[2]);
        this.translateLocal(0, 0, distance);
    };

    AvatarCamera.prototype.frame = function (min, max, center) {
        min = vec4.fromValues(min[0], min[1], min[2], 1);
        max = vec4.fromValues(max[0], max[1], max[2], 1);

        var extend = vec4.len(vec4.sub(vec4.create(), max, min)) / 2.0;

        var fov = this._view.fieldOfView;
        var distance = extend / Math.tan(fov * 0.45);

        var aspect = this._xml3dRoot.clientWidth / this._xml3dRoot.clientHeight;

        if (aspect < 1)
            distance /= aspect;

        this.setPosition(center[0], center[1], center[2]);
        this.translateLocal(0, 0, distance);

        this._updateTransformation();
    };

	AvatarCamera.prototype.reset = function () {
		this._transformation = mat4.create();
		this._orientation = quat.create();
		this._position = vec4.fromValues(0, 0, 0, 1);
		this._localPitchAngle = 0;
		this._localRollAngle = 0;
		this.rotateLocal(-this._initialAngles[0] / PIOVER180, -this._initialAngles[1] / PIOVER180, this._initialAngles[2] / PIOVER180);
		this.translateGlobal(this._initialTranslation[0], this._initialTranslation[1], this._initialTranslation[2]);
	};

    AvatarCamera.prototype._updateTransformation = function () {
        mat4.fromRotationTranslation(this._transformation, this._orientation, this._position);

        this._transform.rotation.set(this._orientation);
        this._transform.translation.set(this._position);
    };

    window.AvatarCamera = AvatarCamera;
}());

(function() {
    "use strict";

    function findXML3DRoot(el) {
        if(el.tagName == "xml3d")
            return el;

        if(el.parentNode)
            return findXML3DRoot(el.parentNode);

        return null;
    }

    function getOrCreateDefs(xml3dRoot) {

        var defs = XML3D.util.evaluateXPathExpr(
            xml3dRoot, './/xml3d:defs[1]').singleNodeValue;

        if (!defs) {
            defs = XML3D.createElement("defs");
            xml3dRoot.appendChild(defs);
        }

        return defs;
    }

	var PIOVER180 = Math.PI / 180.0;

	// --- mouse buttons ---
    var MOUSEBUTTON_LEFT = 0;
    var MOUSEBUTTON_RIGHT = 2;

    // --- keys ---
    var KEY_A = 65;
	var KEY_C = 67;
    var KEY_D = 68;
    var KEY_E = 69;
    var KEY_F = 70;
	var KEY_H = 72;
    var KEY_Q = 81;
	var KEY_R = 82;
    var KEY_S = 83;
    var KEY_W = 87;

    var KEY_CTRL = 17;

    var KEY_PGUP = 33;
    var KEY_PGDOWN = 34;

    // arrow keys
    var KEY_LEFT = 37;
    var KEY_UP = 38;
    var KEY_RIGHT = 39;
    var KEY_DOWN = 40;

    function AvatarCameraController(viewGroupId) {
        this._viewGroup = document.getElementById(viewGroupId);
        this._view = this._viewGroup.children[0];
        this._xml3dRoot = findXML3DRoot(this._viewGroup);

        if (!this._xml3dRoot === null)
            throw new Error("Could not find xml3d root element for given groupid");

        var transform = document.getElementById(this._viewGroup.transform.substring(1));// problemsgetAttribute("transform")
        if (!transform) {
            transform = XML3D.createElement("transform");
            transform.setAttribute("id", "AvatarCameraTransform");
            var defs = getOrCreateDefs($("xml3d")[0]);
            defs.appendChild(transform);
            this._viewGroup.transform = "#" + transform.id;
        }

        this._camera = new AvatarCamera(transform, this._view, this._xml3dRoot);

	    this.reset();

        this._currentlyPressedKeys = {};

        this._translationSensitivity = 3;
        this._rotationSensitivity = 1;
	    this._invertMouseXAxis = false;
	    this._invertMouseYAxis = false;

        this._leftMouseDown = false;
        this._rightMouseDown = false;
	    this._lastMouseX = 0;
	    this._lastMouseY = 0;

	    this._shouldHighlightSelection = true;
	    this._selectionShader = "#selection-effect";
	    this._highlightGroup = null;
	    this._currentSelection = null;

        this._registerEventListeners();
    }

	Object.defineProperties(AvatarCameraController.prototype, {
		invertMouseXAxis: {
			get: function () {
				return this._invertMouseXAxis;
			},
			set: function (invert) {
				this._invertMouseXAxis = invert;
			}
		},
		invertMouseYAxis: {
			get: function () {
				return this._invertMouseYAxis;
			},
			set: function (invert) {
				this._invertMouseYAxis = invert;
			}
		},
		rotationSensitivity: {
			get: function () {
				return this._rotationSensitivity
			},
			set: function (sensitivity) {
				this._rotationSensitivity = sensitivity;
			}
		},
		translationSensitivity: {
			get: function () {
				return this._translationSensitivity;
			},
			set: function (sensitivity) {
				this._translationSensitivity = sensitivity;
			}
		},
		highlightSelection: {
			get: function () {
				return this._shouldHighlightSelection;
			},
			set: function (shouldHighlight) {
				this._shouldHighlightSelection = shouldHighlight;
				if (this._shouldHighlightSelection)
					this._highlightSelection();
				else
					this._clearHighlightSelection();
			}
		}
	});

	AvatarCameraController.prototype.select = function (mesh) {
		if (this._currentSelection)
			this.clearSelection();

		this._currentSelection = mesh;
		this._highlightSelection();
	};

	AvatarCameraController.prototype.clearSelection = function () {
		if (!this._currentSelection)
			return;
		this._clearHighlightSelection();
		this._currentSelection = null;
	};

	AvatarCameraController.prototype.reset = function () {
		this._camera.reset();
	};

    AvatarCameraController.prototype._registerEventListeners = function () {
        this._xml3dRoot.addEventListener("mouseup", this._onMouseUp.bind(this));
        document.addEventListener("mouseup", this._onMouseUp.bind(this));
        this._xml3dRoot.addEventListener("mousedown", this._onMouseDown.bind(this));
        this._xml3dRoot.addEventListener("mousemove", this._onMouseMove.bind(this));
        document.addEventListener("keydown", this._onKeyDown.bind(this));
        document.addEventListener("keyup", this._onKeyUp.bind(this));
        requestAnimationFrame(this._processKeys.bind(this));
    };

    AvatarCameraController.prototype._onMouseUp = function (event) {
        if (event.button === MOUSEBUTTON_LEFT)
            this._leftMouseDown = false;

        if (event.button === MOUSEBUTTON_RIGHT)
            this._rightMouseDown = false;

	    event.preventDefault();
    };

    AvatarCameraController.prototype._onMouseDown = function (event) {
        this._leftMouseDown = event.button === MOUSEBUTTON_LEFT;
        this._rightMouseDown = event.button === MOUSEBUTTON_RIGHT;

        if (this._leftMouseDown) {
            if (event.target.tagName === "mesh")
	            this.select(event.target);
            else
                this.clearSelection();
        }
    };

	AvatarCameraController.prototype._toggleHighlight = function () {
		this.highlightSelection = !this.highlightSelection;
	};

	AvatarCameraController.prototype._highlightSelection = function () {
		if (!this._shouldHighlightSelection || !this._currentSelection)
			return;

		var g = this._currentSelection.parentNode.cloneNode(true);
		g.setAttribute("shader", "");
		this._highlightGroup = XML3D.createElement("group");
		this._highlightGroup.setAttribute("shader", this._selectionShader);
		this._currentSelection.parentNode.parentNode.appendChild(this._highlightGroup);
		this._highlightGroup.appendChild(g);
	};

	AvatarCameraController.prototype._clearHighlightSelection = function () {
		if (this._highlightGroup)
			this._currentSelection.parentNode.parentNode.removeChild(this._highlightGroup);
		this._highlightGroup = null;
	};

    AvatarCameraController.prototype._onMouseMove = function (event) {
        var deltaX = (event.clientX - this._lastMouseX) * this._rotationSensitivity;
	    deltaX = this._invertMouseXAxis ? -deltaX : deltaX;

        var deltaY = (event.clientY - this._lastMouseY) * this._rotationSensitivity;
		deltaY = this._invertMouseYAxis ? -deltaY : deltaY;

        if (this._leftMouseDown) {
            if (this._currentlyPressedKeys[KEY_CTRL])
                this._camera.translateLocal(-deltaX, deltaY, 0);
            else
                this._camera.rotateLocal(deltaY, -deltaX, 0);
        }
        else {
            if (this._rightMouseDown) {
	            if (this._currentlyPressedKeys[KEY_CTRL])
	                this._camera.translateLocal(0, 0, deltaY);
                else if (this._currentSelection)
                    this._camera.orbit(this._currentSelection.getBoundingBox().transform(this._currentSelection.getWorldMatrix()).center().toArray(), deltaY, -deltaX, 0);
                else
                    this._camera.orbit(this._xml3dRoot.getBoundingBox().center().toArray(), deltaY, -deltaX, 0);
            }
        }

        this._lastMouseX = event.clientX;
        this._lastMouseY = event.clientY;
    };

    AvatarCameraController.prototype._onKeyDown = function (event) {
        this._currentlyPressedKeys[event.keyCode] = true;
	    event.preventDefault();
    };

    AvatarCameraController.prototype._onKeyUp = function (event) {
        this._currentlyPressedKeys[event.keyCode] = false;
	    if (event.keyCode === KEY_H)
		    this._toggleHighlight();
	    if (event.keyCode === KEY_C)
		    this.clearSelection();
	    if (event.keyCode === KEY_R)
		    this.reset();
	    event.preventDefault();
    };

    AvatarCameraController.prototype._processKeys = function () {
        if (this._currentlyPressedKeys[KEY_W])
            this._camera.translateLocal(0, 0, -this._translationSensitivity);
        if (this._currentlyPressedKeys[KEY_S])
            this._camera.translateLocal(0, 0, this._translationSensitivity);
        if (this._currentlyPressedKeys[KEY_A])
            this._camera.translateLocal(-this._translationSensitivity, 0, 0);
        if (this._currentlyPressedKeys[KEY_D])
            this._camera.translateLocal(this._translationSensitivity, 0, 0);
        if (this._currentlyPressedKeys[KEY_E] || this._currentlyPressedKeys[KEY_PGUP])
            this._camera.translateLocal(0, this._translationSensitivity, 0);
        if (this._currentlyPressedKeys[KEY_Q] || this._currentlyPressedKeys[KEY_PGDOWN])
            this._camera.translateLocal(0, -this._translationSensitivity, 0);

        if (this._currentlyPressedKeys[KEY_LEFT])
            this._camera.rotateLocal(0, this._rotationSensitivity, 0);
        if (this._currentlyPressedKeys[KEY_RIGHT])
			 this._camera.rotateLocal(0, -this._rotationSensitivity, 0);
        if (this._currentlyPressedKeys[KEY_UP])
            this._camera.rotateLocal(-this._rotationSensitivity, 0, 0);
        if (this._currentlyPressedKeys[KEY_DOWN])
            this._camera.rotateLocal(this._rotationSensitivity, 0, 0);

        if (this._currentlyPressedKeys[KEY_F]) {
            if (this._currentSelection) {
	            var bb = this._currentSelection.getBoundingBox().transform(this._currentSelection.getWorldMatrix());
                this._camera.frame(bb.min.toArray(), bb.max.toArray(), bb.center().toArray());
            }
            else
                this._camera.frame(this._xml3dRoot.getBoundingBox().min.toArray(), this._xml3dRoot.getBoundingBox().max.toArray(), this._xml3dRoot.getBoundingBox().center().toArray());
        }

        requestAnimationFrame(this._processKeys.bind(this));
    };

    window.AvatarCameraController = AvatarCameraController;
}());
