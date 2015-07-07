XML3D.shaders.register("selection", {

	vertex : [
		"attribute vec3 position;",
		"attribute vec3 normal;",
		"uniform mat4 modelViewProjectionMatrix;",
		"uniform mat4 modelMatrix;",
		"void main(void) {",
		"   float c = 0.2 / length(modelMatrix[0]);",
		"   vec4 pos = (modelViewProjectionMatrix * vec4(position + c * normal, 1.0));",
		"   gl_Position = pos;",
		"}"
	].join("\n"),

	fragment : [
		"#ifdef GL_ES",
		"precision highp float;",
		"#endif",
		"void main(void) {",
		"gl_FragColor = vec4(0.8, 0.6, 0, 0.3);",
		"}"
	].join("\n"),

	addDirectives: function(directives, lights, params) {
		// Add directives to the shader, depending on lights and parameters
	},
	hasTransparency: function(params) {
		return true;
	},
	uniforms: { // Used shader parameters with default values
		transparency: 0.1
	},

	samplers: { // Add name of samplers the shader needs
	}
});
