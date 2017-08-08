
class SceneInterface(object):
    def __init__(self):
        self.offset = [0 ,0]
        self.scene = None

    def set_scene(self, scene):
        self.scene = scene

    def get_height(self, x, z):
        x += self.offset[0]
        z += self.offset[1]
        if self.scene is not None:
            return self.scene.get_height(x, z)
        else:
            return 0

    def set_offset(self, x, z):
        self.offset = [x, z]
