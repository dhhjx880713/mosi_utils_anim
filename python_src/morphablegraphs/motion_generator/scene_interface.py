
class SceneInterface(object):
    def __init__(self):
        self.scene = None

    def set_scene(self, scene):
        self.scene = scene

    def get_height(self, x, z):
        if self.scene is not None:
            return self.scene.get_height(x, z)
        else:
            return 0


