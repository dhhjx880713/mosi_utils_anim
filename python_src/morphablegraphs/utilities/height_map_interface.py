
class HeightMapInterface(object):
    def __init__(self, image, width, depth, height_scale):
        self.height_map_image = image
        self.height_scale = height_scale
        self.width = width
        self.depth = depth
        self.x_offset = 0
        self.z_offset = 0

    def to_relative_coordinates(self, center_x, center_z, x, z):
        """ get position relative to upper left
        """
        relative_x = x - center_x
        relative_z = z - center_z
        relative_x += self.width / 2
        relative_z += self.depth / 2

        # scale by width and depth to range of 1
        relative_x /= self.width
        relative_z /= self.depth
        return relative_x, relative_z

    def get_height_from_relative_coordinates(self, relative_x, relative_z):
        if relative_x < 0 or relative_x > 1.0 or relative_z < 0 or relative_z > 1.0:
            print("Coordinates outside of the range")
            return 0
        # scale by image width and height to image range
        ix = relative_x * self.height_map_image.size[0]
        iy = relative_z * self.height_map_image.size[1]
        p = self.height_map_image.getpixel((ix, iy))
        return (p / 255) * self.height_scale

    def get_height(self, x, z):
        rel_x, rel_z = self.to_relative_coordinates(self.x_offset, self.z_offset, x, z)
        y = self.get_height_from_relative_coordinates(rel_x, rel_z)
        #print("get height", x, z,":", y)
        return y
