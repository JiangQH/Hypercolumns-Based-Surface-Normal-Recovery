import numpy as np
class SimpleTransformer(object):

    def __init__(self, mean=[104, 117, 123]):
        self.mean = np.array(mean, dtype=np.float32)

    def preprocess(self, im):
        im = np.float32(im)
        im -= self.mean
        im = im[:, :, ::-1] #chang to BGR
        im = im.transpose((2, 0, 1)) # change to c * w * h
        return im

    def depreprocess(self, im):
        im = im.transpose((1, 2, 0))
        im += self.mean
        im = im[:, :, ::-1]
        return np.uint8(im)

    def get_normal(self, normal_map):
        normal = np.float32(np.copy(normal_map))
        normal *= 0.00390625
        normal *= 2
        normal -= 1
        #normal = normal.transpose((2, 0, 1)) # change to c * w * h
        return normal

    def get_nomal_map(self, normal):
        normal_map = normal.transpose((1, 2, 0))
        normal_map /= 2
        normal_map += 0.5
        normal_map *= 255
        return np.uint8(normal_map)







