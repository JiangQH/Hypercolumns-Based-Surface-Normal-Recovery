import caffe
import os
from transformer import  SimpleTransformer
from random import shuffle
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

class NormalDataLayer(caffe.Layer):
    """
    self design python data layer to load rgb and normal image both
    """
    def setup(self, bottom, top):
        self.top_names = ['data', 'normal']
        params = eval(self.param_str)
        self.batch_loader = BatchLoader(params)
        self.batch_size = params['batch_size']
        self.im_shape = params['im_shape']

    def forward(self, bottom, top):
        """
        load data
        :param bottom:
        :param top:
        :return:
        """
        for iter in range(self.batch_size):
            im, normal = self.batch_loader.load_next_image()
            #img = SimpleTransformer().get_nomal_map(normal)
            #plt.imshow(img)
            #plt.show()
            top[0].data[iter,...] = im
            top[1].data[iter,...] = normal

    def reshape(self, bottom, top):
        top[0].reshape(self.batch_size, 3,
                       self.im_shape[0], self.im_shape[1])
        top[1].reshape(self.batch_size, 3,
                       self.im_shape[0], self.im_shape[1])

    def backward(self, top, propagate_down, bottom):
        pass

class BatchLoader(object):
    """
    batch loader to load image and normal
    """
    def __init__(self, params):
        self.rgb_root = params['rgb_root']
        self.normal_root = params['normal_root']
        self.im_shape = params['im_shape']
        self.augmentation = params['augmentation']
        self.transformer = SimpleTransformer()
        self.im_list = os.listdir(self.rgb_root)
        shuffle(self.im_list)
        self.cur = 0
        print 'BatchLoader load with {} images'.format(len(self.im_list))

    def load_next_image(self):
        if self.cur == len(self.im_list):
            self.cur = 0
            shuffle(self.im_list)
        name = self.im_list[self.cur]
        im_file = os.path.join(self.rgb_root, name)
        normal_file = os.path.join(self.normal_root, name)
        im = np.asarray(Image.open(im_file))
        normal = np.asarray(Image.open(normal_file))
        im = scipy.misc.imresize(im, self.im_shape)
        normal = scipy.misc.imresize(normal, self.im_shape)
       # plt.imshow(normal)
        #plt.imshow(im)
        #plt.show()
        if self.augmentation:
            # generate random flipping
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]
            normal = normal[:, ::flip, :]
        self.cur += 1
        return self.transformer.preprocess(im), self.transformer.get_normal(normal)


