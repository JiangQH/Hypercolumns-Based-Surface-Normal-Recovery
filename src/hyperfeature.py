import caffe
from random import shuffle
import numpy as np
import scipy.misc
"""
This file implements the hyperfeature construction. Using the
conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 and fc7-conv layer response.
note that
normal: (batch_size, 3, 224, 224)
conv1_2: (batch_size, 64, 224, 224)
conv2_2: (batch_size, 128, 112, 112)
conv3_3: (batch_size, 256, 56, 56)
conv4_3: (batch_size, 512, 28, 28)
conv5_3: (batch_size, 512, 14, 14)
fc7-conv: (batch_size, 4096, 1, 1)
only the pooling changes the layer's size.
the output should be of shape hyperfeature-(batch_size * sampled_num, 5568?)
                              sampled_normal-(batch_size * sampled_num, 3)
tricks?
1. every feature point has the corresponding point
2. how to do bp? a)concat layer borrow? b) resize borrow? just accumulating them?
"""
class HyperColumnFeatures(caffe.Layer):

    def setup(self, bottom, top):
        self.top_names = ['hyper_feature', 'sampled_normal']
        params = eval(self.param_str)
        self.phase = params['phase']
        self.batch_size = bottom[0].shape[0]
        self.width = bottom[0].shape[2]
        self.height = bottom[0].shape[3]
        self.selected = []
        if self.phase == 'train':
            self.sample_num = 1000
        else:
            self.sample_num = self.width * self.height

    def reshape(self, bottom, top):
        """
        it is different when do training and testing
        :param bottom:
        :param top:
        :return:
        """
        top[1].reshape(self.batch_size * self.sample_num, bottom[0].shape[1]) # the sampled normal point
        channel = 0
        for itt in range(len(bottom)-1):
            channel += bottom[itt+1].shape[1]
        top[0].reshape(self.batch_size * self.sample_num, channel) # the hyper feature


    def forward(self, bottom, top):
        """
        concat the features
        :param bottom:
        :param top:
        :return:
        """
        for batch in range(self.batch_size):
            # do upsample to every batch
            upconv2_2 = self.upsample_feature(bottom[2].data[batch,...])
            upconv3_3 = self.upsample_feature(bottom[3].data[batch,...])
            upconv4_3 = self.upsample_feature(bottom[4].data[batch,...])
            upconv5_3 = self.upsample_feature(bottom[5].data[batch,...])
            # generate the sample list
            selected_list = self.generate_pointlist(batch, bottom[0].data[batch,...], self.phase)
            # do sample job
            self.selected.append(selected_list)
            for count in range(len(selected_list)):
                row = selected_list[count] / self.width
                col = selected_list[count] % self.width
                top[1].data[batch * self.sample_num + count, ...] = bottom[0].data[batch, :, row, col]
                top[0].data[batch * self.sample_num + count, ...] = np.concatenate((bottom[1].data[batch, :, row, col],
                                                                                    upconv2_2[:, row, col], upconv3_3[:, row, col],
                                                                                    upconv4_3[:, row, col], upconv5_3[:, row, col],
                                                                                    bottom[6].data[batch, :, 0, 0]))





    def backward(self, top, propagate_down, bottom):
        """
        the key point
        :param top:
        :param propagate_down:
        :param bottom:
        :return:
        """
        pass






    def upsample_feature(self, feature_map):
        """
        feature_map is (channels, width, height)
        do bilinear upsample every channel
        :param feature_map:
        :return:
        """
        upsampled = []
        channels = feature_map.shape[0]
        for c in range(channels):
            ups = scipy.misc.imresize(feature_map[c,...], size=(224,224), interp='bilinear', mode='F')
            upsampled.append(ups)
        return np.asarray(upsampled)



    def generate_pointlist(self, batch, normal_map, phase='train'):
        """
        generate the sample list. train: randomly valid 1000. test: all
        :param batch:
        :param normal_map:
        :param phase:
        :return:
        """
        selected_list = []
        if phase == 'train':
            lists = range(0, self.width * self.height)
            shuffle(lists)
            for location in lists:
                if self.isvalid(location, normal_map):
                    selected_list.append(location)
                    if len(selected_list) == self.sample_num:
                        break
        if phase == 'test':
            selected_list = range(0, self.sample_num)
        return selected_list





    def isvalid(self, index, normal_map):
        """
        whether the given point is valid normal point
        for the current given label it is the same value when not valid
        :param normal_map:
        :return:
        """
        row = index / normal_map.shape[1]
        col = index % normal_map.shape[1]
        return not(normal_map[0, row, col] == normal_map[1, row, col] and normal_map[1, row, col] == normal_map[2, row, col])



