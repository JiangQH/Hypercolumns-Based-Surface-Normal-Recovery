import caffe
from random import shuffle
import numpy as np
from transformer import  SimpleTransformer
import matplotlib.pyplot as plt
"""
the naive method simple sample from the deconvlution layer. Note this will consume much memory
bottom 0 is the normal map, which is (batch_size, 3, 224, 224)
final bottom is the fc7-conv which is (batch_size, 4096, 1, 1)
other bottom is the deconv layer, with (batch_size, channels, 224, 224)
"""
class SimpleSample(caffe.Layer):

    def setup(self, bottom, top):
        self.top_names = ['hypercolumns', 'normalmap']
        params = eval(self.param_str)
        self.phase = params['phase']
        self.width = bottom[0].shape[2]
        self.height = bottom[0].shape[3]
        self.batch_size = bottom[0].shape[0]
        self.selected = []
        if self.phase == 'train':
            self.sample_num = 1000
        else:
            self.sample_num = self.width * self.height


    def reshape(self, bottom, top):
        top[1].reshape(self.batch_size * self.sample_num, 3)
        channels = 0
        for i in range(len(bottom)-1):
            channels += bottom[i+1].shape[1]
        top[0].reshape(self.batch_size * self.sample_num, channels)


    def forward(self, bottom, top):
        for batch in range(self.batch_size):
            # generate the selected list
            select_list = self.generate_pointlist(bottom[0].data[batch,...], phase=self.phase)
            self.selected.append(select_list)
            # do the sample job
            #a = bottom[0].data[batch,...]
        #    normal = SimpleTransformer().get_nomal_map(a)
            #plt.imshow(np.uint8(a.transpose((1, 2, 0))))
            for count in range(len(select_list)):
                row = select_list[count] / self.width
                col = select_list[count] % self.width
                cur = 0
                for j in range(1, len(bottom)-1, 1):
                    end = cur + bottom[j].shape[1]
                    top[0].data[batch * self.sample_num + count, cur : end] = bottom[j].data[batch, :, row, col]
                    cur = end
                top[0].data[batch * self.sample_num + count, cur:] = bottom[len(bottom)-1].data[batch, :, 0, 0]
                top[1].data[batch * self.sample_num + count, ...] = bottom[0].data[batch, :, row, col]




    def backward(self, top, propagate_down, bottom):
        """
        according the selected_list to do bp. batch = count / self.sample_num
        :param top:
        :param propagate_down:
        :param bottom:
        :return:
        """
        for i in range(len(bottom)):
            bottom[i].diff.fill(0)

        for batch in range(len(self.selected)):
            # do per batch
            e = top[0].shape[1]
            c = bottom[len(bottom) - 1].shape[1]
            bottom[len(bottom) - 1].diff[batch, :, 0, 0] = np.sum(top[0].diff[batch * self.sample_num
                                                                  : (batch+1) * self.sample_num, e-c:e])
            for count in range(len(self.selected[batch])):
                row = self.selected[batch][count] / self.width
                col = self.selected[batch][count] % self.width
                cur = 0
                for i in range(1, len(bottom)-1, 1):
                    end = cur + bottom[i].shape[1]
                    bottom[i].diff[batch, :, row, col] = top[0].diff[batch * self.sample_num + count, cur:end]
                    cur = end
        self.selected = [] # set to null after each iteration




    def generate_pointlist(self, normal_map, phase='train'):
        """
        generate the sample list. train: randomly valid 1000. test: all
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
        return abs(np.sum(np.square(normal_map[:, row, col])) - 1) < 1e-1
