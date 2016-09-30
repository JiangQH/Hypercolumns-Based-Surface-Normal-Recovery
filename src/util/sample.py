import caffe
from random import shuffle
import numpy as np
from transformer import SimpleTransformer
from biinterp import bilinear_interpolation as interp
import matplotlib.pyplot as plt

class SampleLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.top_names = ['hypercolums', 'normal_point']
        params = eval(self.param_str)
        self.is_train = params['is_train']
        self.batch_size = bottom[0].shape[0]
        self.h = bottom[0].shape[2]
        self.w = bottom[0].shape[3]
        if self.is_train:
            self.sample_num = 1000
        else:
            self.sample_num = self.h * self.w
        self.selected = []


    def reshape(self, bottom, top):
        top[1].reshape(self.sample_num * self.batch_size, 3)
        c = 0
        for it in range(1, len(bottom), 1):
            c += bottom[it].shape[1]
        top[0].reshape(self.sample_num * self.batch_size, c)

    def forward(self, bottom, top):
        for batch in range(self.batch_size):
            normal_map = SimpleTransformer().get_normal(bottom[0].data[batch,...])
            selected_list = self.generate_pointlist(normal_map)
            self.selected.append(selected_list)

            for count in range(len(selected_list)):
                r = selected_list[count] / self.w
                c = selected_list[count] % self.h
                top[1].data[batch * self.sample_num, ...] = normal_map[:, r, c]
                cur = 0
                for i in range(1, len(bottom)-1, 1):
                    end = cur + bottom[i].shape[1]
                    original_size = (bottom[i].shape[2], bottom[i].shape[3])
                    scale = self.h / original_size[0]
                    weights = interp((r, c), original_size, scale)
                    top[0].data[batch * self.sample_num + count, cur:end] = self.build_point(weights,
                                                                                             bottom[i].data[batch, ...])
                    cur = end
                top[0].data[batch * self.sample_num, cur:] = bottom[len(bottom)-1].data[batch, :, 0, 0]



    def backward(self, top, propagate_down, bottom):
        for i in range(1, len(bottom), 1):
            bottom[i].diff.fill(0)

        for batch in range(len(self.selected)):
            for count in range(len(self.selected[batch])):
                r = self.selected[batch][count] / self.w
                c = self.selected[batch][count] % self.w
                cur = 0
                for i in range(1, len(bottom)-1, 1):
                    end = cur + bottom[i].shape[1]
                    original_size = (bottom[i].shape[2], bottom[i].shape[3])
                    scale = self.h / original_size[0]
                    weights = interp((r,c), original_size, scale)
                    self.build_diff(top[0].diff[batch * self.sample_num + count, cur : end], weights, bottom[i].diff[batch, ...])
                    cur = end
                bottom[len(bottom)-1].diff[batch, :, 0, 0] += top[0].diff[batch * self.sample_num + count, cur:]
        self.selected = []


    def generate_pointlist(self, normal_map):
        lists = range(0, self.h * self.w)
        if self.is_train:
            shuffle(lists)
            selected_list = []
            for count in range(len(lists)):
                if self.isvalid(lists[count], normal_map):
                    selected_list.append(lists[count])
                if count == self.sample_num:
                    return selected_list
        else:
            return lists

    def isvalid(self, index, normal_map):
        r = index / self.w
        c = index % self.w
        return abs(np.sum(normal_map[:, r, c] ** 2) - 1.0) < 1e-1


    def build_point(self, weights, feature_map):
        result = np.zeros(feature_map.shape[0])
        for points, weight in weights.iteritems():
            r = points[0]
            c = points[1]
            result += weight * feature_map[:, r, c]
        return result

    def build_diff(self, top_diff, weights, bottom_diff):
        # change the inside will change it outside in python?
        for points, weight in weights.iteritems():
            r = points[0]
            c = points[1]
            bottom_diff[:, r, c] += weight * top_diff

