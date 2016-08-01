/*************************************************************************
	> File Name: hypercolumns_layer.cpp
	> Author: Jiang Qinhong
	> Mail: mylivejiang@gmail.com
	> Created Time: 2016年07月31日 星期一 19时40分04秒
 ************************************************************************/
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

#include "caffe/layers/hypercolumns_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ForwardNormal(const int nthreads,
    const Dtype* bottom_normal, const int num, const int channels,
    const int heights, const int width, const int sample_pernum,
    const int* sampling_list, Dtype* const top_normal) {
    // forward top_normals
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int bottom_n = index / sample_pernum; // the bottom n
      const int top_n = index / channels; // the top n
      const int c = index % channels; // the same channel of top and bottom
      const int bottom_index = sampling_list[top_n]; // the corresponding index of the bottom
      const Dtype* const bottom_slice = bottom_normal + (bottom_n * channels + c) * heights * width;
      top_normal[index] = bottom_slice[bottom_index];
    }
}

template <typename Dtype>
__global__ void ForwardHypercolumns(const int nthreads,
    const Dtype* bottom_data, const int num, const int channels,
    const int heights, const int width, const int sample_pernum,
    const int* sampling_list, Dtype* const top_data) {
    //forward hypercolumns, separate for each bottom
    CUDA_KERNEL_LOOP(index, nthreads) {
      
    }
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    // forward step, forward normal first
    Dtype* top_normal = top[1]->mutable_gpu_data();
    const Dtype* bottom_normal = bottom[0]->gpu_data();
    const int count = top[0]->count();
    ForwardNormal<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_normal, N_, K_, H_, W_, sample_num_, &selected_points_[0], top_normal
    );

    // then forward the hypercolumns
    Dtype* top_hypercolumns = top[1]->mutable_gpu_data();
    int top_channel_offset = 0;
    for (int i = 1; i < bottom.size(); ++i) {
      // do it according the corresponding bottom
      const Dtype* bottom_data = bottom[i]->gpu_data();
      const int bottom_channels = bottom[i]->shape(1);
      const int bottom_width = bottom[i]->shape(2);
      const int bottom_height = bottom[i]->shape(3);

    }


}








INSTANTIATE_LAYER_GPU_FUNCS(HyperColumnsLayer);
}// namespace caffe
