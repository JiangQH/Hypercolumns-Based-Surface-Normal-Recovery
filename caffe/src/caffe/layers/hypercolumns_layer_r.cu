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
#include "../../../../../../../../../usr/include/c++/4.7/vector"

namespace caffe {

template <typename Dtype>
__global__ void ForwardNormal(const int nthreads,
    const Dtype* bottom_normal, const int num, const int channels,
    const int height, const int width, const int sample_pernum,
    const int* sampling_list, Dtype* const top_normal) {
    // forward top_normals
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int top_n = index / channels; // the top n
      const int bottom_n = top_n / sample_pernum; // the bottom n
      const int c = index % channels; // the same channel of top and bottom
      const int bottom_index = sampling_list[top_n]; // the corresponding index of the bottom
      const Dtype* const bottom_slice = bottom_normal + (bottom_n * channels + c) * height * width;
      top_normal[index] = bottom_slice[bottom_index];
    }
}

template <typename Dtype>
__global__ void ForwardHypercolumns(const int nthreads,
    const int bottom_count, const Dtype** bottom_datas, const int* bottom_channels,
    const int* bottom_heights, const int* bottom_widths, const int sample_pernum,
    const int top_channels,  const int* sampling_list,
    const int W, Dtype* const top_data) {
    //forward hypercolumns, separate for each bottom
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int top_n = index / top_channels; // find the corresponding index in the sampling list
        int offset_channels = index % top_channels;
        int bottom_id = 0;
        while(bottom_id<bottom_count) {
            if(offset_channels - bottom_channels[bottom_id] < 0) {
                break;
            }
            offset_channels -= bottom_channels[bottom_id];
            ++bottom_id;
        }
        // get the corresponding bottom and it's channels

    }
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    // check and instance the cuda needed data
    if (!cuda_instanced_) {
        instance_cuda_data();
    }

    // generate the sampling list and copy it
    generate_list(bottom[0], false);
    CUDA_CHECK(cudaMemcpy(cuda_samplelist_, &selected_points_[0], selected_points_.size()* sizeof(int)));

    // forward step, forward normal first
    Dtype* top_normal = top[1]->mutable_gpu_data();
    const Dtype* bottom_normal = bottom[0]->gpu_data();
    const int count1 = top[1]->count();
    ForwardNormal<Dtype><<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(
      count1, bottom_normal, N_, K_, H_, W_, sample_num_, cuda_samplelist_, top_normal
    );

    // then forward the hypercolumns
    Dtype* top_hypercolumns = top[0]->mutable_gpu_data();
    vector<const Dtype*> bottom_datas;
    const int bottom_count = bottom.size() - 1;
    for (int i = 1; i < bottom.size(); ++i) {
        bottom_datas.push_back(bottom[i]->gpu_data());
    }
    const int nthreads = N_ * sample_num_ * total_channels_;
    ForwardHypercolumns<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_count, &bottom_datas[0], cuda_channels_, cuda_heights_, cuda_widths_, sample_num_,
        total_channels_, cuda_samplelist_, W_, top_hypercolumns
    );
}

template <typename Dtype>
__global__ void BackwardHypercolumns(const int nthreads,
     const Dtype** bottom_diffs, const int* bottom_channels,
     const int* bottom_heights, const int* bottom_widths, const int sample_pernum,
     const int top_channels,  const int* sampling_list,
     const int W, Dtype* const top_diff) {
  // backward hypercolumns, seperate for each bottom
    CUDA_KERNEL_LOOP(index, nthreads) {

    }
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   // backward step, back the diff in top[0] to the bottom, except bottom[0]
    const Dtype* top_diff = top[0]->gpu_diff();
    vector<Dtype*> bottom_diffs;
    for (int i = 1; i < bottom.size(); ++i) {
        bottom_diffs.push_back(bottom[i]->mutable_gpu_diff());
    }
    const int nthreads = N_ * sample_num_ * total_channels_;
    BackwardHypercolumns<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, &bottom_diffs[0], cuda_channels_, cuda_heights_, cuda_widths_, sample_num_,
        total_channels_, cuda_samplelist_, W_, top_diff
    );
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::instance_cuda_data() {
    // instance the sampling list
    CUDA_CHECK(cudaMalloc(&cuda_samplelist_, selected_points_.size() * sizeof(int)));
    // instance the width, height and channel
    CUDA_CHECK(cudaMalloc(&cuda_widths_, width_.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(cuda_widths_, &width_[0], width_.size()* sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cuda_heights_, height_.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(cuda_heights_, &height_[0], height_.size()* sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cuda_channels_, channels_.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(cuda_channels_, &channels_[0], channels_.size()* sizeof(int)));
    CUDA_POST_KERNEL_CHECK;
    cuda_instanced_ = true;
}

HyperColumnsLayer()::~HyperColumnsLayer() {
    CUDA_CHECK(cudaFree(cuda_samplelist_));
    CUDA_CHECK(cudaFree(cuda_widths_));
    CUDA_CHECK(cudaFree(cuda_heights_));
    CUDA_CHECK(cudaFree(cuda_channels_));
    CUDA_POST_KERNEL_CHECK;
    cuda_instanced_ = false;
}




INSTANTIATE_LAYER_GPU_FUNCS(HyperColumnsLayer);
}// namespace caffe
