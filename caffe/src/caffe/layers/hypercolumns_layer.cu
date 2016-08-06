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
    const Dtype* bottom_data, const int num, const int bottom_channels,
    const int bottom_height, const int bottom_width, const int sample_pernum,
    const int top_channels, const int top_channel_offset, const int* sampling_list,
    const int original_h, Dtype* const top_data) {
    //forward hypercolumns, separate for each bottom
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int top_n = index / bottom_channels;
      const int bottom_n = top_n / sample_pernum;
      const int bottom_channel = index % bottom_channels; // get the actual channel of the bottom
      const int top_index = bottom_channel + top_n * top_channels + top_channel_offset;
      const int sample_index = sampling_list[top_n];
      const Dtype* const bottom_slice = bottom_data + (bottom_n * bottom_channels + bottom_channel) * bottom_height * bottom_width;
      // get the corresponding bottom_index, according to the top. using bilinear intercept
      const double scale = original_h * 1.0 / bottom_height;
      const int x = sample_index / original_h;
      const int y = sample_index % original_h;
      const double r = x / scale + 1.0 / (2.0 * scale) - 0.5;
      const double c = y / scale + 1.0 / (2.0 * scale) - 0.5;
      const int u = floor(r);
      const int v = floor(c);
      double delta_r = r - u;
      double delta_c = c - v;
      if (u < 0)
        delta_r = 1;
      if (u + 1 >= bottom_height)
        delta_r = 0;
      if (v < 0)
        delta_c = 1;
      if (v + 1 >= bottom_width)
        delta_c = 0;
      // assign the value, notice the boundary check
      double value = 0;
      if ((1 - delta_r) * (1 - delta_c) != 0)
        value += bottom_slice[u * bottom_width + v] * (1 - delta_r) * (1 - delta_c);
      if (delta_r * (1 - delta_c) != 0)
        value += bottom_slice[(u+1) * bottom_width + v] * delta_r * (1 - delta_c);
      if (delta_c * (1 - delta_r) != 0)
        value += bottom_slice[u * bottom_width + v + 1] * delta_c * (1 - delta_r);
      if (delta_r * delta_c != 0)
        value += bottom_slice[(u+1) * bottom_width + v + 1] * delta_r * delta_c;
      top_data[top_index] = value;
    }
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    // generate the sampling list
    selected_points_.clear();
    vector<int> sampling_list;
    for (int n = 0; n < N_; ++n) {
        generate_list(sampling_list, bottom[0], n);
        selected_points_.insert(selected_points_.end(),
          sampling_list.begin(), sampling_list.end());
    }

    // forward step, forward normal first
    int* cuda_samplelist;
    cudaMalloc(&cuda_samplelist, selected_points_.size() * sizeof(int));
    cudaMemcpy(cuda_samplelist, &selected_points_[0], selected_points_.size()*sizeof(int), cudaMemcpyHostToDevice);

    Dtype* top_normal = top[1]->mutable_gpu_data();
    const Dtype* bottom_normal = bottom[0]->gpu_data();
    const int count1 = top[1]->count();
    ForwardNormal<Dtype><<<CAFFE_GET_BLOCKS(count1), CAFFE_CUDA_NUM_THREADS>>>(
      count1, bottom_normal, N_, K_, H_, W_, sample_num_, cuda_samplelist, top_normal
    );

    // then forward the hypercolumns
    Dtype* top_hypercolumns = top[0]->mutable_gpu_data();
    int top_channel_offset = 0;
   // const int count0 = top[0]->count();
    //const int top_total_channels = top[0]->shape(1);
    for (int i = 1; i < bottom.size(); ++i) {
      // do it according the corresponding bottom
      const Dtype* bottom_data = bottom[i]->gpu_data();
      const int bottom_channels = bottom[i]->shape(1);
      const int bottom_height = bottom[i]->shape(2);
      const int bottom_width = bottom[i]->shape(3);
      const int nthreads = N_ * sample_num_ * bottom_channels;
      ForwardHypercolumns<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_data, N_, bottom_channels, bottom_height, bottom_width, sample_num_,
        total_channels_, top_channel_offset, cuda_samplelist, H_, top_hypercolumns
      );
      top_channel_offset += bottom_channels;
    }
    cudaFree(cuda_samplelist);
    CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
__global__ void BackwardHypercolumns(const int nthreads,
    Dtype* const bottom_data, const int num, const int bottom_channels,
    const int bottom_height, const int bottom_width, const int sample_pernum,
    const int top_channels, const int top_channel_offset, const int* sampling_list,
    const int original_h, const Dtype* const top_data) {
  // backward hypercolumns, seperate for each bottom
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int top_n = index / bottom_channels;
      const int bottom_n = top_n / sample_pernum;
      const int bottom_channel = index % bottom_channels;
      const int top_index = bottom_channel + top_n * top_channels + top_channel_offset;
      const int sampling_index = sampling_list[top_n];
      Dtype* bottom_slice = bottom_data + (bottom_n * bottom_channels + bottom_channel) * bottom_height * bottom_width;
      // back, get the corresponding bottom index and do the job
      const double scale = original_h * 1.0 / bottom_height;
      const int x = sampling_index / original_h;
      const int y = sampling_index % original_h;
      const double r = x / scale + 1.0 / (2.0 * scale) - 0.5;
      const double c = y / scale + 1.0 / (2.0 * scale) - 0.5;
      const int u = floor(r);
      const int v = floor(c);
      double delta_r = r - u;
      double delta_c = c - v;
      if (u < 0)
        delta_r = 1;
      if (u + 1 >= bottom_height)
        delta_r = 0;
      if (v < 0)
        delta_c = 1;
      if (v + 1 >= bottom_width)
        delta_c = 0;
      // notice the boundary check
      if ((1 - delta_r) * (1 - delta_c) != 0)
        bottom_slice[u * bottom_width + v] += top_data[top_index] * (1 - delta_r) * (1 - delta_c);
      if (delta_r * (1 - delta_c) != 0)
        bottom_slice[(u+1) * bottom_width + v] += top_data[top_index] * delta_r * (1 - delta_c);
      if (delta_c * (1 - delta_r) != 0)
        bottom_slice[u * bottom_width + v + 1] += top_data[top_index] * delta_c * (1 - delta_r);
      if (delta_r * delta_c != 0)
        bottom_slice[(u+1) * bottom_width + v + 1] += top_data[top_index] * delta_r * delta_c;
    }
}



template <typename Dtype>
void HyperColumnsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   // backward step, back the diff in top[0] to the bottom, except bottom[0]
    int* cuda_samplelist;
    cudaMalloc(&cuda_samplelist, selected_points_.size() * sizeof(int));
    cudaMemcpy(cuda_samplelist, &selected_points_[0], selected_points_.size()*sizeof(int), cudaMemcpyHostToDevice);

    const Dtype* top_diff = top[0]->gpu_diff();
    int top_channel_offset = 0;
    //const int count = top[0]->count();
    for (int i = 1; i < bottom.size(); ++i) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      const int bottom_channels = bottom[i]->shape(1);
      const int bottom_height = bottom[i]->shape(2);
      const int bottom_width = bottom[i]->shape(3);
      const int nthreads = N_ * sample_num_ * bottom_channels;
      BackwardHypercolumns<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_diff, N_, bottom_channels, bottom_height, bottom_width, sample_num_,
        total_channels_, top_channel_offset, cuda_samplelist, H_, top_diff
      );
      top_channel_offset += bottom_channels;
    }

    cudaFree(cuda_samplelist);
    CUDA_POST_KERNEL_CHECK;
}




INSTANTIATE_LAYER_GPU_FUNCS(HyperColumnsLayer);
}// namespace caffe
