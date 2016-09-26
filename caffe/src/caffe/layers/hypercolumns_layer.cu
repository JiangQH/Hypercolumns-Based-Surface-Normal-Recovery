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
    const int bottom_count, const Dtype** bottom_datas, const int* bottom_channels,
    const int* bottom_heights, const int* bottom_widths, const double* bottom_maplists,
    const int sample_pernum, const int top_channels,  const int* sampling_list,
    const int W, Dtype* const top_data) {
    //forward hypercolumns, separate for each bottom
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int top_n = index / top_channels; // find the corresponding index in the sampling list
        const int bottom_n = top_n / sample_pernum;
        int bottom_channel = index % top_channels;
        int bottom_id = 0;
        while(bottom_id<bottom_count) {
            if(bottom_channel - bottom_channels[bottom_id] < 0) {
                break;
            }
            bottom_channel -= bottom_channels[bottom_id];
            ++bottom_id;
        }
        // now have the bottom_id, bottom_num, bottom_channel. needs to get the corresponding bottom feature map point
        const int sampled_index = sampling_list[top_n];
        const int startid = (sampled_index * bottom_count + bottom_id) * 6; // hard coding here
        double tempw = bottom_maplists[startid];
        double temph = bottom_maplists[startid+1];
        int fw = bottom_maplists[startid+2];
        int fh = bottom_maplists[startid+3];
        int cw = bottom_maplists[startid+4];
        int ch = bottom_maplists[startid+5];
        // assign values
        int padding = bottom_heights[bottom_id] * bottom_widths[bottom_id];
        int slice = (bottom_n * bottom_channels[bottom_id] + bottom_channel)* padding;

        const Dtype* bottom_data = bottom_datas[bottom_id];
        if ((fw == cw) && (fh == ch)) {
            int offset = slice + fh * bottom_widths[bottom_id] + fw;
            top_data[index] = bottom_data[offset];
        }
        else if (fh == ch) {
            double delta_w = tempw - fw;
            int offset1 = slice + fh * bottom_widths[bottom_id] + fw;
            int offset2 = offset1 + 1;
            top_data[index] = bottom_data[offset1] * (1-delta_w) + bottom_data[offset2] * delta_w;
        }
        else if (fw == cw) {
            double delta_h = temph - fh;
            int offset1 = slice + fh * bottom_widths[bottom_id] + fw;
            int offset2 = offset1 + bottom_widths[bottom_id];
            top_data[index] = bottom_data[offset1] * (1-delta_h) + bottom_data[offset2] * delta_h;
        }
        else {
            double delta_w = tempw - fw;
            double delta_h = temph - fh;
            int offset1 = slice + fh * bottom_widths[bottom_id] + fw;
            int offset2 = offset1 + 1;
            int offset3 = offset1 + bottom_widths[bottom_id];
            int offset4 = offset3 + 1;
            top_data[index] =
                        (bottom_data[offset1]*(1-delta_h) + bottom_data[offset3]*(delta_h)) * (1-delta_w) +
                        (bottom_data[offset2]*(1-delta_h) + bottom_data[offset4]*(delta_h)) * delta_w;
        }
    }
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    // generate sampling list
    generate_list(bottom[0], false);
    // check and instance the cuda needed data
    if (!cuda_instanced_) {
        instance_cuda_data();
        CUDA_CHECK(cudaMalloc(&cuda_samplelist_, selected_points_.size() * sizeof(int)));
    }
    // generate the sampling list and copy it
    CUDA_CHECK(cudaMemcpy(cuda_samplelist_, &selected_points_[0], selected_points_.size()* sizeof(int), cudaMemcpyHostToDevice));

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
    // here, in order to save time. I have to decide to use the hard coding
    // which means I will 
    // a bug. cannot use vector here
    const int nthreads = N_ * sample_num_ * total_channels_;
    ForwardHypercolumns<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_count, bottom_datas, cuda_channels_, cuda_heights_, cuda_widths_, cuda_map_lists_,
        sample_num_, total_channels_, cuda_samplelist_, W_, top_hypercolumns
    );

}




template <typename Dtype>
__global__ void BackwardHypercolumns(const int nthreads,
     const int bottom_count, Dtype** const bottom_diffs, const int* bottom_channels,
     const int* bottom_heights, const int* bottom_widths, const double* bottom_maplists, const int sample_pernum,
     const int top_channels,  const int* sampling_list,
     const int W, const Dtype* const top_diff) {
  // backward hypercolumns, seperate for each bottom
    CUDA_KERNEL_LOOP(index, nthreads) {
        const int top_n = index / top_channels; // find the corresponding index in the sampling list
        const int bottom_n = top_n / sample_pernum;
        int bottom_channel = index % top_channels;
        int bottom_id = 0;
        while(bottom_id<bottom_count) {
            if(bottom_channel - bottom_channels[bottom_id] < 0) {
                break;
            }
            bottom_channel -= bottom_channels[bottom_id];
            ++bottom_id;
        }
        // now have the bottom_id, bottom_num, bottom_channel. needs to get the corresponding bottom feature map point
        const int sampled_index = sampling_list[top_n];
        const int startid = (sampled_index * bottom_count + bottom_id) * 6; // hard coding here
        double tempw = bottom_maplists[startid];
        double temph = bottom_maplists[startid+1];
        int fw = bottom_maplists[startid+2];
        int fh = bottom_maplists[startid+3];
        int cw = bottom_maplists[startid+4];
        int ch = bottom_maplists[startid+5];
        // assign values
        int padding = bottom_heights[bottom_id] * bottom_widths[bottom_id];
        int slice = (bottom_n * bottom_channels[bottom_id] + bottom_channel)* padding;
        Dtype* bottom_diff = bottom_diffs[bottom_id];
        if ((fw == cw) && (fh == ch)) {
            int offset = slice + fh * bottom_widths[bottom_id] + fw;
            bottom_diff[offset] += top_diff[index];
        }
        else if (fh == ch) {
            double delta_w = tempw - fw;
            int offset1 = slice + fh * bottom_widths[bottom_id]+ fw;
            int offset2 = offset1 + 1;
            bottom_diff[offset1] += top_diff[index] * (1-delta_w);
            bottom_diff[offset2] += top_diff[index] * delta_w;
        }
        else if (fw == cw) {
            double delta_h = temph - fh;
            int offset1 = slice + fh * bottom_widths[bottom_id] + fw;
            int offset2 = offset1 + bottom_widths[bottom_id];
            bottom_diff[offset1] += top_diff[index] * (1 - delta_h);
            bottom_diff[offset2] += top_diff[index] * delta_h;
        }
        else {
            double delta_w = tempw - fw;
            double delta_h = temph - fh;
            int offset1 = slice + fh * bottom_widths[bottom_id] + fw;
            int offset2 = offset1 + 1;
            int offset3 = offset1 + bottom_widths[bottom_id];
            int offset4 = offset3 + 1;
            bottom_diff[offset1] += top_diff[index] * (1-delta_w) * (1-delta_h);
            bottom_diff[offset2] += top_diff[index] * (1-delta_h) * delta_w;
            bottom_diff[offset3] += top_diff[index] * delta_h * (1-delta_w);
            bottom_diff[offset4] += top_diff[index] * delta_h * delta_w;
        }

    }
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   // backward step, back the diff in top[0] to the bottom, except bottom[0]
    const Dtype* top_diff = top[0]->gpu_diff();
    vector<Dtype*> bottom_diffs;
    const int bottom_count = bottom.size() - 1;
    for (int i = 1; i < bottom.size(); ++i) {
        bottom_diffs.push_back(bottom[i]->mutable_gpu_diff());
    }
    const int nthreads = N_ * sample_num_ * total_channels_;
    BackwardHypercolumns<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, bottom_count, &bottom_diffs[0], cuda_channels_, cuda_heights_, cuda_widths_, cuda_map_lists_, sample_num_,
        total_channels_, cuda_samplelist_, W_, top_diff
    );
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::instance_cuda_data() {
    // instance the width, height and channel
    CUDA_CHECK(cudaMalloc(&cuda_widths_, width_.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(cuda_widths_, &width_[0], width_.size()* sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&cuda_heights_, height_.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(cuda_heights_, &height_[0], height_.size()* sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&cuda_channels_, channels_.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(cuda_channels_, &channels_[0], channels_.size()* sizeof(int), cudaMemcpyHostToDevice));
    generate_bilinear_map(); // generate the mapping list
    CUDA_POST_KERNEL_CHECK;
    cuda_instanced_ = true;
}

/**
template <typename Dtype>
HyperColumnsLayer<Dtype>::~HyperColumnsLayer() {
    CUDA_CHECK(cudaFree(cuda_samplelist_));
    CUDA_CHECK(cudaFree(cuda_widths_));
    CUDA_CHECK(cudaFree(cuda_heights_));
    CUDA_CHECK(cudaFree(cuda_channels_));
    CUDA_CHECK(cudaFree(cuda_map_lists_));
    CUDA_POST_KERNEL_CHECK;
    cuda_instanced_ = false;
}
**/

template <typename Dtype>
void HyperColumnsLayer<Dtype>::generate_bilinear_map() {
// generate the bilinear map all in one at begin
    const int total_index = H_ * W_;
    const int bottom_count = width_.size() - 1;
    CUDA_CHECK(cudaMalloc(&cuda_map_lists_, 6 * bottom_count * total_index * sizeof(double)));
    // get the value for every sample index
    int h, w;
    double fw, fh, cw, ch;
    double tempw, temph;
    vector<double> mappings;
    for (int index = 0; index < total_index; ++index) {
        h = index / W_;
        w = index % W_;
        for (int b = 1; b < width_.size(); ++b) {
            tempw = (w - padf_[b]) / scalef_[b];
            temph = (h - padf_[b]) / scalef_[b];
            fw = static_cast<int>(floor(tempw));
            fh = static_cast<int>(floor(temph));
            cw = static_cast<int>(ceil(tempw));
            ch = static_cast<int>(ceil(temph));
            // boundary check
            fw = fw > 0 ? fw : 0;
            cw = cw > 0 ? cw : 0;
            fh = fh > 0 ? fh : 0;
            ch = ch > 0 ? ch : 0;
            cw = cw < width_[b] ? cw : fw;
            ch = ch < height_[b] ? ch : fh;
            mappings.push_back(tempw);
            mappings.push_back(temph);
            mappings.push_back(fw);
            mappings.push_back(fh);
            mappings.push_back(cw);
            mappings.push_back(ch);
        }
    }
    CUDA_CHECK(cudaMemcpy(cuda_map_lists_, &mappings[0], 6 * bottom_count * total_index * sizeof(double), cudaMemcpyHostToDevice));
}




INSTANTIATE_LAYER_GPU_FUNCS(HyperColumnsLayer);
}// namespace caffe
