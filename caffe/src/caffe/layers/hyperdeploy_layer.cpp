/*************************************************************************
	> File Name: hyperdeploy_layer.cpp
	> Author: 
	> Mail: 
	> Created Time: 2016年09月12日 星期一 12时15分28秒
 ************************************************************************/
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

#include "caffe/layers/hyperdeploy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void HyperDeployLayer<Dtype>::get_map_point(std::map<int, double>& result,
        int out_index, const vector<int>& original_map_size) {
    /***
     * This function uses the bilinear interpolation to locate the corresponding
     * point in the original feature map with their corresponding weights.
     * note, since the orinal size is known, represents the coordinates by a single
     * value
     * */
   double scale = H_ / original_map_size[0];
   int x = out_index / H_;
   int y = out_index % H_;
   int h = original_map_size[0];
   int w = original_map_size[1];
   double r = x / scale + 1.0 / (2.0 * scale) - 0.5;
   double c = y / scale + 1.0 / (2.0 * scale) - 0.5;
   int u = floor(r);
   int v = floor(c);
   double delta_r = r - u;
   double delta_c = c - v;
   if (u < 0)
       delta_r = 1;
   if (u + 1 >= h)
       delta_r = 0;
   if (v < 0)
       delta_c = 1;
   if (v + 1 >= w)
       delta_c = 0;
   result.clear();
   if ((1-delta_r) * (1-delta_c) != 0)
       result.insert(std::make_pair(u * w + v, 
                   (1-delta_r)* (1-delta_c)));
   if (delta_r * (1-delta_c) != 0)
       result.insert(std::make_pair((u+1)*w + v,
                   delta_r * (1-delta_c)));
   if (delta_c * (1-delta_r) != 0)
       result.insert(std::make_pair(u * w + v + 1,
                   delta_c * (1-delta_r)));
   if (delta_r * delta_c != 0)
       result.insert(std::make_pair((u+1)*w + v + 1,
                   delta_r * delta_c));

}


template <typename Dtype>
void HyperDeployLayer<Dtype>::generate_list(vector<int>& result, 
        const Blob<Dtype>* feature_map, int batch) {
    int h = feature_map->shape(2);
    int w = feature_map->shape(3);
    vector<int> holds(h * w);
    for (int i = 0; i < holds.size(); ++i) {
        holds[i] = i;
    }
    result = holds;
}


template <typename Dtype>
void HyperDeployLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // setup layer params. note that the hypercolumns is M * K. where M is the sampled point,
    // K is the feature length.
    // bottom: normal_map, conv1_2, conv2_2, conv3_3, conv4_3, conv5_3, fc-conv7. length of 7
    // top: top[0] hypercolumns, top[1] the corresponding sampled normal point
    const Blob<Dtype>* normal_map = bottom[0];
    N_ = normal_map->shape(0);
    K_ = normal_map->shape(1);
    H_ = normal_map->shape(2);
    W_ = normal_map->shape(3);
    sample_num_ = H_ * W_;
    int channel = 0;
    for (int i = 0; i < bottom.size(); ++i) {
        channel += bottom[i]->shape(1);
    }
    total_channels_ = channel;
}


template <typename Dtype>
void HyperDeployLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // reshape the top
    // top[0] is the hypercolumns, which is M * K, where M = sample_num_ * N, K = channels_
    vector<int> top_shape;
    // top[0]
    top_shape.push_back(sample_num_ * N_);
    top_shape.push_back(total_channels_);
    top_shape.push_back(1);
    top_shape.push_back(1);
    top[0]->Reshape(top_shape);
}


template <typename Dtype>
void HyperDeployLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // forward step
    Dtype* top_hypercolumns = top[0]->mutable_cpu_data();
    vector<int> sampling_list;
    caffe_set(top[0]->count(), Dtype(0), top_hypercolumns);

    // for each batch, do the sampling job
    for (int n = 0; n < N_; ++n) {
        generate_list(sampling_list, bottom[0], n);
        // for every sampling point
        for (int id = 0; id < sampling_list.size(); ++id) {
            const int index = sampling_list[id];
            // hyperfeature next
            int hyper_channel = 0;
            for (int bottom_id = 0; bottom_id < bottom.size(); ++bottom_id) {
                const Dtype* bottom_data = bottom[bottom_id]->cpu_data();
                const int channel = bottom[bottom_id]->shape(1);
                // get the correponding point in the corresponding bottom
                vector<int> original_size;
                original_size.push_back(bottom[bottom_id]->shape(2));
                original_size.push_back(bottom[bottom_id]->shape(3));
                std::map<int, double> weights;
                get_map_point(weights, index, original_size);
                // for debug usage, output the last and first to check
                /**
                LOG(INFO) << "for sample point " << id ;
                for (std::map<int, double>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
                    LOG(INFO) << "the coresponding to bottom " << bottom_id << " point is " << iter->first << " weights is " << iter->second << std::endl;
                }
                **/
                for (int c = 0; c < channel; ++c){
                    // compute the value, according to the point
                    double value = 0;
                    for (std::map<int, double>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
                        const int bottom_index = bottom[bottom_id]->offset(n, c) + iter->first;
                        value += bottom_data[bottom_index] * iter->second;
                    }
                    const int top_index = top[0]->offset(n * sample_num_  + id, hyper_channel);
                    top_hypercolumns[top_index] = value;
                    ++hyper_channel;
                }
            }
        }
    }
   // LOG(INFO) << "forward done";
}

template <typename Dtype>
void HyperDeployLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom) {
    // do nothing
}



}//namespace caffe

