/*************************************************************************
	> File Name: hypercolumns_layer.cpp
	> Author: Jiang Qinhong
	> Mail: mylivejiang@gmail.com
	> Created Time: 2016年06月06日 星期一 19时40分04秒
 ************************************************************************/
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

#include "caffe/layers/hypercolumns_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HyperColumnsLayer<Dtype>::get_map_point(std::map<int, double>& result,
        int out_index, const vector<int>& original_map_size) {
    /***
     * This function uses the bilinear interpolation to locate the corresponding
     * point in the original feature map with their corresponding weights.
     * note, since the orinal size is known, represents the coordinates by a single
     * value
     * */
   double scale = H_ / original_map_size[0];
   int x = out_index / W_; 
   int y = out_index % W_;
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
void HyperColumnsLayer<Dtype>::generate_list(vector<int>& result, 
        const Blob<Dtype>* feature_map, int batch) {
    // generate sampling list. when training, 1k. when testing whole
    // make sure when training, the point is valid
    int h = feature_map->shape(2);
    int w = feature_map->shape(3);
    vector<int> holds(h * w);
    for (int i = 0; i < holds.size(); ++i) {
        holds[i] = i;
    }
    if (!is_train_) {
        result = holds;
    } else {
        // shuffle the vector first
        srand(time(NULL));
        std::random_shuffle(holds.begin(), holds.end());
        // do the job
        result.clear();
        int count = 0;
        for (int i = 0; i < holds.size(); ++i) {
            if (is_valid(feature_map, batch, holds[i])) { 
                result.push_back(holds[i]);
                ++count;
            }
            if (count == 1000)
                break;
        }
    }

}

template <typename Dtype>
bool HyperColumnsLayer<Dtype>::is_valid(const Blob<Dtype>* feature_map, 
        int number, int index) {
    // to indicate whether the sampled point is valid in normal feature_map
    const Dtype* feature_data = feature_map->cpu_data();
    const int offset1 = feature_map->offset(number, 0);
    const int offset2 = feature_map->offset(number, 1);
    const int offset3 = feature_map->offset(number, 2);
    const double value1 = get_true_normal(feature_data[offset1+index]);
    const double value2 = get_true_normal(feature_data[offset2+index]);
    const double value3 = get_true_normal(feature_data[offset3+index]);
    return std::abs(value1 * value1 + value2 * value2 + value3 * value3 - 1.0) < 1e-1;
}

template <typename Dtype>
double HyperColumnsLayer<Dtype>::get_true_normal(const double normal_map) {
    double result = normal_map * (2 * 0.00390625);
    result -= 1;
    return result;
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // setup layer params. note that the hypercolumns is M * K. where M is the sampled point,
    // K is the feature length.
    // bottom: normal_map, conv1_2, conv2_2, conv3_3, conv4_3, conv5_3, fc-conv7. length of 7
    // top: top[0] hypercolumns, top[1] the corresponding sampled normal point
    is_train_ = this->layer_param_.hypercolumns_param().is_train();
    const Blob<Dtype>* normal_map = bottom[0];
    N_ = normal_map->shape(0);
    K_ = normal_map->shape(1);
    H_ = normal_map->shape(2);
    W_ = normal_map->shape(3);
    if (is_train_)
        sample_num_ = 1000;
    else
        sample_num_ = H_ * W_;
    int channel = 0;
    for (int i = 1; i < bottom.size(); ++i) {
        channel += bottom[i]->shape(1);
    }
    channels_ = channel;
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // reshape the top
    // top[0] is the hypercolumns, which is M * K, where M = sample_num_ * N, K = channels_
    // top[1] is the sampled normal point, which is M * 3
    vector<int> top_shape;
    // top[0]
    top_shape.push_back(sample_num_ * N_);
    top_shape.push_back(channels_);
    top[0]->Reshape(top_shape);
    // top[1]
    top_shape[1] = 3;
    top[1]->Reshape(top_shape);
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // forward step
    Dtype* top_normal = top[1]->mutable_cpu_data();
    Dtype* top_hypercolumns = top[0]->mutable_cpu_data();
    const Dtype* bottom_normal = bottom[0]->cpu_data();
    vector<int> sampling_list;
    caffe_set(top[1]->count(), Dtype(-1.0), top_normal);
    caffe_set(top[0]->count(), Dtype(0), top_hypercolumns);

    // for each batch, do the sampling job
    for (int n = 0; n < N_; ++n) {
        generate_list(sampling_list, bottom[0], n);
        if (is_train_) {
            selected_points_.insert(selected_points_.end(),
                    sampling_list.begin(), sampling_list.end());
        }
        // for every sampling point
        for (int id = 0; id < sampling_list.size(); ++id) {
            const int index = sampling_list[id];
            // normal first
            for (int c = 0; c < top[1]->shape(1); ++c) {
                const int top_index = top[1]->offset(n * sample_num_ + id, c);
                const int bottom_index = bottom[0]->offset(n,c) + index;
                top_normal[top_index] = get_true_normal(bottom_normal[bottom_index]);
            }
            // hyperfeature next
            int hyper_channel = 0;
            for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
                const Dtype* bottom_data = bottom[bottom_id]->cpu_data();
                const int channel = bottom[bottom_id]->shape(1);
                // get the correponding point in the corresponding bottom
                vector<int> original_size;
                original_size.push_back(bottom[bottom_id]->shape(2));
                original_size.push_back(bottom[bottom_id]->shape(3));
                std::map<int, double> weights;
                get_map_point(weights, index, original_size);
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
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom) {
    // backward step
    const Dtype* top_diff = top[0]->cpu_diff();
    // first set the value to zero for every bottom diff
    for (int i = 1; i < bottom.size(); ++i) {
        caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
    }
    
    // do backward
    for (int index = 0; index < selected_points_.size(); ++index) {
        const int selected_index = selected_points_[index];
        const int n = index / sample_num_;
        int hyper_channel = 0;
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
            Dtype* bottom_diff = bottom[bottom_id]->mutable_cpu_diff();
            const int channel = bottom[bottom_id]->shape(1);
            vector<int> original_size;
            original_size.push_back(bottom[bottom_id]->shape(2));
            original_size.push_back(bottom[bottom_id]->shape(3));
            std::map<int, double> weights;
            get_map_point(weights, selected_index, original_size);
            for (int c = 0; c < channel; ++c) {
                const int top_index = top[0]->offset(index, hyper_channel);
                for (std::map<int, double>::iterator iter = weights.begin();
                     iter != weights.end(); ++iter) {
                    const int bottom_index = bottom[bottom_id]->offset(n, c) + iter->first;
                    bottom_diff[bottom_index] += top_diff[top_index] * iter->second;
                }
                ++hyper_channel;
            }
        }
    }
}

INSTANTIATE_CLASS(HyperColumnsLayer);
REGISTER_LAYER_CLASS(HyperColumns);
}// namespace caffe
