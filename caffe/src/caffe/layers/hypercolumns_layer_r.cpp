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
void HyperColumnsLayer<Dtype>::generate_list(const Blob<Dtype>* feature_map,
        bool is_gpu) {
    // generate sampling list. when training do random sampling. when testing whole
    // make sure when training, the point is valid
    // sample them all in one
    selected_points_.clear();
    if (!is_train_) {
        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < sample_num_; ++j) {
                selected_points_.push_back(j);
            }
        }
    } 
    else {
        // do the random sample job
        vector<int> holds;
        for(int i = 0; i < H_*W_; ++i) {
            holds.push_back(i);
        }
        // the sample seed
        if (!is_gpu) {
            const Dtype* feature_data = feature_map->cpu_data();// for cpu
        }
        else {
            const Dtype* feature_data = feature_map->gpu_data(); // for gpu 
        }
        std::srand(NULL);
        for (int i = 0; i < N_; ++i) {
            std::random_shuffle(holds.begin(), holds.end());
            int count = 0;
            for (int j = 0; j < holds.size(); ++j) {
                const int index = holds[j];
                // check whether it is valid, and the value of each channel not all be zero
                bool valid = true;
                int zero_count = 0;
                for (int c = 0; c < K_; ++c) {
                    const int offset = feature_map->offset(i, c);
                    const Dtype value = feature_data[offset+index];
                    if (value != value) {
                        valid = false;
                        break;
                    }
                    if (value == Dtype(0.0)) {
                        ++zero_count;
                    }
                }
                if (valid && zero_count != K_) {
                    selected_points_.push_back(index);
                    ++count;
                }
                if (count > sample_num_) {
                    break;
                }
            }
        }
    }
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
        sample_num_ = this->layer_param_.hypercolumns_param().sample_num();
    else
        sample_num_ = H_ * W_;
    // note here I make the normal be bottom 0, and push it to the data store
    // to make consistence
    width_.push_back(bottom[0]->shape(2));
    height_.push_back(bottom[0]->shape(3));
    scalef_.push_back(1);
    padf_.push_back(0.0);
    int channel = 0;
    for (int i = 1; i < bottom.size(); ++i) {
        channel += bottom[i]->shape(1);
        width_.push_back(bottom[i]->shape(3);
        height_.push_back(bottom[i]->shape(2));
        const int scale = H_ / bottom[i]->shape(2);
        scalef_.push_back(scale);
        padf_.push_back(static_cast<Dtype>((scale-1.0)/2));
    }
    total_channels_ = channel;
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // reshape the top
    // top[0] is the hypercolumns, which is M * total_channels_, where M = sample_num_ * N_
    // top[1] is the sampled normal point, which is M * k
    vector<int> top_shape;
    // top[0]
    top_shape.push_back(sample_num_ * N_);
    top_shape.push_back(total_channels_);
    top[0]->Reshape(top_shape);
    // top[1]
    top_shape[1] = K_;
    top[1]->Reshape(top_shape);
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // forward step
    Dtype* top_normal = top[1]->mutable_cpu_data();
    Dtype* top_hypercolumns = top[0]->mutable_cpu_data();
    const Dtype* bottom_normal = bottom[0]->cpu_data();
    caffe_set(top[1]->count(), Dtype(0.0), top_normal);
    caffe_set(top[0]->count(), Dtype(0), top_hypercolumns);
    // generate sampling list 
    generate_list(bottom[0], false);
    // do the forward job
    int h, w, n, index;
    int fw, fh, cw, ch; // the floor and ceil elements
    Dtype tempw, temph; // the raw divide
    Dtype delta_w, delta_h;
    int top_hyper_index = 0;
    int top_n_index = 0;
    for (int i = 0; i < sample_num_*N_; ++i) {
        // for every top sampling feature point
        index = selected_points_[i]; // the index
        h = index / W_; // the w in the original
        w = index % W_; // the h in the original
        n = i % sample_num_; // the num

        // find the corresponding locations for every bottom
        for (int b = 1; b < bottom.size(); ++b) {
            const Dtype* bottom_data = bottom[b]->cpu_data;
            int slice = n * bottom[b]->channels() * height_[b] * width_[b];
            tempw = (w - padf_[b]) / scalef_[b]; // the computed 
            temph = (h - padf_[b]) / scalef_[b];
            fw = floor(tempw);
            fh = floor(temph);
            cw = ceil(tempw);
            ch = ceil(temph);
            // boundary check
            fw = fw > 0 ? fw : 0;
            cw = cw > 0 ? cw : 0;
            fh = fh > 0 ? fh : 0;
            ch = ch > 0 ? ch : 0;
            cw = cw < width_[b] ? cw : fw;
            ch = ch < height_[b] ? ch : fh;
            // assign values
            if ((fw == cw) && (fh == ch)) {
                offset = slice +  fh * width_[b] + fw;
                for (int c = 0; c < bottom[b]->channels(); ++c) {
                    top_hypercolumns[top_index++] = bottom_data[offset];
                    offset += width_[b] * height_[b];
                }
            }
            else if (fh == ch) {
                delta_w = tempw - fw;
                int offset1 = slice + fh * width_[b] + fw;
                int offset2 = offset1 + 1;
                for (int c = 0; c < bottom[b]->channels(); ++c) {
                    top_hypercolumns[top_index++] = 
                        bottom_data[offset1] * (1-delta_w) + bottom_data[offset2] * delta_w;
                    offset1 += width_[b] * height_[b];
                    offset2 += width_[b] * height_[b];
                }
            }
            else if (fw == cw) {
                delta_h = temph - fh;
                int offset1 = slice + fh * width_[b] + fw;
                int offset2 = offset1 + width_[b];
                for (int c = 0; c < bottom[b]->channels(); ++c) {
                    top_hypercolumns[top_index++] = 
                        bottom_data[offset1] * (1-delta_h) + bottom_data[offset2] * delta_h;
                    offset1 += width_[b] * height_[b];
                    offset2 += width_[b] * height_[b];
                }
            }
            else {
                delta_w = tempw - fw;
                delta_h = temph - fh;
                int offset1 = slice + fh * width_[b] + fw;
                int offset2 = offset1 + 1;
                int offset3 = offset1 + width_[b];
                int offset4 = offset3 + 1;
                for (int c = 0; c < bottom[b]->channels(); ++c) {
                    top_hypercolumns[top_index++] = 
                        (bottom_data[offset1]*(1-delta_h) + bottom_data[offset3]*(delta_h)) * (1-delta_w) + 
                        (bottom_data[offset2]*(1-delta_h) + bottom_data[offset4]*(delta_h)) * delta_w;
                    offset1 += width_[b] * height_[b];
                    offset2 += width_[b] * height_[b];
                    offset3 += width_[b] * height_[b];
                    offset4 += width_[b] * height_[b];
                }
            }
        }

        // sample the normal feature
        for (int c = 0; c < K_; ++c) {
            int offset = (n * K_ + c)*H_*W_ + index;
            top_normal[top_n_index++] = bottom_normal[offset];
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
    selected_points_.clear();
    // LOG(INFO) << "backward done ";
}

#ifdef CPU_ONLY
    STUB_GPU(HyperColumnsLayer);
#endif

INSTANTIATE_CLASS(HyperColumnsLayer);
REGISTER_LAYER_CLASS(HyperColumns);
}// namespace caffe
