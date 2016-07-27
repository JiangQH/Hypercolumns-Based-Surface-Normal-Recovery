#ifndef CAFFE_HYPERCOLUMNS_LAYER_HPP_
#define CAFFE_HYPERCOLUMNS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * this is a self design layer to implement some self functions.
 * mainly the feature used to train a multinet work
 * @Jiang Qinhong 
 * mylivejiang@gmail.com
 *
 *
 * */
template <typename Dtype>
class HyperColumnsLayer: public Layer<Dtype> {
public:
    explicit HyperColumnsLayer(const LayerParameter& param) :
        Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "HyperColumns"; }
    virtual int MinBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);

//    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                            const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

//    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    vector<int> selected_points_; // the selected point
    bool is_train_;
    int N_, K_, H_, W_; // the N, K, H, W of normal map
    int sample_num_; // sample_num per batch
    int channels_; // the channels_ of the hypercolumns

private:
    void get_map_point(std::map<int, double>& result, int out_index,
                       const vector<int>& original_size); // generate the correponding map point    
    void generate_list(vector<int>& result, const Blob<Dtype>* feature_map, int batch); // generate random list
    
    double get_true_normal(const double normal_map);// get the true normal value according to the  normal_map point

    bool is_valid(const Blob<Dtype>* feature_map, int batch, int index);
};// end of HyperColumnsLayer
}// namespace caffe
#endif // CAFFE_HYPERCOLUMNS_LAYER_HPP_
