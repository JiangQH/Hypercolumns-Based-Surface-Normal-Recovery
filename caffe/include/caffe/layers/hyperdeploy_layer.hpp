#ifndef CAFFE_HYPERDEPLOY_LAYER_HPP_
#define CAFFE_HYPERDEPLOY_LAYER_HPP_
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * this is a deploy layer used only when do deploy output
 **/ 
template <typename Dtype>
class HyperDeployLayer: public Layer<Dtype> {
public:
    explicit HyperDeployLayer(const LayerParameter& param) :
        Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "HyperDeploy"; }

    virtual int MinBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
    
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);

    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);


    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    // Note the gpu version is to implement. make sure the cpu version work first

    vector<int> selected_points_; // the selected point
    int N_, K_, H_, W_; // the N, K, H, W of normal map
    int sample_num_; // sample_num per batch
    int total_channels_; // the channels_ of the hypercolumns

    
private:
    void get_map_point(std::map<int, double>& result, int out_index,
                       const vector<int>& original_size); // generate the correponding map point    

    void generate_list(vector<int>& result, const Blob<Dtype>* feature_map, int batch); // generate random list

};//end of class
}//namespace caffe
#endif
