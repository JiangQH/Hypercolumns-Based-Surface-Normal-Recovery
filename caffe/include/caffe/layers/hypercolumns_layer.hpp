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
        Layer<Dtype>(param) { cuda_instanced_ = false;}

    virtual ~HyperColumnsLayer();

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

    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);


    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    // Note the gpu version is to implement. make sure the cpu version work first

    vector<int> selected_points_; // the selected point
    bool is_train_;
    int N_, K_, H_, W_; // the N, K, H, W of normal map
    int sample_num_; // sample_num per batch
    int total_channels_; // the channels_ of the hypercolumns
    vector<int> width_, height_;
    vector<int> scalef_;
    vector<Dtype> padf_;
    vector<int> channels_; // store the channels for every bottom

    // for the use of gpu, I declare some elements here to avoid the multi-declare and save time
    int* cuda_samplelist_;
    int *cuda_widths_, *cuda_heights_, *cuda_channels_;
    double* cuda_map_lists_; // store the (tempw, temph, fh, fw, ch, cw) for every index in the sample map. so the size
                            // should be 6 * bottom_count * total_index
    bool cuda_instanced_;
    vector<double> mappings_;
private:



    void generate_bilinear_map(); // this function is used to do the map generation at the first time, to avoid multiple
                                  // computation in the gpu version

    void generate_list(const Blob<Dtype>* feature_map); // generate random list
    
    double get_true_normal(const double normal_map);// get the true normal value according to the  normal_map point

    bool is_valid(const Blob<Dtype>* feature_map, int batch, int index);

    void instance_cuda_data();
};// end of HyperColumnsLayer
}// namespace caffe
#endif // CAFFE_HYPERCOLUMNS_LAYER_HPP_
