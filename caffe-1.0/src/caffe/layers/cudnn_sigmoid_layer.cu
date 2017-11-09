#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  CUDNN_CHECK(cudnnActivationForward(this->handle_,
      CUDNN_ACTIVATION_SIGMOID,
      this->bottom_desc_, bottom_data, this->top_desc_, top_data));
}

template <typename Dtype>
void CuDNNSigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  CUDNN_CHECK(cudnnActivationBackward(this->handle_,
      CUDNN_ACTIVATION_SIGMOID,
      this->top_desc_, top_data, this->top_desc_, top_diff,
      this->bottom_desc_, bottom_data, this->bottom_desc_, bottom_diff));
}

INSTANTIATE_CLASS(CuDNNSigmoidLayer);

}  // namespace caffe
#endif
