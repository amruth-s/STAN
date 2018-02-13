#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

vector <float *> pruning1;
vector <float *> pruning2;
vector <float *> pruning3;
int epoch=0;
int images=0;

template <typename Dtype>
__global__ void Threshold_pruning(const int n,
    Dtype* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
	  if (in[index]>0.5)
		  out[index] =1234;
	  else
		  out[index] = in[index];
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const int p_t =0; 
	const int Z=1;
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    const int count = this->top_dim_;
    if (epoch==0 && this->phase_ == TRAIN){
	    float* d_x = NULL;
	    cudaMalloc(&d_x, sizeof(float)*count*this->num_);
	    cudaMemset(d_x,0,count*this->num_*sizeof(float));
	    if (count == 32768)//MNIST: 11520,3200
	         pruning1.push_back(d_x);
	    else if (count==8192)
		 pruning2.push_back(d_x);
	    else
        	 pruning3.push_back(d_x);
    }
    for (int n = 0; n < this->num_; ++n) {
      if (this->phase_ == TEST || ((this->phase_ == TRAIN) && (epoch<p_t))){
          this->forward_gpu_gemm(bottom[i]->count() / this->num_,bottom_data + n * this->bottom_dim_, this->blobs_[0]->count(), weight,
              top[i]->count() / this->num_, top_data + n * this->top_dim_, NULL,true);
      } else {
	  if ((epoch-p_t)% (Z+1)==0){
		this->forward_gpu_gemm(bottom[i]->count() / this->num_, bottom_data + n * this->bottom_dim_, this->blobs_[0]->count(), weight,
	              top[i]->count() / this->num_, top_data + n * this->top_dim_,NULL,true);
		if (count == 32768)
			 Threshold_pruning<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_data + n * this->top_dim_, pruning1[images/100]+n * this->top_dim_);
		else if (count==8192)
			 Threshold_pruning<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_data + n * this->top_dim_, pruning2[images/100]+n * this->top_dim_);
		else 
			 Threshold_pruning<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_data + n * this->top_dim_, pruning3[images/100]+n * this->top_dim_);
		CUDA_POST_KERNEL_CHECK;
	  } else {
		if (count == 32768)
			this->forward_gpu_gemm(bottom[i]->count() / this->num_, bottom_data + n * this->bottom_dim_, this->blobs_[0]->count(), weight, top[i]->count() / this->num_, top_data + n * this->top_dim_, pruning1[images/100]+n * this->top_dim_, false);
		else if (count==8192)
			this->forward_gpu_gemm(bottom[i]->count() / this->num_, bottom_data + n * this->bottom_dim_, this->blobs_[0]->count(), weight, top[i]->count() / this->num_, top_data + n * this->top_dim_, pruning2[images/100]+n * this->top_dim_, false);
		else 
			this->forward_gpu_gemm(bottom[i]->count() / this->num_, bottom_data + n * this->bottom_dim_, this->blobs_[0]->count(), weight, top[i]->count() / this->num_, top_data + n * this->top_dim_, pruning3[images/100]+n * this->top_dim_, false); 
	  }
      }
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
      }
    }
    if (this->phase_ == TRAIN){
	if(this->top_dim_ == 4096)
		images+=this->num_;
	if (images >=50000){
		epoch++;
		images=0;
	}
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
