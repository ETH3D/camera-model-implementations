// Copyright 2017 Thomas Schöps
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "camera.h"

#include <limits>
#include <cstring>

#include "cuda-util.h"

ThinPrismFisheyeCamera::ThinPrismFisheyeCamera(
    int width, int height, float fx, float fy, float cx, float cy, float k1,
    float k2, float p1, float p2, float k3, float k4, float sx1, float sy1)
    : k_(make_float4(fx, fy, cx, cy)),
      k_inv_(make_float4(1.0 / fx, 1.0 / fy, -1.0 * cx / fx, -1.0 * cy / fy)),
      normalized_k_(make_float4(fx / width, fy / height,
                    (cx + 0.5f) / width, (cy + 0.5f) / height)),
      width_(width), height_(height),
      distortion_parameters_{k1, k2, p1, p2, k3, k4, sx1, sy1},
      undistortion_lookup_gpu_address_(0),
      undistortion_lookup_cpu_(0) {
  InitCutoff();
}

ThinPrismFisheyeCamera::ThinPrismFisheyeCamera(
    int width, int height, const float* parameters)
    : k_(make_float4(parameters[0], parameters[1], parameters[2],
                     parameters[3])),
      k_inv_(make_float4(1.0 / parameters[0],
                         1.0 / parameters[1],
                         -1.0 * parameters[2] / parameters[0],
                         -1.0 * parameters[3] / parameters[1])),
      normalized_k_(make_float4(parameters[0] / width, parameters[1] / height,
                    (parameters[2] + 0.5f) / width,
                    (parameters[3] + 0.5f) / height)),
      width_(width), height_(height),
      distortion_parameters_{parameters[4], parameters[5], parameters[6],
                             parameters[7], parameters[8], parameters[9],
                             parameters[10], parameters[11]},
      undistortion_lookup_gpu_address_(0),
      undistortion_lookup_cpu_(0) {
  InitCutoff();
}

ThinPrismFisheyeCamera::ThinPrismFisheyeCamera(
    const ThinPrismFisheyeCamera& other) {
  k_ = other.k_;
  k_inv_ = other.k_inv_;
  normalized_k_ = other.normalized_k_;
  width_ = other.width_;
  height_ = other.height_;
  radius_cutoff_ = other.radius_cutoff_;
  
  memcpy(distortion_parameters_, other.distortion_parameters_,
         8 * sizeof(float));
  
  // This is to ensure that copies created in calls to CUDA kernels do not
  // delete the undistortion lookup on destruction. Do not use this assignment
  // operator on other occasions.
  undistortion_lookup_gpu_address_ = 0;
  undistortion_lookup_gpu_texture_ = other.undistortion_lookup_gpu_texture_;
  undistortion_lookup_cpu_ = 0;
}

ThinPrismFisheyeCamera::~ThinPrismFisheyeCamera() {
  if (undistortion_lookup_gpu_address_ != 0) {
    cudaDestroyTextureObject(undistortion_lookup_gpu_texture_);
    CUDA_CHECKED_CALL(
        cudaFree(reinterpret_cast<void*>(undistortion_lookup_gpu_address_)));
  }
  delete[] undistortion_lookup_cpu_;
}

void ThinPrismFisheyeCamera::InitializeUnprojectionLookup(cudaStream_t stream) {
  // As camera settings are immutable, there is no need for re-computation once
  // an undistortion lookup has been computed.
  if (undistortion_lookup_cpu_) {
    return;
  }
  
  // Compute undistortion lookup texture on CPU.
  undistortion_lookup_cpu_ = new float2[height_ * width_];
  float2* ptr = undistortion_lookup_cpu_;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      *ptr = NormalizedDistortedToNormalized(
          make_float2(fx_inv() * x + cx_inv(), fy_inv() * y + cy_inv()));
      ++ptr;
    }
  }

  // Upload undistortion lookup texture to GPU.
  size_t undistortion_lookup_gpu_pitch_;
  CUDA_CHECKED_CALL(cudaMallocPitch(&undistortion_lookup_gpu_address_,
                                    &undistortion_lookup_gpu_pitch_,
                                    width_ * sizeof(float2), height_));
  // TODO(puzzlepaint): this is not ideal, as it does not use a stream.
  // Possibly a "ToCUDA()" method (analogous to CUDABuffer's) could solve this
  // without making a stream a camera constructor parameter. That method would
  // upload the table on demand only (which may also prevent useless uploads).
  // TODO(puzzlepaint): try whether it is faster to calculate the lookup
  // on the GPU and transfer it to the CPU.
  CUDA_CHECKED_CALL(cudaMemcpy2DAsync(
      undistortion_lookup_gpu_address_, undistortion_lookup_gpu_pitch_,
      static_cast<const void*>(undistortion_lookup_cpu_),
      width_ * sizeof(float2), width_ * sizeof(float2), height_,
      cudaMemcpyHostToDevice, stream));

  // Create CUDA texture object for GPU undistortion lookup.
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;

  resDesc.res.pitch2D.devPtr = undistortion_lookup_gpu_address_;
  resDesc.res.pitch2D.pitchInBytes = undistortion_lookup_gpu_pitch_;
  resDesc.res.pitch2D.width = width_;
  resDesc.res.pitch2D.height = height_;
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float2>();

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  CUDA_CHECKED_CALL(cudaCreateTextureObject(&undistortion_lookup_gpu_texture_,
                                            &resDesc, &texDesc, NULL));
}

ThinPrismFisheyeCamera* ThinPrismFisheyeCamera::ScaledBy(float factor) const {
  int scaled_width = static_cast<int>(factor * width_ + 0.5f);
  int scaled_height = static_cast<int>(factor * height_ + 0.5f);
  return new ThinPrismFisheyeCamera(
      scaled_width, scaled_height, factor * fx(),
      factor * fy(), factor * (cx() + 0.5f) - 0.5f,
      factor * (cy() + 0.5f) - 0.5f, distortion_parameters_[0],
      distortion_parameters_[1], distortion_parameters_[2],
      distortion_parameters_[3], distortion_parameters_[4],
      distortion_parameters_[5], distortion_parameters_[6],
      distortion_parameters_[7]);
}

ThinPrismFisheyeCamera* ThinPrismFisheyeCamera::ShiftedBy(
    float cx_offset, float cy_offset) const {
  return new ThinPrismFisheyeCamera(
      width_, height_, fx(), fy(), cx() + cx_offset,
      cy() + cy_offset, distortion_parameters_[0],
      distortion_parameters_[1], distortion_parameters_[2],
      distortion_parameters_[3], distortion_parameters_[4],
      distortion_parameters_[5], distortion_parameters_[6],
      distortion_parameters_[7]);
}

void ThinPrismFisheyeCamera::InitCutoff() {
  // Unproject some sample points at the image borders to find out where to
  // stop projecting points that are too far out. Those might otherwise get
  // projected into the image again at some point with certain distortion
  // parameter settings.
  
  // Disable cutoff while running this function such that the unprojection
  // works.
  radius_cutoff_ = std::numeric_limits<float>::infinity();
  float result = 0;

  for (int x = 0; x < width_; ++ x) {
    float2 nxy = NormalizedDistortedToNormalized(make_float2(fx_inv() * x + cx_inv(),
                                       fy_inv() * 0 + cy_inv()));
    float radius = sqrtf(nxy.x * nxy.x + nxy.y * nxy.y);
    if (1.01f * radius > result) {
      result = 1.01f * radius;
    }
    
    nxy = NormalizedDistortedToNormalized(make_float2(fx_inv() * x + cx_inv(),
                                fy_inv() * (height_ - 1) + cy_inv()));
    radius = sqrtf(nxy.x * nxy.x + nxy.y * nxy.y);
    if (1.01f * radius > result) {
      result = 1.01f * radius;
    }
  }
  for (int y = 1; y < height_ - 1; ++ y) {
    float2 nxy = NormalizedDistortedToNormalized(make_float2(fx_inv() * 0 + cx_inv(),
                                       fy_inv() * y + cy_inv()));
    float radius = sqrtf(nxy.x * nxy.x + nxy.y * nxy.y);
    if (1.01f * radius > result) {
      result = 1.01f * radius;
    }
    
    nxy = NormalizedDistortedToNormalized(make_float2(fx_inv() * (width_ - 1) + cx_inv(),
                                fy_inv() * y + cy_inv()));
    radius = sqrtf(nxy.x * nxy.x + nxy.y * nxy.y);
    if (1.01f * radius > result) {
      result = 1.01f * radius;
    }
  }
  
  radius_cutoff_= result;
  
  // Try to avoid problems with fisheye cameras where the image corners can
  // cause too large cutoff values by imposing a maximum.
  constexpr float kCutoffMax = 7.f;
  radius_cutoff_ = std::min(kCutoffMax, radius_cutoff_);
}
