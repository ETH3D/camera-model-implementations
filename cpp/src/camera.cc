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

ThinPrismFisheyeCamera::ThinPrismFisheyeCamera(
    int width, int height, float fx, float fy, float cx, float cy, float k1,
    float k2, float p1, float p2, float k3, float k4, float sx1, float sy1)
    : k_(Vec4f(fx, fy, cx, cy)),
      k_inv_(Vec4f(1.0 / fx, 1.0 / fy, -1.0 * cx / fx, -1.0 * cy / fy)),
      normalized_k_(Vec4f(fx / width, fy / height,
                    (cx + 0.5f) / width, (cy + 0.5f) / height)),
      width_(width), height_(height),
      distortion_parameters_{k1, k2, p1, p2, k3, k4, sx1, sy1},
      undistortion_lookup_(0) {
  InitCutoff();
}

ThinPrismFisheyeCamera::ThinPrismFisheyeCamera(
    int width, int height, const float* parameters)
    : k_(Vec4f(parameters[0], parameters[1], parameters[2],
                     parameters[3])),
      k_inv_(Vec4f(1.0 / parameters[0],
                         1.0 / parameters[1],
                         -1.0 * parameters[2] / parameters[0],
                         -1.0 * parameters[3] / parameters[1])),
      normalized_k_(Vec4f(parameters[0] / width, parameters[1] / height,
                    (parameters[2] + 0.5f) / width,
                    (parameters[3] + 0.5f) / height)),
      width_(width), height_(height),
      distortion_parameters_{parameters[4], parameters[5], parameters[6],
                             parameters[7], parameters[8], parameters[9],
                             parameters[10], parameters[11]},
      undistortion_lookup_(0) {
  InitCutoff();
}

ThinPrismFisheyeCamera::~ThinPrismFisheyeCamera() {
  delete[] undistortion_lookup_;
}

void ThinPrismFisheyeCamera::InitializeUnprojectionLookup() {
  // As camera settings are immutable, there is no need for re-computation once
  // an undistortion lookup has been computed.
  if (undistortion_lookup_) {
    return;
  }
  
  // Compute undistortion lookup texture on CPU.
  undistortion_lookup_ = new Vec2f[height_ * width_];
  Vec2f* ptr = undistortion_lookup_;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      *ptr = NormalizedDistortedToNormalized(
          Vec2f(fx_inv() * x + cx_inv(), fy_inv() * y + cy_inv()));
      ++ptr;
    }
  }
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
    Vec2f nxy = NormalizedDistortedToNormalized(Vec2f(fx_inv() * x + cx_inv(),
                                       fy_inv() * 0 + cy_inv()));
    float radius = sqrtf(nxy.x * nxy.x + nxy.y * nxy.y);
    if (1.01f * radius > result) {
      result = 1.01f * radius;
    }
    
    nxy = NormalizedDistortedToNormalized(Vec2f(fx_inv() * x + cx_inv(),
                                fy_inv() * (height_ - 1) + cy_inv()));
    radius = sqrtf(nxy.x * nxy.x + nxy.y * nxy.y);
    if (1.01f * radius > result) {
      result = 1.01f * radius;
    }
  }
  for (int y = 1; y < height_ - 1; ++ y) {
    Vec2f nxy = NormalizedDistortedToNormalized(Vec2f(fx_inv() * 0 + cx_inv(),
                                       fy_inv() * y + cy_inv()));
    float radius = sqrtf(nxy.x * nxy.x + nxy.y * nxy.y);
    if (1.01f * radius > result) {
      result = 1.01f * radius;
    }
    
    nxy = NormalizedDistortedToNormalized(Vec2f(fx_inv() * (width_ - 1) + cx_inv(),
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
