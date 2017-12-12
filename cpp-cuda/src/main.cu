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

#include "main.cuh"

#include "camera.h"
#include "cuda-util.h"

template<class Camera>
__global__ void UnprojectAndProjectTestCUDAKernel(Camera camera) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < camera.width() && y < camera.height()) {
    float2 nxy = camera.PixelToNormalized(x, y);
    float2 pxy = camera.NormalizedToPixel(nxy);
    
    constexpr float kThreshold = 1e-3f;
    if (fabs(x - pxy.x) > kThreshold ||
        fabs(y - pxy.y) > kThreshold) {
      printf("UnprojectAndProjectTestCUDAKernel() failed at pixel (%i, %i)"
             " with result (%f, %f) (normalized test)\n", x, y, pxy.x, pxy.y);
    }
    
    float3 point = camera.PixelToPoint(make_float2(x, y), 3.f);
    float2 pxy2;
    if (!camera.PointToPixel(point, &pxy2)) {
      printf("UnprojectAndProjectTestCUDAKernel() failed at pixel (%i, %i)"
             ": unprojected point does not backproject into image\n");
    } else {
      if (fabs(x - pxy2.x) > kThreshold ||
          fabs(y - pxy2.y) > kThreshold) {
        printf("UnprojectAndProjectTestCUDAKernel() failed at pixel (%i, %i)"
              " with result (%f, %f) (point test)\n", x, y, pxy2.x, pxy2.y);
      }
    }
  }
}

template<class Camera>
void UnprojectAndProjectTestCUDA(const Camera& camera, cudaStream_t stream) {
  dim3 block_dim(32, 32);
  dim3 grid_dim(GetBlockCount(camera.width(), block_dim.x),
                GetBlockCount(camera.height(), block_dim.y));
  
  UnprojectAndProjectTestCUDAKernel<<<grid_dim, block_dim, 0, stream>>>(camera);
  CHECK_CUDA_NO_ERROR();
}

// Instantiate the template for all camera models (here, only for
// ThinPrismFisheyeCamera).
template void UnprojectAndProjectTestCUDA<ThinPrismFisheyeCamera>(
    const ThinPrismFisheyeCamera& camera, cudaStream_t stream);
