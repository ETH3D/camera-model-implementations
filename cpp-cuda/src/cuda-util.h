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

#pragma once

#include <iostream>

#define CUDA_CHECKED_CALL(cuda_call)                                           \
    do {                                                                       \
      cudaError error = (cuda_call);                                           \
      if (cudaSuccess != error) {                                              \
        std::cerr << "Cuda Error: " << cudaGetErrorString(error) << std::endl; \
      }                                                                        \
    } while(false)

#define CHECK_CUDA_NO_ERROR()                                                  \
  do {                                                                         \
    cudaError error = cudaGetLastError();                                      \
    if (cudaSuccess != error) {                                                \
      std::cerr << "Cuda Error: " << cudaGetErrorString(error) << std::endl;   \
    }                                                                          \
  } while(false)

// Returns the required number of CUDA blocks to cover a given domain size,
// given a specific block size.
inline int GetBlockCount(int domain_size, int block_size) {
  const int div = domain_size / block_size;
  return (domain_size % block_size == 0) ? div : (div + 1);
}
