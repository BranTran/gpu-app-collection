// Copyright (c) 2018-2021, Vijay Kandiah, Junrui Pan, Mahmoud Khairy, Scott Peverelle, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
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
//This code is a modification of L1 cache benchmark from 
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking": https://arxiv.org/pdf/1804.06826.pdf

//This benchmark stresses the L2 cache

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_PER_BLOCK 256
#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 640
#endif
#define WARP_SIZE 32

//V100 has 6144KB L2, and we are doing 8B entries

#define ARRAY_SIZE 4294967296 //32
#define STRIDE 67108864

uint64_t* dsink;
uint64_t* posArray_g;

// GPU error check
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}


__global__ void l2_stress(uint64_t *posArray, unsigned long long iterations){
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t current_index = tid*8;
    #pragma unroll 100
    for(unsigned long long i = 0; i < iterations; ++i) {
        uint64_t *ptr = posArray + current_index;

        asm volatile ("st.global.wt.u64 [%1], %0;"
                      :: "l" (ptr), "l" (ptr)
                      : "memory");

        current_index = (current_index + STRIDE) % ARRAY_SIZE;
    }
}

int main(int argc, char** argv){
  unsigned long long iterations;
  if (argc != 2){
    fprintf(stderr,"usage: %s #iterations #cores #ActiveThreadsperWarp\n",argv[0]);
    exit(1);
  }
  else {
    iterations = atoll(argv[1]);
  }
  long total_threads = ARRAY_SIZE; //THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 printf("Power Microbenchmarks with iterations %llu\n",iterations);

  dsink = (uint64_t*) malloc(total_threads*sizeof(uint64_t));


  

  checkCudaErrors( cudaMalloc(&posArray_g, total_threads*sizeof(uint64_t)) );
 cudaEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(cudaEventCreate(&start));  
 checkCudaErrors(cudaEventCreate(&stop));

 checkCudaErrors(cudaEventRecord(start));    
  l2_stress<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(posArray_g, iterations);
 checkCudaErrors(cudaEventRecord(stop));               
 
 checkCudaErrors(cudaEventSynchronize(stop));           
 checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));  
 printf("gpu execution time = %.3f ms\n", elapsedTime);  
  
  
  checkCudaErrors( cudaPeekAtLastError() );
  checkCudaErrors(cudaEventDestroy(start));
 checkCudaErrors(cudaEventDestroy(stop));

 return 0;
}

void CleanupResources(void)
{
  // Free device memory
  if (posArray_g)
  cudaFree(posArray_g);

  // Free host memory
  if (dsink)
  free(dsink);

}

