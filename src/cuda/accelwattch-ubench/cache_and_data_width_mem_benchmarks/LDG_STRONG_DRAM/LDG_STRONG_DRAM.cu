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

//This benchmark stresses DRAM

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
//#define FACTOR 2
//#define ARRAY_SIZE (67108864 * FACTOR) // 2^26 
//V100 has 6144KB which would be 16384 8B entries
//#define STRIDE (1048576 * FACTOR) // 2^20

#define ARRAY_SIZE 134217728
#define STRIDE 2097152
uint32_t* dsink;
uint32_t* posArray_g;

// GPU error check
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}


__global__ void dram_stress(uint32_t *posArray, unsigned long long iterations){
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t current_index = tid*2;
    uint32_t data;
    #pragma unroll 100
    for(unsigned long long i = 0; i < iterations; ++i) {
        uint32_t *ptr = posArray + current_index;
        asm volatile ("ld.global.cv.u32 %0, [%1];"
                      :"=r" (data)
		      : "l" (ptr)
                      : "memory");

        asm volatile ("st.global.cg.u32 [%1], %0;"
                      :: "r" (data), "l" (ptr)
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
  int total_threads = ARRAY_SIZE; //THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 printf("Power Microbenchmarks with iterations %llu\n",iterations);

  dsink = (uint32_t*) malloc(total_threads*sizeof(uint32_t));


  

  checkCudaErrors( cudaMalloc(&posArray_g, total_threads*sizeof(uint32_t)) );
 cudaEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(cudaEventCreate(&start));  
 checkCudaErrors(cudaEventCreate(&stop));

 checkCudaErrors(cudaEventRecord(start));    
  dram_stress<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(posArray_g, iterations);
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

