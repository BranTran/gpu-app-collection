
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

//This benchmark stresses the L1 cache

//This code have been tested on Volta V100 architecture

#include <stdio.h>   
#include <stdlib.h> 
#include <cuda.h>

#define THREADS_PER_BLOCK 256
#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 640
#endif
#define WARP_SIZE 32

#define ARRAY_SIZE 67108864
#define STRIDE 1048576

// GPU error check
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}

__global__ void pointers_init(uint4 *posArray){
    // This kernel is launched with a single thread (1 block, 1 thread)
    // to initialize the entire array sequentially.
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid == 0){ // Ensure only one thread performs the initialization
        for (uint64_t i = 0; i < ARRAY_SIZE; i++){
            // Calculate the offset for the next pointer in the circular array
            uint64_t offset = (i + STRIDE) % ARRAY_SIZE;
            
            // Calculate the actual 64-bit address of the next element
            // This is a pointer to an uint4 element
            uint64_t next_ptr_value = (uint64_t)(posArray + offset);
            
            // Split the 64-bit pointer into two 32-bit parts
            uint32_t low_32_bits = (uint32_t)next_ptr_value;
            uint32_t high_32_bits = (uint32_t)(next_ptr_value >> 32);

            // Store the 32-bit parts into the 'x' and 'y' components of the uint4
            // The 'z' and 'w' components are unused for this pointer.
            posArray[i].x = low_32_bits;
            posArray[i].y = high_32_bits;
            posArray[i].z = 0; 
            posArray[i].w = 0;
        }
    }
}
__global__ void dram_stress(uint4 *posArray, uint4 *dsink, unsigned long long iterations){
    // Calculate the global thread index
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize the current pointer for this thread.
    // Each thread starts at a different uint4 element in the posArray.
    uint4 *current_ptr = posArray + tid;

    // Variable to hold the loaded 128-bit value (uint4)
    uint4 loaded_val;

    // Pointer-chasing iterations
    // The #pragma unroll directive encourages the compiler to unroll the loop,
    // which can help in observing consistent cache behavior by reducing loop overhead.
    #pragma unroll 100 // Unroll factor can be adjusted or removed based on performance
    for(unsigned long long i = 0; i < iterations; ++i) {
        // Perform a 128-bit coherent global load operation.
        // 'ld.global.cv.v4.u32' is the SASS instruction for a 128-bit coherent global load
        // into four 32-bit registers (vector load of 4 unsigned 32-bit integers).
        // '.cv' (Coherent Volatile) ensures the load goes through the cache hierarchy
        // and prevents compiler reordering.
        // {%0, %1, %2, %3} are output operands for loaded_val.x, .y, .z, .w respectively.
        // [%4] is the input operand for the base address (current_ptr).
        asm volatile ("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
                      : "=r" (loaded_val.x),         // Output: loaded_val.x
                        "=r" (loaded_val.y),         // Output: loaded_val.y
                        "=r" (loaded_val.z),         // Output: loaded_val.z
                        "=r" (loaded_val.w)          // Output: loaded_val.w
                      : "l" (current_ptr)            // Input: current_ptr (long long register, holding the base address)
                      : "memory");                   // Clobbers: memory (informs compiler about memory side effects)

        current_ptr = (uint4*)(((uint64_t)loaded_val.y << 32) | loaded_val.x);
    }
    // Write the final pointer value for this thread back to device memory
    dsink[tid].x = (unsigned)current_ptr;
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

  //uint64_t *dsink = (uint64_t*) malloc(total_threads*sizeof(uint64_t));
      // Use pinned (page-locked) memory for `dsink`
    uint4 *dsink;
    checkCudaErrors(cudaMallocHost((void**)&dsink, total_threads * sizeof(uint4)));


  uint4 *posArray_g;
  uint4 *dsink_g;
  

  checkCudaErrors( cudaMalloc(&posArray_g, total_threads*sizeof(uint4)) );
  checkCudaErrors( cudaMalloc(&dsink_g, total_threads*sizeof(uint4)) );
 cudaEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(cudaEventCreate(&start));  
 checkCudaErrors(cudaEventCreate(&stop));

    pointers_init<<<1,1>>>(posArray_g);
 checkCudaErrors(cudaEventRecord(start));    
  dram_stress<<<NUM_OF_BLOCKS,THREADS_PER_BLOCK>>>(posArray_g, dsink_g, iterations);
 checkCudaErrors(cudaEventRecord(stop));               
 
 checkCudaErrors(cudaEventSynchronize(stop));           
 checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));  
 printf("gpu execution time = %.3f ms\n", elapsedTime);  
  
  
  checkCudaErrors( cudaPeekAtLastError() );

  checkCudaErrors( cudaMemcpy(dsink, dsink_g, total_threads*sizeof(uint4), cudaMemcpyDeviceToHost) );

  return 0;
} 
