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
#define NUM_OF_BLOCKS 640
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

__global__ void pointers_init(uint64_t *posArray){

  uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid == 0){
      for (uint64_t i=0; i<ARRAY_SIZE; i++){
	uint64_t offset = (i + STRIDE) % ARRAY_SIZE;
        posArray[i] = (uint64_t)(posArray + offset);
      }
  }
}

__global__ void dram_stress(uint64_t *posArray, uint64_t *dsink, unsigned long long iterations){

  // thread index
  uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x;

    uint64_t *current_ptr = posArray + tid;

    // Variables to hold the two 32-bit loaded values
    uint32_t loaded_val_low;
    uint32_t loaded_val_high;

    // Pointer-chasing iterations times
    // The #pragma unroll directive encourages the compiler to unroll the loop,
    // which can help in observing consistent cache behavior by reducing loop overhead.
    #pragma unroll 100
    for(unsigned long long i = 0; i < iterations; ++i) {
        // Cast the current 64-bit pointer to a 32-bit pointer type for assembly loads.
        // This allows us to specify byte offsets for 32-bit accesses.
        uint32_t *addr_32_ptr = (uint32_t*)current_ptr;

        // Perform two coherent global load operations for 32-bit values.
        // 'ld.global.cg.u32' is the SASS instruction for a 32-bit coherent global load.
        // '.cg' (Coherent Global) ensures the load goes through the cache hierarchy.
        // '%0' and '%1' are output operands for loaded_val_low and loaded_val_high.
        // '%2' is the input operand for the base address (addr_32_ptr).
        // The first load is from the base address, the second from base address + 4 bytes.
        asm volatile ("ld.global.cv.u32 %0, [%2];\n\t"  // Load lower 32 bits from current_ptr
                      "ld.global.cv.u32 %1, [%2 + 4];" // Load upper 32 bits from current_ptr + 4 bytes
                      : "=r" (loaded_val_low),         // Output: loaded_val_low (general-purpose register)
                        "=r" (loaded_val_high)         // Output: loaded_val_high (general-purpose register)
                      : "l" (addr_32_ptr)              // Input: addr_32_ptr (long long register, holding the the base address for loads)
                      : "memory");                     // Clobbers: memory (informs compiler about memory side effects)

        // Stitch the two 32-bit values back together to form the complete 64-bit value.
        // This 64-bit value is the *next memory address* (pointer) to jump to.
        current_ptr = (uint64_t*)(((uint64_t)loaded_val_high << 32) | loaded_val_low);
    }
  	// write data back to memory
  	dsink[tid] = (uint64_t)current_ptr;
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
    uint64_t *dsink;
    checkCudaErrors(cudaMallocHost((void**)&dsink, total_threads * sizeof(uint64_t)));


  uint64_t *posArray_g;
  uint64_t *dsink_g;
  

  checkCudaErrors( cudaMalloc(&posArray_g, total_threads*sizeof(uint64_t)) );
  checkCudaErrors( cudaMalloc(&dsink_g, total_threads*sizeof(uint64_t)) );
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

  checkCudaErrors( cudaMemcpy(dsink, dsink_g, total_threads*sizeof(uint64_t), cudaMemcpyDeviceToHost) );

  return 0;
} 
