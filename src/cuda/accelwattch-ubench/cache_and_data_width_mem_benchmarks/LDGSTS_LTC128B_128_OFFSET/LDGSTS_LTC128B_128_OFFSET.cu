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
#include <stdio.h>
#include <stdlib.h>
//#include <cutil.h>
//#include <mgp.h>
// Includes
//#include <stdio.h>
//#include "../include/ContAcq-IntClk.h"

// includes, project
//#include "../include/sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
//#include <shrQATest.h>
//#include <shrUtils.h>

// includes CUDA
#include <cuda_runtime.h>
#include <cuda.h> //BT: Needed for uint32_t
#define THREADS_PER_BLOCK 256
#ifndef NUM_OF_BLOCKS
#define NUM_OF_BLOCKS 640
#endif
#define SHARED_MEM_SIZE THREADS_PER_BLOCK*4
#define BATCH_SIZE 8
#define BYTES_PER_COPY 16
#define COPIES_IN_GROUP (THREADS_PER_BLOCK * BATCH_SIZE)
#define BLOCK_TOTAL_LOAD_BYTES (THREADS_PER_BLOCK * BATCH_SIZE * BYTES_PER_COPY)
// Define the total number of elements in the circular buffer (e.g., 1024 doubles)
#define CIRCULAR_BUFFER_ELEMENTS 1024 
// Total bytes in the circular buffer
#define CIRCULAR_BUFFER_BYTES (CIRCULAR_BUFFER_ELEMENTS * sizeof(uint64_t)) 
// Variables
uint64_t* h_A;
uint64_t* h_B;
uint64_t* d_A;
uint64_t* d_B;
//bool noprompt = false;
//unsigned int my_timer;

// Functions
void CleanupResources(void);
void RandomInit(uint64_t*, int);
//void ParseArguments(int, char**);

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
  if(cudaSuccess != err){
  fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
   exit(-1);
  }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err){
  fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
  exit(-1);
  }
}

// end of CUDA Helper Functions

__global__ void async_load_kernel_circular(
    const uint64_t* global_input, 
    uint64_t* global_output, 
    unsigned long long iterations) 
{
    extern __shared__ char smem[];
    
    const unsigned int tid = threadIdx.x;

    // --- OUTER BATCH LOOP ---
    #pragma unroll 10
    for (unsigned long long i = 0; i < iterations; ++i) {
        
        // 1. Calculate the base offset for the entire block's load (with wrap-around).
        unsigned long long current_block_start_byte = 
            (i * BLOCK_TOTAL_LOAD_BYTES) % CIRCULAR_BUFFER_BYTES;
            
        // 2. Calculate the base pointer for the current block's data in global memory.
        const char* current_global_base_ptr = 
            (const char*)global_input + current_block_start_byte;

        // 3. Calculate the starting offset for THIS THREAD's personal batch.
        // This ensures the thread starts reading after the previous threads' data.
        const unsigned int thread_base_offset_in_block = 
            tid * BATCH_SIZE * BYTES_PER_COPY;

        // --- ASYNCHRONOUS COPY (Inner Per-Thread Batch Loop) ---
        for (int k = 0; k < BATCH_SIZE; ++k) {
            
            // The total offset for the k-th copy in this thread's batch:
            // This is the thread's starting point + k * 16B
            unsigned int total_offset_in_block = 
                thread_base_offset_in_block + k * BYTES_PER_COPY;

            // Calculate the specific GMem and SMem addresses for the current 16B chunk:
            
            // SMem address: Contiguous block of shared memory reserved for this thread's data
            char* smem_ptr = smem + total_offset_in_block;
            
            // GMem address: The global base + the total offset within the block's current load
            const char* gmem_ptr = current_global_base_ptr + total_offset_in_block;
            
            // Issue the asynchronous copy (Each thread issues BATCH_SIZE copies)
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, 16;"
                : // No outputs
                : "l"(smem_ptr), "l"(gmem_ptr)
            );
        }
        
        // --- SYNCHRONIZE AND WAIT (ALL THREADS PARTICIPATE) ---

        // All threads commit the group.
        asm volatile("cp.async.commit_group;");
        
        // ALL threads wait for the combined (BlockSize * BATCH_SIZE) copies to complete.
        asm volatile("cp.async.wait_group %0;" :: "n"(COPIES_IN_GROUP));
        
        // All threads must synchronize locally before accessing the shared memory data.
        __syncthreads();
        
        // --- COMPUTATION/OUTPUT ---
        // At this point, smem[] holds the combined data. The tid-th thread's data 
        // starts at smem[thread_base_offset_in_block / sizeof(double)].
        
        // Example: Have each thread process its first loaded element
        unsigned int data_start_index = thread_base_offset_in_block / sizeof(uint64_t);

        // Simple computation to ensure the compiler doesn't optimize the load away
        global_output[data_start_index] = ((const uint64_t*)smem)[data_start_index] + (uint64_t)i;
    }
}


int main(int argc, char** argv)
{
 unsigned long long iterations;
 if(argc!=2) {
   fprintf(stderr,"usage: %s #iterations\n",argv[0]);
   exit(1);
 }
 else {
   iterations = atoll(argv[1]);
 }
 
 printf("Power Microbenchmarks with iterations %lld\n",iterations);
 
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;

 size_t size = N * sizeof(uint64_t);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (uint64_t*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_B = (uint64_t*)malloc(size);
 if (h_B == 0) CleanupResources();


 // Initialize input vectors
 RandomInit(h_A, N);


 // Allocate vectors in device memory
 checkCudaErrors( cudaMalloc((void**)&d_A, size) );
 checkCudaErrors( cudaMalloc((void**)&d_B, size) );


 // Copy vector from host memory to device memory
 checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );


 cudaEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(cudaEventCreate(&start));  
 checkCudaErrors(cudaEventCreate(&stop));

 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);


 checkCudaErrors(cudaEventRecord(start));              
 async_load_kernel_circular<<<dimGrid,dimBlock,SHARED_MEM_SIZE>>>(d_A, d_B,iterations);  
 checkCudaErrors(cudaEventRecord(stop));               
 
 checkCudaErrors(cudaEventSynchronize(stop));           
 checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));  
 printf("gpu execution time = %.3f ms\n", elapsedTime);  
 getLastCudaError("kernel launch failure");              

 // Copy result from device memory to host memory
 // h_B contains the result in host memory
 checkCudaErrors( cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost) );
  checkCudaErrors(cudaEventDestroy(start));
 checkCudaErrors(cudaEventDestroy(stop));
 CleanupResources();

 return 0;
}

void CleanupResources(void)
{
  // Free device memory
  if (d_A)
  cudaFree(d_A);
  if (d_B)
  cudaFree(d_B);

  // Free host memory
  if (h_A)
  free(h_A);
  if (h_B)
  free(h_B);

}

// Allocates an array with random float entries.
void RandomInit(uint64_t* data, int n)
{
  for (int i = 0; i < n; ++i){
  srand((uint64_t)time(0));  
  data[i] = rand() / RAND_MAX;
  }
}

