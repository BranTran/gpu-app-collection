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

#define THREADS_PER_BLOCK 256
#define NUM_OF_BLOCKS 640
//#define ITERATIONS 40

// Variables
unsigned* h_A;
unsigned* h_B;
unsigned* h_C;
unsigned* d_A;
unsigned* d_B;
unsigned* d_C;
//bool noprompt = false;
//unsigned int my_timer;

// Functions
void CleanupResources(void);
void RandomInit(unsigned*, int);
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





__global__ void PowerKernal2(const unsigned* A, const unsigned* B, unsigned* C, unsigned long long N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned I1 = A[i];
    unsigned I2 = B[i];
    unsigned Value1, Value2, Value3, Value4, Value5, Value6;
    unsigned Value7, Value8, Value9, Value10, Value11, Value12;

    #pragma unroll 100
    for (unsigned long long k = 0; k < N; k++) {
        // BLOCK-0 (For instruction size of 16 bytes for Volta)
        __asm volatile (
            "\n\tmov.u32 %0, %12;"   // Value1 = I1
            "\n\tmov.u32 %1, %13;"   // Value2 = I2
            "\n\tmov.u32 %2, %1;"    // Value3 = Value2
            "\n\tmov.u32 %1, %0;"    // Value2 = Value1
            "\n\tmov.u32 %0, %2;"    // Value1 = Value3
            "\n\tmov.u32 %3, %2;"    // Value4 = Value3
            "\n\tmov.u32 %4, %3;"    // Value5 = Value4
            "\n\tmov.u32 %5, %4;"    // Value6 = Value5
            "\n\tmov.u32 %6, %5;"    // Value7 = Value6
            "\n\tmov.u32 %7, %6;"    // Value8 = Value7
            "\n\tmov.u32 %8, %7;"    // Value9 = Value8
            "\n\tmov.u32 %9, %8;"    // Value10 = Value9
            "\n\tmov.u32 %10, %9;"   // Value11 = Value10
            "\n\tmov.u32 %11, %10;"  // Value12 = Value11
            "\n\tmov.u32 %0, %11;"   // Value1 = Value12
            "\n\tmov.u32 %1, %0;"    // Value2 = Value1
            "\n\tmov.u32 %2, %1;"    // Value3 = Value2
            "\n\tmov.u32 %3, %2;"    // Value4 = Value3
            "\n\tmov.u32 %4, %3;"    // Value5 = Value4
            "\n\tmov.u32 %5, %4;"    // Value6 = Value5
            "\n\tmov.u32 %6, %5;"    // Value7 = Value6
            "\n\tmov.u32 %7, %6;"    // Value8 = Value7
            "\n\tmov.u32 %8, %7;"    // Value9 = Value8
            "\n\tmov.u32 %9, %8;"    // Value10 = Value9
            "\n\tmov.u32 %10, %9;"   // Value11 = Value10
            "\n\tmov.u32 %11, %10;"  // Value12 = Value11
            : "+r"(Value1), "+r"(Value2), "+r"(Value3), "+r"(Value4), "+r"(Value5), "+r"(Value6), "+r"(Value7), "+r"(Value8), "+r"(Value9), "+r"(Value10), "+r"(Value11), "+r"(Value12)
            : "r"(I1), "r"(I2)
        );
    }
    C[i] = Value12;
    __syncthreads();

}

__global__ void PowerKernalSample(unsigned* C, unsigned long long N)
{
  // FROM https://github.com/zchee/cuda-sample/blob/master/0_Simple/inlinePTX_nvrtc/inlinePTX_kernel.cu
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int laneid;

    //This command gets the lane ID within the current warp
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    __syncthreads();
    C[i] = laneid;
}


int main(int argc, char** argv)
{
 unsigned long long iterations;
 if(argc!=2) {
   fprintf(stderr,"usage: %s #iterations\n",argv[0]);
 }
 else {
   iterations = atoll(argv[1]);
 }
 
 printf("Power Microbenchmarks with iterations %lld\n",iterations);
 int N = THREADS_PER_BLOCK*NUM_OF_BLOCKS;
 size_t size = N * sizeof(unsigned);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (unsigned*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_B = (unsigned*)malloc(size);
 if (h_B == 0) CleanupResources();
 h_C = (unsigned*)malloc(size);
 if (h_C == 0) CleanupResources();

 // Initialize input vectors
 RandomInit(h_A, N);
 RandomInit(h_B, N);

 // Allocate vectors in device memory
 checkCudaErrors( cudaMalloc((void**)&d_A, size) );
 checkCudaErrors( cudaMalloc((void**)&d_B, size) );
 checkCudaErrors( cudaMalloc((void**)&d_C, size) );

 // Copy vectors from host memory to device memory
 checkCudaErrors( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
 checkCudaErrors( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

 cudaEvent_t start, stop;                   
 float elapsedTime = 0;                     
 checkCudaErrors(cudaEventCreate(&start));  
 checkCudaErrors(cudaEventCreate(&stop));

 //VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
 dim3 dimGrid(NUM_OF_BLOCKS,1);
 dim3 dimBlock(THREADS_PER_BLOCK,1);

 checkCudaErrors(cudaEventRecord(start));              
//  PowerKernal2<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, iterations);  
 PowerKernalSample<<<dimGrid,dimBlock>>>(d_C, iterations);  
 checkCudaErrors(cudaEventRecord(stop));               
 
 checkCudaErrors(cudaEventSynchronize(stop));           
 checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));  
 printf("gpu execution time = %.3f ms\n", elapsedTime);  
 getLastCudaError("kernel launch failure");              


 // Copy result from device memory to host memory
 // h_C contains the result in host memory
 checkCudaErrors( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
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
  if (d_C)
	cudaFree(d_C);

  // Free host memory
  if (h_A)
	free(h_A);
  if (h_B)
	free(h_B);
  if (h_C)
	free(h_C);

}

// Allocates an array with random float entries.
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
	srand((unsigned)time(0));  
	data[i] = rand() / RAND_MAX;
  }
}






