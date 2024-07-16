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
#define THREADS_PER_BLOCK 1024
#define NUM_OF_BLOCKS 80
#define SHARED_MEM_SIZE THREADS_PER_BLOCK*4
// Variables
unsigned* h_A;
unsigned* h_B;
unsigned* d_A;
unsigned* d_B;
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



__global__ void PowerKernal2( unsigned* A, unsigned* B, unsigned long long N)
{
    uint32_t tid = threadIdx.x;
    uint32_t uid = blockDim.x * blockIdx.x + tid;
//Defining shared memory space for each block
    __shared__  volatile uint64_t s[SHARED_MEM_SIZE];

  //Threads initialize pointer chasing
  //I want each thread to pointer chase without contention
  //So next address should be the next thread
  unsigned stride = THREADS_PER_BLOCK;
  uint64_t s_address = (uint64_t) s;
  if(tid == 0){
	for (unsigned i=tid; i<(SHARED_MEM_SIZE); i++){
        s[i] = (uint64_t)(s_address + (((i+stride)%SHARED_MEM_SIZE)*sizeof(uint64_t)));
    }
}
  __syncthreads();// Probably unneeded


/*//Trying pointer chasing like shared_lat, requires shf.l for byte addressing
  uint64_t p_chaser = uid;

  #pragma unroll 100
  for(unsigned long long k=0; k<N;k++) {
    p_chaser = s[p_chaser];
  }

  //sink back to global
  B[uid] = (unsigned)p_chaser;*/

// a register to avoid compiler optimization
	volatile uint64_t *ptr = s + tid;
	volatile uint64_t ptr1, ptr0;

	// initialize the thread pointer with the start address of the array
	// use ca modifier to cache the in L1
	asm volatile ("{\t\n"
		"ld.volatile.shared.u64 %0, [%1];\n\t"
		"}" : "=l"(ptr1) : "l"(ptr) : "memory"
	);

	// synchronize all threads
	asm volatile ("bar.sync 0;");


	// pointer-chasing ITERS times
	// use ca modifier to cache the load in L1
#pragma unroll 100
	for(uint64_t i=0; i<N; ++i) {	
		asm volatile ("{\t\n"
			"ld.volatile.shared.u64 %0, [%1];\n\t"
			"}" : "=l"(ptr0) : "l"((uint64_t*)ptr1) : "memory"
		);
		ptr1 = ptr0;    //swap the register for the next load

	}

	// write time and data back to memory
	B[uid] = ptr1;
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

 size_t size = N * sizeof(unsigned);
 // Allocate input vectors h_A and h_B in host memory
 h_A = (unsigned*)malloc(size);
 if (h_A == 0) CleanupResources();
 h_B = (unsigned*)malloc(size);
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
 PowerKernal2<<<dimGrid,dimBlock>>>(d_A, d_B,iterations);  
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
void RandomInit(unsigned* data, int n)
{
  for (int i = 0; i < n; ++i){
  srand((unsigned)time(0));  
  data[i] = rand() / RAND_MAX;
  }
}