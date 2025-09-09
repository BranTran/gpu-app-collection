/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/********************************************************************
 *      Modified by:
 * Vijay Kandiah, Northwestern University
 ********************************************************************/

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 1024
#define MATRIX_N 1024
#define MATRIX_K 1024



// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


__global__ void ldmatrix_example(int* c, unsigned long long iterations) {
    // Define a shared memory array large enough to hold multiple elements per thread.
    int block_id = blockDim.x;
    // This is a placeholder for the shared memory address.
    // In a real kernel, you would declare shared memory and use its address.
    extern __shared__ char smem[];
    double* smem_A = (double*)smem;

    unsigned long long addr = (unsigned long long)smem_A;
    int d0, d1, d2, d3;

    // This loop performs repeated loads from the same address.
    // The actual values loaded into the registers will be the same each time.
    #pragma unroll 100
    for (unsigned long long i = 0; i < iterations; i++) {
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "l"(addr)
        );
        c[block_id] = d0;
    }
}

int main(int argc, char* argv[]) {
   unsigned long long iterations;
    if(argc!=2) {
      fprintf(stderr,"usage: %s #iterations\n",argv[0]);
      exit(1);
    }
    else {
      iterations = atoll(argv[1]);
    }

   int *c;

   int *c_h;
   
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(int)));

   c_h = (int*)malloc(MATRIX_M * MATRIX_N * sizeof(int));

   cudaErrCheck(cudaMemcpy(c, c_h, MATRIX_M * MATRIX_N * sizeof(int), cudaMemcpyHostToDevice));



   printf("\nM = %d, N = %d, K = %d\n\n", MATRIX_M, MATRIX_N, MATRIX_K);
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;
   const size_t shmem_size = 256;
   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
   cudaErrCheck(cudaEventRecord(startWMMA));
   ldmatrix_example <<< gridDim, blockDim, shmem_size>>> (c, iterations);
   cudaErrCheck(cudaEventRecord(stopWMMA));

   
   cudaErrCheck(cudaMemcpy(c_h, c, MATRIX_M * MATRIX_N * sizeof(int), cudaMemcpyDeviceToHost));
 

   float wmmaTime;

   cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
   printf("gpu execution time = %.3f ms\n", wmmaTime);
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));
   cudaErrCheck(cudaFree(c));
   free(c_h);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}
