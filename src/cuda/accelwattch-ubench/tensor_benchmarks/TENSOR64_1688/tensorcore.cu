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
#define MATRIX_M 4096
#define MATRIX_N 4096
#define MATRIX_K 4096

#define WMMA_M 16
#define WMMA_N 8
#define WMMA_K 8
#define WARP_SIZE 32

__global__ void dmma1688_example(double *a, double *b, double *c, int M, int N, int K, unsigned long long iterations) {
    // Each warp processes a 16x8x16 tile.
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    
    // Allocate registers for the fragments for this thread.
    // D and C are 16x8 matrices, each thread gets (16*8)/32 = 4 elements.
    double c_frag[4];
    // A is 16x8 matrix, each thread gets (16*8)/32 = 4 elements.
    double a_frag[4];
    // B is 8x8 matrix, each thread gets (8*8)/32 = 2 elements.
    double b_frag[2];

    // Initialize fragments (dummy loads for a microbenchmark)
    // In a real application, you would use ldmatrix from shared memory
    for (int i = 0; i < 4; ++i) {
        a_frag[i] = a[lane_id * 4 + i];
        c_frag[i] = c[lane_id * 4 + i];
    }
    for (int i = 0; i < 2; ++i) {
        b_frag[i] = b[lane_id * 2 + i];
    }
    
    // The mma instruction requires the operands to be passed as individual
    // registers. This is where the original code had the mistake.
    // The PTX expects the registers directly, not pointers to arrays.
    
    #pragma unroll 100
    for (unsigned long long i = 0; i < iterations; i++){
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 "
            "  {%0, %1, %2, %3}, " // D - result
            "  {%4, %5, %6, %7}, " // A - input
            "  {%8, %9}, "         // B - input
            "  {%0, %1, %2, %3};" // C - accumulator
            : "+d"(c_frag[0]), "+d"(c_frag[1]), "+d"(c_frag[2]), "+d"(c_frag[3])
            , "+d"(c_frag[4]), "+d"(c_frag[5]), "+d"(c_frag[6]), "+d"(c_frag[7])
            : "d"(a_frag[0]), "d"(a_frag[1]), "d"(a_frag[2]), "d"(a_frag[3])
            , "d"(b_frag[0]), "d"(b_frag[1])
        );
    }

    // Store the result back to global memory (again, simplified)
    for (int i = 0; i < 4; ++i) {
        c[lane_id * 4 + i] = c_frag[i];
    }
}




void RandomInit_fp(double* data, int n)
{
   for (int i = 0; i < n; ++i){
   data[i] = (double) rand() / RAND_MAX;
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
   double *a_fp64;
   double *b_fp64;
//   half *a_fp16;
//   half *b_fp16;

   double *c_wmma;

   double *c_host_wmma;
   
   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   
   cudaErrCheck(cudaEventCreate(&startWMMA));
   cudaErrCheck(cudaEventCreate(&stopWMMA));
   
   cudaErrCheck(cudaMalloc((void**)&a_fp64, MATRIX_M * MATRIX_K * sizeof(double)));
   cudaErrCheck(cudaMalloc((void**)&b_fp64, MATRIX_K * MATRIX_N * sizeof(double)));
//   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
//   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(double)));

   c_host_wmma = (double*)malloc(MATRIX_M * MATRIX_N * sizeof(double));

   double *a_fp64_h = (double*) malloc(MATRIX_M * MATRIX_K*sizeof(double));
   double *b_fp64_h = (double*) malloc(MATRIX_K * MATRIX_N*sizeof(double));
   RandomInit_fp(a_fp64_h, MATRIX_M * MATRIX_K);
   RandomInit_fp(b_fp64_h, MATRIX_K * MATRIX_N);
   RandomInit_fp( c_host_wmma, MATRIX_M * MATRIX_N);
   cudaErrCheck(cudaMemcpy(c_wmma, c_host_wmma, MATRIX_M * MATRIX_N * sizeof(double), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(a_fp64, a_fp64_h, MATRIX_M * MATRIX_K * sizeof(double), cudaMemcpyHostToDevice));
   cudaErrCheck(cudaMemcpy(b_fp64, b_fp64_h, MATRIX_K * MATRIX_N * sizeof(double), cudaMemcpyHostToDevice));


   printf("\nM = %d, N = %d, K = %d\n\n", MATRIX_M, MATRIX_N, MATRIX_K);
   
   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;
 
   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
   
   cudaErrCheck(cudaEventRecord(startWMMA));
   dmma1688_example <<< gridDim, blockDim >>> (a_fp64, b_fp64, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, iterations);
   cudaErrCheck(cudaEventRecord(stopWMMA));

   
   cudaErrCheck(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(double), cudaMemcpyDeviceToHost));
 

   float wmmaTime;

   cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
   printf("gpu execution time = %.3f ms\n", wmmaTime);
   
   cudaErrCheck(cudaEventDestroy(startWMMA));
   cudaErrCheck(cudaEventDestroy(stopWMMA));
   cudaErrCheck(cudaFree(a_fp64));
   cudaErrCheck(cudaFree(b_fp64));
   cudaErrCheck(cudaFree(c_wmma));
   free(a_fp64_h);
   free(b_fp64_h);
   free(c_host_wmma);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}
