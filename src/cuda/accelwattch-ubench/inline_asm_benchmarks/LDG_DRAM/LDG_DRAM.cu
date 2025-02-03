
//This benchmark measures the maximum read bandwidth of GPU memory
//Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

//This code have been tested on Volta V100 architecture
//You can check the mem BW from the NVPROF (dram_read_throughput+dram_write_throughput)

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCKS_NUM 160
#define THREADS_NUM 1024 //thread number/block
#define TOTAL_THREADS (BLOCKS_NUM*THREADS_NUM)
#define ARRAY_SIZE 8388608   //Array size has to exceed L2 size to avoid L2 cache residence
#define WARP_SIZE 32 
#define L2_SIZE 1572864 //number of floats L2 can store
#define clock_freq_MHZ 1132

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// A thread 0 will do all of my initializations where A -> B -> C -> D -> E -> F
__global__ void pointers_init(uint64_t* A,  uint64_t* B, uint64_t* C, uint64_t* D, uint64_t* E, uint64_t* F){

  uint32_t tid = blockIdx.x*blockDim.x + threadIdx.x;

  if(tid == 0){
    for(uint32_t i = 0; i < ARRAY_SIZE/2; i=i+2){
        A[2*i] =   (uint64_t)(B + 2*i + 2);
        A[2*i+1] = (uint64_t)(B + 2*i + 3);
        B[2*i] =   (uint64_t)(C + 2*i + 2);
        B[2*i+1] = (uint64_t)(C + 2*i + 3);
        C[2*i] =   (uint64_t)(D + 2*i + 2);
        C[2*i+1] = (uint64_t)(D + 2*i + 3);
        D[2*i] =   (uint64_t)(E + 2*i + 2);
        D[2*i+1] = (uint64_t)(E + 2*i + 3);
        E[2*i] =   (uint64_t)(F + 2*i + 2);
        E[2*i+1] = (uint64_t)(F + 2*i + 3);
        F[2*i] =   (uint64_t)(A + 2*i + 2);
        F[2*i+1] = (uint64_t)(A + 2*i + 3);
    }
	A[ARRAY_SIZE-2] = (uint64_t)(B);
	A[ARRAY_SIZE-1] = (uint64_t)(B + 1);
	B[ARRAY_SIZE-2] = (uint64_t)(C);
	B[ARRAY_SIZE-1] = (uint64_t)(C + 1);
	C[ARRAY_SIZE-2] = (uint64_t)(D);
	C[ARRAY_SIZE-1] = (uint64_t)(D + 1);
	D[ARRAY_SIZE-2] = (uint64_t)(E);
	D[ARRAY_SIZE-1] = (uint64_t)(E + 1);
	E[ARRAY_SIZE-2] = (uint64_t)(F);
	E[ARRAY_SIZE-1] = (uint64_t)(F + 1);
	F[ARRAY_SIZE-2] = (uint64_t)(A);
	F[ARRAY_SIZE-1] = (uint64_t)(A + 1);
  }
}



/*
Pointer Chasing
 */

__global__ void dram_pointer_chase (uint64_t* A, uint64_t* F, unsigned long long iterations){
	// block and thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;


  	// a register to avoid compiler optimization
  	uint64_t *ptr = A + tid;
  	uint64_t ptr1, ptr0;

  	// initialize the thread pointer with the start address of the array
  	// use ca modifier to cache the in L1
  	asm volatile ("{\t\n"
  	  "ld.global.cv.u64 %0, [%1];\n\t"
  	  "}" : "=l"(ptr1) : "l"(ptr) : "memory"
  	);

	// synchronize all threads
	asm volatile ("bar.sync 0;");

  	// pointer-chasing iterations times
	// because of the initialization, we should constantly be missing 
  	#pragma unroll 100
  	for(unsigned long long i=0; i<iterations; ++i) { 
  	  asm volatile ("{\t\n"
  	    "ld.global.cv.u64 %0, [%1];\n\t"
  	    "}" : "=l"(ptr0) : "l"((uint64_t*)ptr1) : "memory"
  	  );
  	  ptr1 = ptr0;    //swap the register for the next load
  	}

  	// write data back to memory
  	F[tid] = ptr1;
  }

	// synchronize all threads
	asm volatile ("bar.sync 0;");
}

int main(int argc, char** argv){
  unsigned long long iterations;
  if (argc != 2){
    fprintf(stderr,"usage: %s #iterations \n",argv[0]);
    exit(1);
  }
  else {
    iterations = atoll(argv[1]);
  }
 printf("Power Microbenchmarks with iterations %lu\n",iterations);
	uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
	uint64_t *A = (uint64_t*) malloc(ARRAY_SIZE*sizeof(uint64_t));
	uint64_t *B = (uint64_t*) malloc(ARRAY_SIZE*sizeof(uint64_t));
	uint64_t *C = (uint64_t*) malloc(ARRAY_SIZE*sizeof(uint64_t));
	uint64_t *D = (uint64_t*) malloc(ARRAY_SIZE*sizeof(uint64_t));
	uint64_t *E = (uint64_t*) malloc(ARRAY_SIZE*sizeof(uint64_t));
	uint64_t *F = (uint64_t*) malloc(ARRAY_SIZE*sizeof(uint64_t));


	uint64_t *A_g;
	uint64_t *B_g;
	uint64_t *C_g;
	uint64_t *D_g;
	uint64_t *E_g;
	uint64_t *F_g;

	gpuErrchk( cudaMalloc(&A_g, ARRAY_SIZE*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&B_g, ARRAY_SIZE*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&C_g, ARRAY_SIZE*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&D_g, ARRAY_SIZE*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&E_g, ARRAY_SIZE*sizeof(uint64_t)) );
	gpuErrchk( cudaMalloc(&F_g, ARRAY_SIZE*sizeof(uint64_t)) );


	gpuErrchk( cudaMemcpy(A_g, A, ARRAY_SIZE*sizeof(uint64_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(B_g, B, ARRAY_SIZE*sizeof(uint64_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(D_g, D, ARRAY_SIZE*sizeof(uint64_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(E_g, E, ARRAY_SIZE*sizeof(uint64_t), cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy(F_g, F, ARRAY_SIZE*sizeof(uint64_t), cudaMemcpyHostToDevice) );

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	mem_bw<<<BLOCKS_NUM,THREADS_NUM>>>(A_g, B_g, C_g, D_g, E_g, F_g, iterations);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk( cudaMemcpy(C, C_g, ARRAY_SIZE*sizeof(uint64_t), cudaMemcpyDeviceToHost) );

	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	 printf("gpu execution time = %.3f ms\n", elapsedTime);  
}

