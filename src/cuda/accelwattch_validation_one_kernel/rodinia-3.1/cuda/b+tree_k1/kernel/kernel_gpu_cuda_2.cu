//========================================================================================================================================================================================================200
//	findRangeK function
//========================================================================================================================================================================================================200

__global__ void 
findRangeK(	long height,

			knode *knodesD,
			long knodes_elem,

			long *currKnodeD,
			long *offsetD,
			long *lastKnodeD,
			long *offset_2D,
			int *startD,
			int *endD,
			int *RecstartD, 
			int *ReclenD)
{
#pragma unroll 100
for(uint64_t onek = 0; onek < UINT64_MAX; onek++){
	// private thread IDs
	int thid = threadIdx.x;
	int bid = blockIdx.x;

	// ???
	int i;
	for(i = 0; i < height; i++){

		if((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid+1] > startD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){
				offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
			}
		}
		if((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid+1] > endD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){
				offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
			}
		}
		__syncthreads();

		// set for next tree level
		if(thid==0){
			currKnodeD[bid] = offsetD[bid];
			lastKnodeD[bid] = offset_2D[bid];
		}
		__syncthreads();
	}

	// Find the index of the starting record
	if(knodesD[currKnodeD[bid]].keys[thid] == startD[bid]){
		RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
	}
	__syncthreads();

	// Find the index of the ending record
	// BT add += did not do anything
	if(knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]){
		ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1;
	}
}//for onek
}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
