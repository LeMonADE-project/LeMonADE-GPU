
#ifndef LEMONADEGPU_CORE_KERNELCONNECTION_CU
#define LEMONADEGPU_CORE_KERNELCONNECTION_CU
#include <LeMonADEGPU/core/kernelConnection.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
template < typename T >
__global__ void  kernel(
    const int ArraySize,
    T *  Partner_k
)
{
    extern __shared__ T dat[];
    T* sdataA = (T*)dat; 
//     int* sdataB = (int*)&sdataA[blocksize*blockDim.x];
    T* sdataB = (T*)&sdataA[ArraySize];
    /**The algorithm cannot handle the boundaries of the arrays between blocks, thus
     * we are restricted to one block and the maximum number of threads. Certainly the
     *performance will not be good... Due to this reason we use a kind of grid striding loop.  
     * blockDim.x = max_threads
     * gridDim.x = 1 -> one block
     */
    for (int i = threadIdx.x; i < ArraySize; i += blockDim.x ) 
    {
        sdataA[i] = Partner_k[i];
        sdataB[i] = Partner_k[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < ArraySize-1; i += blockDim.x ) 
    {
        if(i % 2 == 0 ){
            if (sdataA[i] == sdataA[i+1]){
                sdataA[i] = 0;
                sdataA[i+1] = 0;
            }
        }else{
            if (sdataB[i] == sdataB[i+1]){
                sdataB[i] = 0;
                sdataB[i+1] = 0;
            }
        }
    }
    __syncthreads();
    //If values of both arrays (A and B) are the same, then the value 
    //can be taken ( either is already zero or an index ).
    //Otherwise, the index occured more than once and is thus thrown out.
    for (int i = threadIdx.x; i < ArraySize; i += blockDim.x ) 
        Partner_k[i] = (sdataA[i] == sdataB[i]) ? sdataA[i] : 0 ; 
}

void Connection::resetMultipleIDs( uint32_t * crosslinkId, uint32_t * chainID )
{
  //sort for the partner array for increasing numbers :0 0 0 4 4 4 7 9 ...
  thrust::sort_by_key(thrust::device,chainID,chainID+arraySize,crosslinkId  );
    
  kernel<<<1,nThreads,arraySize*2*sizeof(uint32_t)>>>(arraySize, chainID );
  
  //sort back to get the real indecies 
  thrust::sort_by_key(thrust::device,crosslinkId,crosslinkId+arraySize,chainID  );
}

#endif /*LEMONADEGPU_CORE_KERNELCONNECTION_CU*/ 