
#ifndef LEMONADEGPU_CORE_KERNELCONNECTION_CU
#define LEMONADEGPU_CORE_KERNELCONNECTION_CU
#include <LeMonADEGPU/core/kernelConnection.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
template < typename T >
__global__ void  kernelResetMultiples(
    const uint32_t ArraySize,
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
        printf("Partner_k[%d]=%d", i, Partner_k[i]);
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

__global__ void kernelfillBuffer	(uint32_t * const buffer , uint32_t * const data, const uint32_t size)
{
for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < size; i += gridDim.x * blockDim.x ){
          buffer[i]=data[i];
  }
}
__global__ void kernelCompareBuffer	(uint32_t * const buffer , uint32_t * const data, const uint32_t size)
{
for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < size-1; i += gridDim.x * blockDim.x ){
          if(i % 2 == 0 ){
            if (buffer[i] == buffer[i+1]){
                buffer[i] = 0;
                buffer[i+1] = 0;
            }
	  }else{
	      if (data[i] == data[i+1]){
		  data[i] = 0;
		  data[i+1] = 0;
	      }
	  }
  }
}
__global__ void kernelWriteResult	(uint32_t * const buffer , uint32_t * const data, const uint32_t size)
{
for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < size; i += gridDim.x * blockDim.x ){
	  data[i] = (buffer[i] == data[i]) ? data[i] : 0 ; 
	  buffer[i]=0;
  }
}
void Connection::init()
{
  std::cout << "Allocate memory for the buffer " << sizeof(uint32_t)*arraySize <<" bytes\n";
  CUDA_ERROR(cudaMalloc((void**) &dBuffer, sizeof(uint32_t)*arraySize) );
}
void Connection::clean()
{
  std::cout << "Free memory for the buffer " << sizeof(uint32_t)*arraySize <<" bytes\n";
  CUDA_ERROR(cudaFree(dBuffer));
}
Connection::~Connection(){clean();}
void Connection::resetMultipleIDs( uint32_t * crosslinkId, uint32_t * chainID, cudaStream_t mStream )
{
  
  auto sharedMemoryBytes(arraySize*2*sizeof(uint32_t));
//   std::cout << "Connection::resetMultipleIDs for an array of size " << arraySize <<"\n"
// 	    << "which needs " << sharedMemoryBytes/1024 << " kB \n";
  
  //sort for the partner array for increasing numbers :0 0 0 4 4 4 7 9 ...
  thrust::sort_by_key(thrust::device,chainID,chainID+arraySize,crosslinkId  );
  if (arraySize <6000)  
    kernelResetMultiples<<<1,nThreads,sharedMemoryBytes>>>(arraySize, chainID );
  else
  {
    auto const nThreads(128);
    auto const nBlocks(ceilDiv(arraySize,nThreads));
//     std::cout << "Connection::resetMultipleIDs use "<< nThreads << " threads and " <<nBlocks <<" nBlocks\n";
    kernelfillBuffer<<<nBlocks,nThreads,0,mStream>>>(dBuffer,chainID, arraySize);
    kernelCompareBuffer<<<nBlocks,nThreads,0,mStream>>>(dBuffer,chainID, arraySize);
    kernelWriteResult<<<nBlocks,nThreads,0,mStream>>>(dBuffer,chainID, arraySize);
//     CUDA_ERROR( cudaStreamSynchronize( mStream ) );
  }
//   std::cout << "Sort back the Ids: \n";
  //sort back to get the real indecies 
//   thrust::sort_by_key(thrust::device,crosslinkId,crosslinkId+arraySize,chainID  ); // I think this is not needed...
//   std::cout << "Sort back the Ids.done \n";
}

#endif /*LEMONADEGPU_CORE_KERNELCONNECTION_CU*/ 