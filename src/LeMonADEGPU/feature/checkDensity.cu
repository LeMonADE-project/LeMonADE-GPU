#include <LeMonADEGPU/feature/checkDensity.h>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/utility/cudacommon.hpp>
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
///some simple functions for the power handling/////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
uint32_t nextPow2(uint32_t x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}
////////////////////////////////////////////////////////////////////////////////
uint32_t isPowerOfTwo (uint32_t x)
{
  return ((x != 0) && ((x & (~x + 1)) == x));
}
////////////////////////////////////////////////////////////////////////////////
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
////////////////////////////////////////////////////////////////////////////////
//constructor
checkDensity::checkDensity():BoxX(0),BoxY(0),BoxZ(0){};
checkDensity::checkDensity(boxType BoxX_, boxType BoxY_, boxType BoxZ_)
{
    setBoxSizes(BoxX_,BoxY_,BoxZ_);
};
////////////////////////////////////////////////////////////////////////////////
///member functions ////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//! set the box size and copy them to the constant memory
void checkDensity::setBoxSizes(boxType BoxX_, boxType BoxY_, boxType BoxZ_){
	BoxX=BoxX_;
	BoxY=BoxY_;
	BoxZ=BoxZ_;
	// copy to constant device memory 
    CUDA_ERROR(cudaMemcpyToSymbol(BoxX    , &dBoxX   , sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxY    , &dBoxY   , sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ    , &dBoxZ   , sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ-1  , &dBoxZM1 , sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ-2  , &dBoxZM2 , sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ-3  , &dBoxZM3 , sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ/2+1, &dBoxZhP1, sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ/2+2, &dBoxZhP2, sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ/2-3, &dBoxZhM3, sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ/2-2, &dBoxZhM2, sizeof(boxType)));
    CUDA_ERROR(cudaMemcpyToSymbol(BoxZ/2+3, &dBoxZhP3, sizeof(boxType)));
}
////////////////////////////////////////////////////////////////////////////////
void checkDensity::init(uint32_t NMonomers_, uint32_t nSortedMonomers, cudaDeviceProp&  mCudaProps ) {            
    NMonomers=NMonomers_;
    //the average number of monomers in four slices of the box 
    float hAvMonomerNumberInShearVolume=static_cast<float>(NMonomers)*4.0/static_cast<float>(BoxZ);
    CUDA_ERROR(cudaMemcpyToSymbol(dAvMonomerNumberInShearVolume, &hAvMonomerNumberInShearVolume, sizeof(float)));
    std::cout << "set average number of monomers in shear volume to  " << hAvMonomerNumberInShearVolume << std::endl;

    //parameter needed to start the kernel for the reduction 
    //see: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 	const uint32_t maxThreads(512);
    const uint32_t maxBlocks(512);
    arraySize=nextPow2(nSortedMonomers);
    nThreads = (arraySize < maxThreads*2) ? nextPow2((arraySize + 1)/ 2) : maxThreads;
    nBlocks = (arraySize + (nThreads * 2 - 1)) / (nThreads * 2);
    //check wether the number of monomers can be handled
	if ((float)nThreads*nBlocks > (float)mCudaProps.maxGridSize[0] * mCudaProps.maxThreadsPerBlock)
	    throw std::runtime_error("number of monomers is too large!\n");
    //check wether the calculated block and thread sizes fits 
	if (nBlocks > mCudaProps.maxGridSize[0])
	{
	    printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
		       nBlocks, mCudaProps.maxGridSize[0], nThreads*2, nThreads);
	    nBlocks /= 2;
	    nThreads *= 2;
    }
    //check if the block size is power of two
	if ( 0 == isPowerOfTwo(nBlocks)) {
	  std::cout<<"numBlock for reduction is not power of two  "<<nBlocks<<std::endl; 
	  nBlocks = nextPow2(nBlocks);
	  std::cout <<"new num block for reduction is  "<<nBlocks<<std::endl; 
	}
	//
	nBlocks = MIN(maxBlocks, nBlocks);
	std::cout << "Number of nThreads for summing arrays " << nThreads <<std::endl;
	std::cout << "Using " << nBlocks << " blocks for reduction "<<std::endl;

    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    //memory allocation 
    //monomer counting arrays
	mCountMiddleMonos   = new MirroredVector<intArray> (arraySize);
	mCountBoundaryMonos = new MirroredVector<intArray> (arraySize);

    //output array for reduction 
    mReducedCountMiddleMonos   = new MirroredVector<intArray>(nBlocks);
    mReducedCountBoundaryMonos = new MirroredVector<intArray>(nBlocks);
    
    //number of monoemrs in the sheared parts of the box 
	CUDA_ERROR(cudaMalloc((void **) &dMonomerNumber_in_ShearVolumeBoundary, sizeof(uint32_t)));
	CUDA_ERROR(cudaMalloc((void **) &dMonomerNumber_in_ShearVolumeMiddle, sizeof(uint32_t)));
    ///////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////
    //count monomers on host in the middle and the boundary
    hMonomerNumber_in_ShearVolumeMiddle = 0;
    hMonomerNumber_in_ShearVolumeBoundary = 0;
    
	for(uint32_t IDMonomer = 0; IDMonomer<arraySize; IDMonomer++){
	//   if (IDMonomer < NMonomers){
		// const intCUDA zPosMono = PolymerSystem_host[IDMonomer*4+2];
		// const uint32_t zPosMonoAbs = ((zPosMono) & BoxZ-1 ());
		mCountMiddleMonos->host[IDMonomer] = static_cast<intArray>(0);
        mCountBoundaryMonos->host[IDMonomer] = static_cast<intArray>(0);
        
		// if(zPosMonoAbs < 2 || zPosMonoAbs >= BoxZ-2 ) {
        //     mCountBoundaryMonos->host[IDMonomer] = static_cast<intArray>(1);
        //     hMonomerNumber_in_ShearVolumeBoundary++;
        // }
		// if((zPosMonoAbs <= (BoxZ/2+1)) && (zPosMonoAbs > (BoxZ/2-3)) ) {
        //     mCountBoundaryMonos->host[IDMonomer] = static_cast<intArray>(1);
        //     hMonomerNumber_in_ShearVolumeMiddle++;
        // }

	//   }
    }
    mCountMiddleMonos->pushAsync();
    mCountBoundaryMonos->pushAsync();
	//push the number of monomers in the sheared parts to the gpu 
	CUDA_ERROR(cudaMemcpy(dMonomerNumber_in_ShearVolumeBoundary, &hMonomerNumber_in_ShearVolumeBoundary,sizeof(uint32_t), cudaMemcpyHostToDevice) );
	CUDA_ERROR(cudaMemcpy(dMonomerNumber_in_ShearVolumeMiddle, &hMonomerNumber_in_ShearVolumeMiddle,sizeof(uint32_t), cudaMemcpyHostToDevice) );	
	std::cout << " number of monomer in boundaries "<<hMonomerNumber_in_ShearVolumeBoundary << std::endl;
	std::cout << " number of monomer in middle of box "<<hMonomerNumber_in_ShearVolumeMiddle << std::endl;

}


////////////////////////////////////////////////////////////////////////////////
///start kernel calculating the number density//////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//start calculate number density
//reduction from example number 7
//see: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blocksize>
__global__ void reduction(intArray *g_idata, intArray *g_odata, unsigned int n ){
  volatile __shared__ intArray sdata[blocksize];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int gridSize = blocksize*2*gridDim.x;
  sdata[tid]=0;
  while(i < n){
     sdata[tid] += g_idata[i] + g_idata[i+blocksize];
     i += gridSize;
  } 
  __syncthreads();
//   do reduction in shared mem

  if (blocksize >=512){
    if (tid < 256) {sdata[tid] += sdata[tid + 256];} __syncthreads();
  }
  if (blocksize >=256){
    if (tid < 128) {sdata[tid] += sdata[tid + 128];} __syncthreads();
  }
  if (blocksize >=128){
    if (tid <  64) {sdata[tid] += sdata[tid +  64];} __syncthreads();
  }
  
  if (tid < 32)
  {
    if(blocksize >= 64) sdata[tid] += sdata[tid + 32];
    if(blocksize >= 32) sdata[tid] += sdata[tid + 16];
    if(blocksize >= 16) sdata[tid] += sdata[tid +  8];
    if(blocksize >=  8) sdata[tid] += sdata[tid +  4];
    if(blocksize >=  4) sdata[tid] += sdata[tid +  2];
    if(blocksize >=  2) sdata[tid] += sdata[tid +  1];
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
////////////////////////////////////////////////////////////////////////////////
void checkDensity::calcDensity()
{
    //call the kernel with the correct template parameter depending on the number of nThreads 
    switch(nThreads){
        case 512:
            reduction<512><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<512><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 256:
            reduction<256><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<256><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 128:
            reduction<128><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<128><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 64:
            reduction<64><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<64><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 32:
            reduction<32><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<32><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 16:
            reduction<16><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<16><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 8:
            reduction<8><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<8><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 4:
            reduction<4><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<4><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 2:
            reduction<2><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<2><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
        case 1:
            reduction<1><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountMiddleMonos->gpu,mReducedCountMiddleMonos->gpu, NMonomers);
            reduction<1><<<nBlocks,nThreads,nThreads*sizeof(intArray)>>>(mCountBoundaryMonos->gpu,mReducedCountBoundaryMonos->gpu, NMonomers);break;
    }
    //copy to host 
    mReducedCountBoundaryMonos->pop();
    mReducedCountMiddleMonos->pop();
    //sum up reamining array 
    hMonomerNumber_in_ShearVolumeBoundary=0;
    hMonomerNumber_in_ShearVolumeMiddle=0;
    for(auto i=0; i<nBlocks;i++){
        hMonomerNumber_in_ShearVolumeBoundary += mReducedCountBoundaryMonos->host[i];
        hMonomerNumber_in_ShearVolumeMiddle   += mReducedCountMiddleMonos->host[i];
    }

    //copy to device 
    CUDA_ERROR(cudaMemcpy(dMonomerNumber_in_ShearVolumeBoundary, &hMonomerNumber_in_ShearVolumeBoundary,sizeof(uint32_t), cudaMemcpyHostToDevice) );
    CUDA_ERROR(cudaMemcpy(dMonomerNumber_in_ShearVolumeMiddle, &hMonomerNumber_in_ShearVolumeMiddle,sizeof(uint32_t), cudaMemcpyHostToDevice) );

    //empty the mirrored vectors for the next step
    mReducedCountBoundaryMonos->memsetAsync(0);
    mReducedCountMiddleMonos->memsetAsync(0);
    mCountMiddleMonos->memsetAsync(0);
    mCountBoundaryMonos->memsetAsync();
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//end calculate number density
template <typename T_UCoordinateCuda >
__global__ void kernelCount( 
    typename CudaVec4< T_UCoordinateCuda >::value_type const * const __restrict__ dpPolymerSystem,
    intArray *dCountMiddleMonos,
    intArray *dCountBoundaryMonos, 
    uint32_t const nMons){
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
            iMonomer < nMons; iMonomer += gridDim.x * blockDim.x ){
        auto const r0 = dpPolymerSystem[ iMonomer ];
        //only works for power of two lattices! 
        auto const z=r0.z&dBoxZM1; 
        dCountMiddleMonos[iMonomer]   = static_cast<intArray>(0);
        dCountBoundaryMonos[iMonomer] = static_cast<intArray>(0);
        if( (z <= dBoxZhP1) && (z > dBoxZhM3) ) dCountMiddleMonos[iMonomer] = static_cast<intArray>(1);
        if( (z < 2        ) || (z >= dBoxZM2) ) dCountBoundaryMonos[iMonomer] = static_cast<intArray>(1);
    }
}
template <typename T_UCoordinateCuda >
void checkDensity::launch_countMonomers(
    typename CudaVec4< T_UCoordinateCuda >::value_type const * const __restrict__ dpPolymerSystem,
    size_t nMons, 
    size_t const nBlocksSpecies,
    uint32_t const nThreadsSprecies){

kernelCount<<<nBlocksSpecies,nThreadsSprecies >>> (
    dpPolymerSystem, 
    mCountMiddleMonos->gpu, 
    mCountBoundaryMonos->gpu, 
    nMons );
} 
// template void checkDensity::launch_countMonomers<uint8_t >(uint8_t  const * const,  size_t, const size_t, const uint32_t );     
// template void checkDensity::launch_countMonomers<uint16_t>(uint16_t ,  size_t, const size_t, const uint32_t );
// template void checkDensity::launch_countMonomers<uint32_t>(uint32_t ,  size_t, const size_t, const uint32_t );
// template void checkDensity::launch_countMonomers<int16_t >(int16_t  ,  size_t, const size_t, const uint32_t );
// template void checkDensity::launch_countMonomers<int32_t >(int32_t  ,  size_t, const size_t, const uint32_t );

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////