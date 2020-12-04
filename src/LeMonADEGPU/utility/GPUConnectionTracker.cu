
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/constants.cuh>

using ID_t               = Tracker<uint8_t>::ID_t;
using T_Coordinates      = Tracker< uint8_t >::T_Coordinates;
using T_Coordinate       = Tracker< uint8_t >::T_Coordinate;
using T_BoxSize          = uint64_t;
template< typename T_UCoordinateCuda >
__global__ void kernelTrackBreaks
(
  ID_t           * const dID1      ,
  ID_t           * const dID2      ,
  size_t           const dSize     ,
  int32_t          const dOffsetA  ,
  int32_t          const dOffsetB  ,
  ID_t           * const diNewToi  ,
  T_Coordinates * const __restrict__ dOutputID1,
  T_Coordinates * const dOutputID2,
  typename CudaVec4< T_UCoordinateCuda >::value_type const * const __restrict__ dpPolymerSystem,
  T_Coordinates const * const dpiPolymerSystemSortedVirtualBox 
)
{
  for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < dSize; i += gridDim.x * blockDim.x )
  {
    auto iMonomer(dID1[i]);
    auto iPartner(dID2[i]);
    if (iPartner == 0 || iMonomer == 0 ) 
    {
      continue; //no Partner found -> go to next Crosslink in the grid 
    }
    iMonomer--;
    iPartner--;
    dID1[i]=0;
    dID2[i]=0;
    auto gID1(iMonomer + dOffsetA);
    auto rsmall = dpPolymerSystem[ gID1 ];
    //cast from T_UCoordinateCuda = uint32_t "down" to int32_t
    T_Coordinates rSorted = { T_Coordinate( rsmall.x ), T_Coordinate( rsmall.y ),
                              T_Coordinate( rsmall.z ), T_Coordinate( rsmall.w )};
    auto nPos = dpiPolymerSystemSortedVirtualBox[ gID1 ];
    rSorted.x += nPos.x * dcBoxX;
    rSorted.y += nPos.y * dcBoxY;
    rSorted.z += nPos.z * dcBoxZ;
    rSorted.w  = diNewToi[gID1];
    dOutputID1[i] = rSorted;

    auto gID2(iPartner+dOffsetB);
    rsmall = dpPolymerSystem[ gID2 ];
    //cast from T_UCoordinateCuda = uint32_t "down" to int32_t
    rSorted = { T_Coordinate( rsmall.x ), T_Coordinate( rsmall.y ),
                T_Coordinate( rsmall.z ), T_Coordinate( rsmall.w )};
    nPos = dpiPolymerSystemSortedVirtualBox[ gID2 ];
    rSorted.x += nPos.x * dcBoxX;
    rSorted.y += nPos.y * dcBoxY;
    rSorted.z += nPos.z * dcBoxZ;
    rSorted.w  = ( (diNewToi[gID2]+1)<<1 )+0;
    dOutputID2[i] = rSorted;

  }
}
template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::trackBreaks( ID_t * const ID1     ,
			   ID_t * const ID2     ,
			   size_t const size    ,     
			   ID_t * const diNewToi,
			   int32_t const offsetA,
			   int32_t const offsetB,
			   uint32_t const mAge  ,
         MirroredVector< T_UCoordinatesCuda >const * const mPolymerSystemSorted , 
         MirroredVector< T_Coordinates      >const * const mviPolymerSystemSortedVirtualBox  )
{
  auto nThreads(256);
  auto nBlocks(ceilDiv(size,nThreads));
  kernelTrackBreaks<T_UCoordinateCuda><<<nBlocks,nThreads,0, mStream>>>(
  ID1, 
  ID2, 
  size, 
  offsetA,
  offsetB,
  diNewToi, 
  BondHistoryID1->gpu + counter*nIDs,
  BondHistoryID2->gpu + counter*nIDs,
  mPolymerSystemSorted->gpu,
  mviPolymerSystemSortedVirtualBox->gpu
  );  
  age.push_back(mAge);
  increaseCounter();	
  if(counter == bufferSize )
  dumpReactions();
}

template< typename T_UCoordinateCuda >
__global__ void kernelTrackConnections
(
  ID_t           * const dID1      ,
  ID_t           * const dID2      ,
  size_t           const dSize     ,
  int32_t          const dOffsetA  ,
  int32_t          const dOffsetB  ,
  ID_t           * const diNewToi  ,
  T_Coordinates * const dOutputID1,
  T_Coordinates * const dOutputID2,
  typename CudaVec4< T_UCoordinateCuda >::value_type const * const __restrict__ dpPolymerSystem,
  T_Coordinates const * const dpiPolymerSystemSortedVirtualBox
)
{
  for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < dSize; i += gridDim.x * blockDim.x )
    {
      auto iMonomer(dID1[i]);
      auto iPartner(dID2[i]);
      dID1[i]=0;
      dID2[i]=0;
      if (iPartner == 0 || iMonomer == 0 ) {
	      continue; //no Partner found -> go to next Crosslink in the grid 
      }
      iMonomer--;
      iPartner--;
      auto gID1(iMonomer + dOffsetA);
      auto rsmall = dpPolymerSystem[ gID1 ];
      //cast from T_UCoordinateCuda = uint32_t "down" to int32_t
      T_Coordinates rSorted = { T_Coordinate( rsmall.x ), T_Coordinate( rsmall.y ),
                                T_Coordinate( rsmall.z ), T_Coordinate( rsmall.w )};
      auto  nPos = dpiPolymerSystemSortedVirtualBox[ gID1 ];
      rSorted.x += nPos.x * dcBoxX;
      rSorted.y += nPos.y * dcBoxY;
      rSorted.z += nPos.z * dcBoxZ;
      rSorted.w  = diNewToi[gID1];
      dOutputID1[i] = rSorted;
      // dOutputID1[i].x = diNewToi[iMonomer + dOffsetA ];

      auto gID2(iPartner+dOffsetB);
      rsmall = dpPolymerSystem[ gID2 ];
      //cast from T_UCoordinateCuda = uint32_t "down" to int32_t
      rSorted = { T_Coordinate( rsmall.x ), T_Coordinate( rsmall.y ),
                  T_Coordinate( rsmall.z ), T_Coordinate( rsmall.w )};
      nPos = dpiPolymerSystemSortedVirtualBox[ gID2 ];
      rSorted.x += nPos.x * dcBoxX;
      rSorted.y += nPos.y * dcBoxY;
      rSorted.z += nPos.z * dcBoxZ;
      rSorted.w  = ( (diNewToi[gID2]+1)<<1 )+1;
      dOutputID2[i] = rSorted;
      // dOutputID2[i].x = ( (diNewToi[iPartner+dOffsetB]+1)<<1 )+1;
//       output[miNewToi[iMonomer-1+offsetA]]= ( (miNewToi[iPartner-1+offsetB]+1)<<1 )+1;
//       printf("Bonds Mon1 = %d  Mon2 = %d Id1=%d Id2=%d %d %d  \n ", iMonomer+dOffsetA, iPartner+dOffsetB, diNewToi[iMonomer+dOffsetA],diNewToi[iPartner+dOffsetB], dOffsetA,dOffsetB );
    }
}

template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::trackConnections( 
        ID_t * const ID1     ,
        ID_t * const ID2     ,
        size_t const size    ,     
				ID_t * const diNewToi,
				int32_t const offsetA,
				int32_t const offsetB,
        uint32_t const mAge,
        MirroredVector< T_UCoordinatesCuda >const * const  mPolymerSystemSorted , 
        MirroredVector< T_Coordinates      >const * const mviPolymerSystemSortedVirtualBox
      )
{
  auto nThreads(256);	
  auto nBlocks(ceilDiv(size,nThreads));
//   std::cout << "Tracker::trackConnections: offsetA= "<< offset << " mAge= " << mAge  << " size= " << size <<std::endl;

  kernelTrackConnections<T_UCoordinateCuda><<<nBlocks,nThreads,0, mStream>>>(
  ID1, 
  ID2, 
  size, 
  offsetA,
  offsetB,
  diNewToi, 
  BondHistoryID1->gpu + counter*nIDs,
  BondHistoryID2->gpu + counter*nIDs,
  mPolymerSystemSorted->gpu,
  mviPolymerSystemSortedVirtualBox->gpu
  );
  age.push_back(mAge);
  increaseCounter();
  if(counter == bufferSize )
  dumpReactions();
}

template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::init(uint32_t bufferSize_, uint32_t nIDs_, cudaStream_t mStream_,
  T_BoxSize const boxX,
  T_BoxSize const boxY,
  T_BoxSize const boxZ)
{
  bufferSize=bufferSize_; nIDs=nIDs_; mStream=mStream_;
  BaseClass::setInformationSize(10);
  BaseClass::addComment("MCS Bond/Break ID1 Position1 ID2 Position2 ");
  std::cout << "Tracker::init: each BondHistory can take " 
            << 2*bufferSize*nIDs << " number of elements with " 
            << 2*bufferSize*nIDs *sizeof(T_Coordinates)/1024.<< " kB \n";
  BondHistoryID1 = new MirroredVector< T_Coordinates >( 2*bufferSize*nIDs, mStream ); //essentially the ids of the first monomer 
  BondHistoryID2 = new MirroredVector< T_Coordinates >( 2*bufferSize*nIDs, mStream ); //essentially the ids of the second monomer
  { decltype( dcBoxX      ) x = boxX     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
  { decltype( dcBoxY      ) x = boxY     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
  { decltype( dcBoxZ      ) x = boxZ     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
}

template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::increaseCounter()
{
  counter++;
}

template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::dumpReactions()
{
  BaseClass::setBufferSize(bufferSize);
  BondHistoryID1->popAsync();
  BondHistoryID2->popAsync();
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
//   std::cout <<"counter= " << counter << " nIDs="<<nIDs << std::endl;
  for (uint32_t j=0 ; j < counter ; j ++ ) {
    int32_t currentAge(age[j]);
    for(uint32_t i =0 ; i < nIDs; i ++)
    {
      auto index(i+nIDs*j);
      auto Mon1(BondHistoryID1->host[index]);
      auto Mon2(BondHistoryID2->host[index]);
      auto MonID1(Mon1.w); 
      auto MonID2(Mon2.w); 
      if( MonID2  > 0 )
      {
        std::vector<int32_t> vec;
        vec.push_back(currentAge); //time 
        vec.push_back( MonID2 & 1 ); // either 0 or 1 for remove or add 
        MonID2 = (MonID2 >> 1) -1;
        if (MonID1 < MonID2 ) {
          vec.push_back(MonID1); 
          vec.push_back(Mon1.x);
          vec.push_back(Mon1.y);
          vec.push_back(Mon1.z);
          vec.push_back(MonID2); 
          vec.push_back(Mon2.x);
          vec.push_back(Mon2.y);
          vec.push_back(Mon2.z);
        }else {
          vec.push_back(MonID2); 
          vec.push_back(Mon2.x);
          vec.push_back(Mon2.y);
          vec.push_back(Mon2.z);
          vec.push_back(MonID1); 
          vec.push_back(Mon1.x);
          vec.push_back(Mon1.y);
          vec.push_back(Mon1.z);
        }
        // vec.push_back(std::min(Mon1,Mon2));   
        // vec.push_back(std::max(Mon1,Mon2));
        BaseClass::addConnection(vec);
        BondHistoryID1->host[index].w=0;
        BondHistoryID1->host[index].x=0;
        BondHistoryID1->host[index].y=0;
        BondHistoryID1->host[index].z=0;
        BondHistoryID2->host[index].w=0;
        BondHistoryID2->host[index].x=0;
        BondHistoryID2->host[index].y=0;
        BondHistoryID2->host[index].z=0;
      }
    }
  }
  BaseClass::dumpReactions();
  counter=0;
  age.resize(0);
  BondHistoryID1->push();
  BondHistoryID2->push();
  
}