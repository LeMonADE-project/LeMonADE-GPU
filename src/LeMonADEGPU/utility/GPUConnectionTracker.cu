
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
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



/**
 * @brief calculates the minimal distances of images for one component 
 * @return int 
 * @param x1 absolute coordinate
 * @param x2 absolute coordinate
 * @param LatticeSize size of the box in the direction of the given coordinates
 */
 template <class T >
__device__ __host__  T inline MinImageDistanceComponentForPowerOfTwo(const T x, const uint32_t latticeSize ){
  //this is only valid for absolute coordinates
  uint32_t latticeSizeM1(latticeSize-1);
  return ( ((x&latticeSizeM1) < (latticeSize/2)) ? (x & latticeSizeM1) :  -(-x & latticeSizeM1));
}

template <class T >
__device__  T_Coordinates inline calcVector(const T vec, const T_Coordinates pVec ){
  //cast from T_UCoordinateCuda = uint32_t "down" to int32_t
  T_Coordinates rSorted = { T_Coordinate( vec.x ), T_Coordinate( vec.y ),T_Coordinate( vec.z ), T_Coordinate( vec.w )};
  rSorted.x += pVec.x * dcBoxX;
  rSorted.y += pVec.y * dcBoxY;
  rSorted.z += pVec.z * dcBoxZ;
  return rSorted; 
}

__device__ T_Coordinates inline MinImageVector(const T_Coordinates vec1, const T_Coordinates vec2) {

  T_Coordinates vec={   MinImageDistanceComponentForPowerOfTwo( vec2.x-vec1.x, dcBoxX ), 
                        MinImageDistanceComponentForPowerOfTwo( vec2.y-vec1.y, dcBoxY ), 
                        MinImageDistanceComponentForPowerOfTwo( vec2.z-vec1.z, dcBoxZ ), 
                        T_Coordinate(0)
                    };
  return vec;

}
__device__ T_Coordinates addVectors(const T_Coordinates vec1, const T_Coordinates vec2)
{
 return T_Coordinates{vec1.x+vec2.x, vec1.y+vec2.y,vec1.z+vec2.z,vec1.w+vec2.w,};
}
__device__ T_Coordinates substractVectors(const T_Coordinates vec1, const T_Coordinates vec2)
{
 return T_Coordinates{vec1.x-vec2.x, vec1.y-vec2.y,vec1.z-vec2.z,vec1.w-vec2.w,};
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
  ID_t           * const diToiNew  ,
  T_Coordinates  * const dOutputID1,
  T_Coordinates  * const dOutputID2,
  ID_t           * const dChainID  ,
  typename CudaVec4< T_UCoordinateCuda >::value_type const * const __restrict__ dpPolymerSystem,
  T_Coordinates const * const dpiPolymerSystemSortedVirtualBox,
  ID_t           * const dMidToNid,
  ID_t           * const dNidToMid,
  ID_t           * const dNidToNid,
  ID_t           * const dNidToCid
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
      //Crosslink1:
      auto gID1(iMonomer + dOffsetA);
      T_Coordinates rCrosslink1( calcVector(dpPolymerSystem[ gID1 ],dpiPolymerSystemSortedVirtualBox[ gID1 ]  ) ); 
      rCrosslink1.w  = diNewToi[gID1];
      dOutputID1[i] = rCrosslink1;
      ////////////////////////////
      // Chain monomer 1 : 
      auto gID2(iPartner+dOffsetB); // one chain end monomer, global id 
      auto gID2Old(diNewToi[gID2]);
      auto reducedMonChainID1(dMidToNid[gID2Old]); // reduced chain id -> chain start 
      auto reducedMonChainID2(dNidToNid[reducedMonChainID1 ]); // reduced chain id -> chain end 
      auto crosslinkID(dNidToCid[reducedMonChainID2]); // second cross link id, id zero there is no cross link connected to the first cross link, global id + 1
      // auto gMonoOnChain2(diToiNew[dNidToMid[reducedMonChainID2]] );// second  chain end monomer, global id 
      dChainID[i]=(reducedMonChainID1-(reducedMonChainID1%2) )/2 ; // chain ID where the first monomer is attached to 
      T_Coordinates rRefoldCrosslink2={ 0 , 0 ,  0 ,  3 }; // 3=(( 0+1)<<1 )+1 ; 
      dNidToCid[reducedMonChainID1]=gID1+1;
      if( crosslinkID >0 ){
        crosslinkID= crosslinkID-1;
        // The cross links and the chains are connected across periodic images. Thus the bonds can be "bond+multiple of box size". 
        // To reduce this to the real value, we calculate the bond1 from the cross link to the chain start (reduce to MIC) 
        // , add the vector from the end-to-end vector of the chain and the vector of the chain end to the second cross link(MIC again).
        // printf("gID2=%d gMonoOnChain2=%d crosslinkID=%d\n",gID2, gMonoOnChain2, crosslinkID);
        // position of the chain start 
        // T_Coordinates rChain1(calcVector(dpPolymerSystem[ gID2 ], dpiPolymerSystemSortedVirtualBox[ gID2 ]));
        // // position of the chain end 
        // T_Coordinates rChain2(calcVector(dpPolymerSystem[ gMonoOnChain2 ], dpiPolymerSystemSortedVirtualBox[ gMonoOnChain2 ]));
        // position of the connected cross link 
        T_Coordinates rCrosslink2(calcVector(dpPolymerSystem[ crosslinkID ], dpiPolymerSystemSortedVirtualBox[ crosslinkID ]));
        //calculate the refolded position of the second cross link 
        // rRefoldCrosslink2=( rCrosslink1 + MinImageVector(rCrosslink1,rChain1) + substractVectors(rChain2,rChain1) + MinImageVector( rChain2,rCrosslink2)  );
        rRefoldCrosslink2=rCrosslink2;
        rRefoldCrosslink2.w  = ( (diNewToi[crosslinkID]+1)<<1 )+1;
      }
      dOutputID2[i] = rRefoldCrosslink2;

    }
}

template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::trackConnections( 
        ID_t * const ID1     ,
        ID_t * const ID2     ,
        size_t const size    ,     
        ID_t * const diNewToi,
        ID_t * const diToiNew,
				int32_t const offsetA,
				int32_t const offsetB,
        uint32_t const mAge,
        MirroredVector< T_UCoordinatesCuda >const * const  mPolymerSystemSorted , 
        MirroredVector< T_Coordinates      >const * const mviPolymerSystemSortedVirtualBox
      )
{
  auto nThreads(256);	
  auto nBlocks(ceilDiv(size,nThreads));
  // std::cout << "Tracker::trackConnections:   mAge= " << mAge  << " size= " << size <<std::endl;
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
  kernelTrackConnections<T_UCoordinateCuda><<<nBlocks,nThreads,0, mStream>>>(
  ID1, 
  ID2, 
  size, 
  offsetA,
  offsetB,
  diNewToi, 
  diToiNew, 
  BondHistoryID1->gpu + counter*nIDs,
  BondHistoryID2->gpu + counter*nIDs,
  mChainID->gpu       + counter*nIDs,
  mPolymerSystemSorted->gpu,
  mviPolymerSystemSortedVirtualBox->gpu,
  mMidToNid->gpu,
  mNidToMid->gpu,
  mNidToNid->gpu,
  mNidToCid->gpu
  );
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );

  age.push_back(mAge);
  increaseCounter();
  if(counter == bufferSize ) dumpReactions();
}
template< typename T_UCoordinateCuda > 
Tracker<T_UCoordinateCuda>::Tracker():
bufferSize      ( 0             ), 
nIDs            ( 0             ), 
mStream         ( cudaStream_t()), 
counter         ( 0             ), 
IDoffset        ( 0             ), 
BaseClass       (               ),
mMidToNid       ( NULL          ),
mNidToMid       ( NULL          ),
mNidToNid       ( NULL          ),
mNidToCid       ( NULL          ),
BondHistoryID1  ( NULL          ),
BondHistoryID2  ( NULL          ),
mChainID        ( NULL          )
{}



template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::init(uint32_t bufferSize_, uint32_t nIDs_, cudaStream_t mStream_,
  T_BoxSize const boxX,
  T_BoxSize const boxY,
  T_BoxSize const boxZ,
  uint32_t chainLength_,
  uint32_t nChains_)
{
  chainLength=chainLength_;
  nChains=nChains_;
  bufferSize=bufferSize_; 
  nIDs=nIDs_; 
  mStream=mStream_;
  std::cout << "Tracker::init: \nnChains=" << nChains <<"\n"
            << "chainLength=" <<chainLength<<"\n"
            << "bufferSize=" <<bufferSize<<"\n"
            << "nIDs=" <<nIDs<<"\n"
            << "mStream=" <<mStream<<"\n";
  BaseClass::setInformationSize(11);
  BaseClass::addComment("MCS Bond/Break ChainID ID1 Position1 ID2 Position2 ");
  std::cout << "Tracker::init: each BondHistory can take " 
            << 2*bufferSize*nIDs << " number of elements with " 
            << 2*bufferSize*nIDs *sizeof(T_Coordinates)/1024.<< " kB \n";
  BondHistoryID1 = new MirroredVector< T_Coordinates >( bufferSize*nIDs, mStream ); //essentially the ids of the first monomer and its positions
  BondHistoryID2 = new MirroredVector< T_Coordinates >( bufferSize*nIDs, mStream ); //essentially the ids of the second monomer and its positions
  mChainID       = new MirroredVector<          ID_t >( bufferSize*nIDs, mStream ); //the chain id between monomer one and two 
  mMidToNid      = new MirroredVector<          ID_t >( chainLength*nChains, mStream );
  mNidToMid      = new MirroredVector<          ID_t >( 2*nChains, mStream );
  mNidToNid      = new MirroredVector<          ID_t >( 2*nChains, mStream );
  mNidToCid      = new MirroredVector<          ID_t >( 2*nChains, mStream );
  for ( size_t i=0; i < nChains; i++){
    mMidToNid->host[i*chainLength]=i*2;
    mMidToNid->host[i*chainLength-1]=i*2+1;
    mNidToMid->host[i*2]=i*chainLength;
    mNidToMid->host[i*2+1]=i*chainLength-1;
    mNidToNid->host[i*2]=i*2+1;
    mNidToNid->host[i*2+1]=i*2;
    mNidToCid->host[i*2]=0;
    mNidToCid->host[i*2+1]=0;
  }
  { decltype( dcBoxX      ) x = boxX     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
  { decltype( dcBoxY      ) x = boxY     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
  { decltype( dcBoxZ      ) x = boxZ     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
}

template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::addCrosslinkConnection(uint32_t chainEndID_, uint32_t crosslinkID_)
{
  mNidToCid->host[mMidToNid->host[chainEndID_]]=crosslinkID_; 
}
template< typename T_UCoordinateCuda > 
void Tracker<T_UCoordinateCuda>::pushToGPU(ID_t const * const miToiNew){
  mMidToNid->push();
  mNidToMid->push();
  mNidToNid->push();

  for( size_t i=0; i <mNidToCid->nElements; i++){
    if (mNidToCid->host[i] > 0 )
      mNidToCid->host[i]= miToiNew[mNidToCid->host[i] ] +1 ;
  }

  mNidToCid->push();
  for (uint32_t j=0 ; j < bufferSize ; j ++ ) {
    for(uint32_t i =0 ; i < nIDs; i ++){
      auto index(i+nIDs*j);
      BondHistoryID1->host[index].w=0;
      BondHistoryID1->host[index].x=0;
      BondHistoryID1->host[index].y=0;
      BondHistoryID1->host[index].z=0;
      BondHistoryID2->host[index].w=0;
      BondHistoryID2->host[index].x=0;
      BondHistoryID2->host[index].y=0;
      BondHistoryID2->host[index].z=0;
      mChainID->host[index]=0;
    }
  }
  BondHistoryID1->push();
  BondHistoryID2->push();
  mChainID -> push();
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
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
  BondHistoryID1->popAsync();
  BondHistoryID2->popAsync();
  mChainID->popAsync();
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
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
        vec.push_back(mChainID->host[index]);

        MonID2 = (MonID2 >> 1) -1;
        if (MonID1 > MonID2 ) {
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
        mChainID->host[index]=0;
      }
    }
  }
  BaseClass::dumpReactions();
  counter=0;
  age.resize(0);
  BondHistoryID1->push();
  BondHistoryID2->push();
  mChainID->push();
  
}
template class Tracker< uint8_t  >;
template class Tracker< uint16_t >;
template class Tracker< uint32_t >;
template class Tracker<  int16_t >;
template class Tracker<  int32_t >;