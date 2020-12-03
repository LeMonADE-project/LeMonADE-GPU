

#pragma once 
#include <LeMonADE/utility/TrackConnection.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <cuda_runtime_api.h>    

template< typename T_UCoordinateCuda > 
class Tracker:public TrackLinks<int32_t>
{

  
public:
  typedef  TrackLinks<int32_t> BaseClass; 
  typedef  uint32_t ID_t;
	//type for coordinate for the host 
	using T_Coordinate       = int32_t; // int64_t // should be signed!
	//type for vector of coordinates on host 
	using T_Coordinates      = typename CudaVec4< T_Coordinate      >::value_type;
	//type for unsigned coordinates on the device 
	using T_UCoordinatesCuda = typename CudaVec4< T_UCoordinateCuda >::value_type;

Tracker():bufferSize(0), nIDs(0), mStream(cudaStream_t()), counter(0), IDoffset(0), BaseClass(){}

// __device__ void addBond(uint32_t ID1, uint32_t ID2);
// __device__ void removeBond(uint32_t ID1, uint32_t ID2);
void increaseCounter();

// inline void setAge(uint32_t age_){
//   age.push_back(age_);
// }
void init(uint32_t bufferSize_, uint32_t nIDs_, cudaStream_t mStream_);
protected:
uint32_t bufferSize, nIDs, counter;
cudaStream_t mStream;
uint32_t  IDoffset;
public:
  
void trackBreaks( ID_t * const ID1     ,
		  ID_t * const ID2     ,
		  size_t   const size    ,     
		  ID_t * const miNewToi,
		  int32_t const offsetA,
		  int32_t const offsetB,
		  uint32_t const mAge,
		  MirroredVector< T_UCoordinatesCuda >const * const  mPolymerSystemSorted , 
		  MirroredVector< T_Coordinates      >const * const mviPolymerSystemSortedVirtualBox );
void trackConnections( ID_t * const ID1     ,
		       ID_t * const ID2     ,
		       size_t   const size    ,     
		       ID_t * const miNewToi,
		       int32_t const offsetA,
		       int32_t const offsetB,
		       uint32_t const mAge,
			   MirroredVector< T_UCoordinatesCuda >const * const mPolymerSystemSorted , 
        	   MirroredVector< T_Coordinates      >const * const mviPolymerSystemSortedVirtualBox );
//! setter function for the buffer size
void setBufferSize(uint32_t bufferSize_){bufferSize=bufferSize_;}
MirroredVector<ID_t> * BondHistoryID1;
MirroredVector<ID_t> * BondHistoryID2;
//! dumps the data of the connection process into a file 
void dumpReactions(); 
inline void setIDOffset(uint32_t IDoffset_){ IDoffset=IDoffset_; }
private:
  std::vector<uint32_t> age;
};

template class Tracker< uint8_t  >;
template class Tracker< uint16_t >;
template class Tracker< uint32_t >;
template class Tracker<  int16_t >;
template class Tracker<  int32_t >;
