

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
	using T_BoxSize          = uint64_t;
	//type for coordinate for the host 
	using T_Coordinate       = int32_t; // int64_t // should be signed!
	//type for vector of coordinates on host 
	using T_Coordinates      = typename CudaVec4< T_Coordinate      >::value_type;
	//type for unsigned coordinates on the device 
	using T_UCoordinatesCuda = typename CudaVec4< T_UCoordinateCuda >::value_type;

	//! constructor which sets some values 
	Tracker();

	void increaseCounter();

	// inline void setAge(uint32_t age_){
	//   age.push_back(age_);
	// }
	void init(uint32_t bufferSize_, uint32_t nIDs_, cudaStream_t mStream_,
	  T_BoxSize const boxX,
  	  T_BoxSize const boxY,
  	  T_BoxSize const boxZ,
	  uint32_t chainLength=0,
	  uint32_t nChains=0);
protected:
	uint32_t bufferSize, nIDs, counter;
	cudaStream_t mStream;
	uint32_t  IDoffset;
	uint32_t nChains, chainLength;
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
				ID_t * const miToiNew,
				int32_t const offsetA,
				int32_t const offsetB,
				uint32_t const mAge,
				MirroredVector< T_UCoordinatesCuda >const * const mPolymerSystemSorted , 
				MirroredVector< T_Coordinates      >const * const mviPolymerSystemSortedVirtualBox);
	//! setter function for the buffer size
	void setBufferSize(uint32_t bufferSize_){bufferSize=bufferSize_;}

	//! dumps the data of the connection process into a file 
	void dumpReactions(); 
	inline void setIDOffset(uint32_t IDoffset_){ IDoffset=IDoffset_; }
	//!
	void addCrosslinkConnection(uint32_t chainEndID_, uint32_t crosslinkID_);
	void pushToGPU(ID_t const * const miToiNew );
private:
  	std::vector<uint32_t> age;
	/** monomer id of chain end to reduced monomer chain end id 
	 * 0-1-2-3-4-5-6-7-8 monomeric id
	 * |               | 
	 * 0---------------1 reduced id
	**/
	MirroredVector<ID_t> * mMidToNid; 
	//! reduced monomer chain id to the monomer id (initial id)
	MirroredVector<ID_t> * mNidToMid; 
	//! monomer id of one chain end to the other chain end id 
	MirroredVector<ID_t> * mNidToNid; 
	//! id of chain end to the crosslinkID (gloabl ID), start the crosslink ID at 1 to distinguish the case where no connection is present 
	MirroredVector<ID_t> * mNidToCid;
	//! ID  (and position ) of the first monomer 
	MirroredVector<T_Coordinates> * BondHistoryID1;
	//! ID  (and position ) of the second monomer  (of the connected crosslink)
	MirroredVector<T_Coordinates> * BondHistoryID2;
	//! chainID of the connection from the first and the second monomer
	MirroredVector<ID_t> * mChainID;
};

template class Tracker< uint8_t  >;
template class Tracker< uint16_t >;
template class Tracker< uint32_t >;
template class Tracker<  int16_t >;
template class Tracker<  int32_t >;
