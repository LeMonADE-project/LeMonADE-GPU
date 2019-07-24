

#pragma once 
#include <LeMonADE/utility/TrackConnection.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <cuda_runtime_api.h>    


class Tracker:public TrackLinks<uint32_t>
{

  
public:
  typedef  TrackLinks<uint32_t> BaseClass; 
  typedef  uint32_t ID_t;
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
		  ID_t   const size    ,     
		  ID_t * const miNewToi,
		  int32_t const offsetA,
		  int32_t const offsetB,
		  uint32_t const mAge );
void trackConnections( ID_t * const ID1     ,
		       ID_t * const ID2     ,
		       ID_t   const size    ,     
		       ID_t * const miNewToi,
		       int32_t const offsetA,
		       int32_t const offsetB,
		       uint32_t const mAge );
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

