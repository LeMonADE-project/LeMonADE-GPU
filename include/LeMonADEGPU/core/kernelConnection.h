

#ifndef LEMONADEGPU_CORE_KERNELCONNECTION_H
#define LEMONADEGPU_CORE_KERNELCONNECTION_H
#include <stdint.h>
class Connection
{
public: 
  Connection(uint32_t nReactiveMonomers_):arraySize((uint32_t)(nReactiveMonomers_/4)+1), nThreads(1024){};
  void setNThreads(uint32_t nThreads_ ) {nThreads=nThreads_;}
  uint32_t getNThreads() const {return nThreads;}
//   void setArraySize(uint32_t arraySize_){arraySize=arraySize_;}
  uint32_t getArraySize() const {return arraySize;}
  
  void resetMultipleIDs( uint32_t * crosslinkId, uint32_t * chainID );
  
private: 
  uint32_t nThreads;
  const uint32_t arraySize;
};
#endif /*LEMONADEGPU_CORE_KERNELCONNECTION_H*/