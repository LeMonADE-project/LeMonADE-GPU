

#ifndef LEMONADEGPU_CORE_KERNELCONNECTION_H
#define LEMONADEGPU_CORE_KERNELCONNECTION_H
#include <stdint.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
class Connection
{
public: 
  Connection():arraySize(0), nThreads(1024){};
  Connection(uint32_t nReactiveMonomers_):arraySize(4*ceil((nReactiveMonomers_+1)*1.0/4.)), nThreads(1024), dBuffer(NULL){init();};
  
//   ~Connection(){clean();}
  ~Connection();
  
  void setNThreads(uint32_t nThreads_ ) {nThreads=nThreads_;}
  uint32_t getNThreads() const {return nThreads;}
//   void setArraySize(uint32_t arraySize_){arraySize=arraySize_;}
  uint32_t getArraySize() const {return arraySize;}
  void init();
  void clean();
  inline void setArraySize(uint32_t nReactiveMonomers_ ){arraySize=(4*ceil((nReactiveMonomers_+1)*1.0/4.));}
  /**
   * @brief resets the mutliple occurent ides to zero.
   * @details  we have to differentiate between two cases because the shared memory 
   * is restricted to 48 kB. For each entry in the array we need 2 times 4 byte(=uint32_t).
   * 48kb/8 ~ 6000. For higher number of cross links we need to use the global memory 
   * @todo optimize this function:
   * 	- reduce array size by remoiving (leading) zeros  after resorting 
   * 	- use again shared memory...	
   */
  void resetMultipleIDs( uint32_t * crosslinkId, uint32_t * chainID, cudaStream_t mStream );
  
private: 
  uint32_t nThreads;
  uint32_t arraySize;
  uint32_t * dBuffer;
};
#endif /*LEMONADEGPU_CORE_KERNELCONNECTION_H*/