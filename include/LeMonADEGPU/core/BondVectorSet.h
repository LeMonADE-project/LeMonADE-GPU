#ifndef LEMONADEGPU_CORE_BONDVECTORSET_H_
#define LEMONADEGPU_CORE_BONDVECTORSET_H_
#include <LeMonADEGPU/utility/cudacommon.hpp> //for  CUDA_ERROR
/* 512=8^3 for a range of bonds per direction of [-4,3] */
__device__ __constant__ bool dpForbiddenBonds[512]; //false-allowed; true-forbidden
/**
 * @brief I declared this function globally otherwise i decrease the performance by around 200 % 
 * 	  unfortunatly I have no idea why?!
 */
__device__ __host__ int16_t inline  linearizeBondVectorIndex
(
    int16_t const x,
    int16_t const y,
    int16_t const z
) {
      /* Just like for normal integers we clip the range to go more down than up
  * i.e. [-127 ,128] or in this case [-4,3]
  * +4 maps to the same location as -4 but is needed or else forbidden
  * bonds couldn't be detected. Larger bonds are not possible, because
  * monomers only move by 1 per step */
  //assert( -4 <= x && x <= 4 );
  //assert( -4 <= y && y <= 4 );
  //assert( -4 <= z && z <= 4 );
  return   ( x & int16_t(7) /* 0b111 */ ) +
	( ( y & int16_t(7) /* 0b111 */ ) << 3 ) +
	( ( z & int16_t(7) /* 0b111 */ ) << 6 );
}

/**
 * @brief simple class to decide about acceptable bonds
 * @details in the standard construction of the program this operator returns true for a bond which is NOT valid 
 * and the values for dx,dy,dz range from [-4:3]. Other values cannot be handled and thus cann cause serious errors!
 * @todo add a selectiveLogger to the class which performs a check for the range if the "check" case is activated 
 */

class BondVectorSet
{
public:


  inline void addBondVector( int dx, int dy, int dz, bool bondForbidden ){mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;};
  
  __device__ __host__ inline   bool operator()(int dx, int dy, int dz) const 
  {
    //if check activated-> check if dx,dy,dz are in the correct range [-4:3]
    #ifdef __CUDA_ACC__
      return dpForbiddenBonds[linearizeBondVectorIndex(dx,dy,dz)];
    #else
      return mForbiddenBonds[linearizeBondVectorIndex(dx,dy,dz)];
    #endif
  }

  void initBondTable();
private:   
  bool mForbiddenBonds[512];  
  
};



#endif /* LEMONADEGPU_CORE_BONDVECTORSET_H_ */
