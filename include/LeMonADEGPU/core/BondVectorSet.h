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
 * @brief calculates the minimal distances of images for one component 
 * @return int 
 * @param x1 absolute coordinate
 * @param x2 absolute coordinate
 * @param LatticeSize size of the box in the direction of the given coordinates
 */
__device__ __host__  int inline MinImageDistanceComponentForPowerOfTwo(const int x, const uint32_t latticeSize )
{
  //this is only valid for absolute coordinates
  uint32_t latticeSizeM1(latticeSize-1);
  return ( ((x&latticeSizeM1) < (latticeSize/2)) ? (x & latticeSizeM1) :  -(-x & latticeSizeM1));
} 

/**
 * @brief simple class to decide about acceptable bonds
 * @details in the standard construction of the program this operator returns true for a bond which is NOT valid 
 * and the values for dx,dy,dz range from [-4:3]. Other values cannot be handled and thus cann cause serious errors!
 */

class BondVectorSet
{
public:


  inline void addBondVector( int dx, int dy, int dz, bool bondForbidden ){mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;};
  
  __device__ __host__ inline   bool operator()(int dx, int dy, int dz) const 
  {
    #ifdef __CUDA_ACC__
      return dpForbiddenBonds[linearizeBondVectorIndex(dx,dy,dz)];
    #else
      return mForbiddenBonds[linearizeBondVectorIndex(dx,dy,dz)];
    #endif
  }
  
  __device__ __host__ inline   bool checkMinImagePow2Lattice(int dx, int dy, int dz, int BoxX, int BoxY, int BoxZ) const 
  {
    dx=MinImageDistanceComponentForPowerOfTwo(dx,BoxX);
    dy=MinImageDistanceComponentForPowerOfTwo(dy,BoxY);
    dz=MinImageDistanceComponentForPowerOfTwo(dz,BoxZ);
    if ( dx*dx > 9 ) return true; 
    if ( dy*dy > 9 ) return true;
    if ( dz*dz > 9 ) return true;
    #ifdef __CUDA_ACC__
      return dpForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz)];
    #else
      return mForbiddenBonds[  linearizeBondVectorIndex(dx,dy,dz)];
    #endif
  
   
  }
  void initBondTable();
private:   
  bool mForbiddenBonds[512];  
};



#endif /* LEMONADEGPU_CORE_BONDVECTORSET_H_ */
