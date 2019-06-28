#ifndef LEMONADEGPU_CORE_BONDVECTORSET_H_
#endif LEMONADEGPU_CORE_BONDVECTORSET_H_
#include <LeMonADEGPU/utility/cudacommon.hpp> //for  CUDA_ERROR
/* 512=8^3 for a range of bonds per direction of [-4,3] */
__device__ __constant__ bool dpForbiddenBonds[512]; //false-allowed; true-forbidden
class BondVectorSet
{
public:

  __device__ __host__ inline int16_t linearizeBondVectorIndex
  (
      int16_t const x,
      int16_t const y,
      int16_t const z
  )
  {
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
  bool mForbiddenBonds[512];
  inline void addBondVector( int dx, int dy, int dz, bool bondForbidden ){mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;};
  __device__ __host__ inline   bool isForbiddenBond(int dx, int dy, int dz)
  {
    #ifdef __CUDA_ACC__
      return dForbiddenBonds[linearizeBondVectorIndex(dx,dy,dz)];
    #else
      return mForbiddenBonds[linearizeBondVectorIndex(dx,dy,dz)];
    #endif
  }
  
  inline void initBondTable(){
    /* create the BondTable and copy it to constant memory */
    bool * tmpForbiddenBonds = (bool*) malloc( sizeof( bool ) * 512 );
    unsigned nAllowedBonds = 0;
    for ( int i = 0; i < 512; ++i )
	if ( ! ( tmpForbiddenBonds[i] = mForbiddenBonds[i] ) )
	    ++nAllowedBonds;
    /* Why does it matter? Shouldn't it work with arbitrary bond sets ??? */
    if ( nAllowedBonds != 108 )
    {
	std::stringstream msg;
	msg << "[" << __FILENAME__ << "::initializeBondTable] "
	    << "Wrong bond-set! Expected 108 allowed bonds, but got " << nAllowedBonds << "\n";
	mLog( "Error" ) << msg.str();
	throw std::runtime_error( msg.str() );
    }
    CUDA_ERROR( cudaMemcpyToSymbol( dpForbiddenBonds, tmpForbiddenBonds, sizeof(bool)*512 ) );
      free( tmpForbiddenBonds );
  }
};

#endif LEMONADEGPU_CORE_BONDVECTORSET_H_