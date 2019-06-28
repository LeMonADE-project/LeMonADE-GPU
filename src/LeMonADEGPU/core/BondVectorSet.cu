#include <LeMonADEGPU/core/BondVectorSet.h>

void BondVectorSet::initBondTable(){
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
      throw std::runtime_error( msg.str() );
  }
  CUDA_ERROR( cudaMemcpyToSymbol( dpForbiddenBonds, tmpForbiddenBonds, sizeof(bool)*512 ) );
    free( tmpForbiddenBonds );
}