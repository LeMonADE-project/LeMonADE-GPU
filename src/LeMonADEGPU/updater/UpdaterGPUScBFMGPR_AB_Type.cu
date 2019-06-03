/*
 * UpdaterGPUScBFMGPR_AB_Type.cpp
 *
 *  Created on: 27.07.2017
 *     Authors: Ron Dockhorn, Maximilian Knespel and Toni Mueller
 */


#include <LeMonADEGPU/updater/UpdaterGPUScBFMGPR_AB_Type.h>

//#define USE_THRUST_FILL
//The USE_BIT_PACKING_TMP_LATTICE does not work within combination of the standard linearization and noncubic lattices!
// #define USE_BIT_PACKING_TMP_LATTICE 
#define USE_BIT_PACKING_LATTICE
// #define USE_ZCURVE_FOR_LATTICE
#define USE_ZCURVE_FOR_NONCUBIC_LATTICE
// #define NOMAGIC
//#define AUTO_CONFIGURE_BEST_SETTINGS_FOR_PSCBFM_ALGORITHM


#include <algorithm>                        // fill, sort
#include <chrono>                           // std::chrono::high_resolution_clock
#include <cstdio>                           // printf
#include <cstdlib>                          // exit
#include <cstring>                          // memset
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include <cuda_profiler_api.h>              // cudaProfilerStop
#ifdef USE_THRUST_FILL
#   include <thrust/system/cuda/execution_policy.h>
#   include <thrust/fill.h>
#endif

#include <extern/Fundamental/BitsCompileTime.hpp>

#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/graphColoring.tpp>
#include <extern/Fundamental/BitsCompileTime.hpp>
#include <LeMonADEGPU/core/rngs/Saru.h>


#define DEBUG_UPDATERGPUSCBFM_AB_TYPE 100
#if defined( USE_BIT_PACKING_TMP_LATTICE ) || defined( USE_BIT_PACKING_LATTICE )
#   define USE_BIT_PACKING
#endif

#include <nvfunctional>



/* 512=8^3 for a range of bonds per direction of [-4,3] */
__device__ __constant__ bool dpForbiddenBonds[512]; //false-allowed; true-forbidden

/**
 * These will be initialized to:
 *   DXTable_d = { -1,1,0,0,0,0 }
 *   DYTable_d = { 0,0,-1,1,0,0 }
 *   DZTable_d = { 0,0,0,0,-1,1 }
 * I.e. a table of three random directional 3D vectors \vec{dr} = (dx,dy,dz)
 */
__device__ __constant__ uint32_t DXTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DYTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DZTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
// __device__ __constant__ uint32_t DXTable_d[18]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
// __device__ __constant__ uint32_t DYTable_d[18]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
// __device__ __constant__ uint32_t DZTable_d[18]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
/**
 * If intCUDA is different from uint32_t, then this second table prevents
 * expensive type conversions, but both tables are still needed, the
 * uint32_t version, because the calculation of the linear index will result
 * in uint32_t anyway and the intCUDA version for solely updating the
 * position information
 */
__device__ __constant__ intCUDA DXTableIntCUDA_d[6];
__device__ __constant__ intCUDA DYTableIntCUDA_d[6];
__device__ __constant__ intCUDA DZTableIntCUDA_d[6];
// __device__ __constant__ intCUDA DXTableIntCUDA_d[18];
// __device__ __constant__ intCUDA DYTableIntCUDA_d[18];
// __device__ __constant__ intCUDA DZTableIntCUDA_d[18];

/* will this really bring performance improvement? At least constant cache
 * might be as fast as register access when all threads in a warp access the
 * the same constant */
__device__ __constant__ uint32_t dcBoxXM1     ;  // mLattice size in X-1
__device__ __constant__ uint32_t dcBoxYM1     ;  // mLattice size in Y-1
__device__ __constant__ uint32_t dcBoxZM1     ;  // mLattice size in Z-1
__device__ __constant__ uint32_t dcBoxXLog2   ;  // mLattice shift in X
__device__ __constant__ uint32_t dcBoxXYLog2  ;  // mLattice shift in X*Y
/* this will be used for performance improvement in the FeatureShearForce
 */
__device__ __constant__ uint32_t dcBoxZ_HalfP1;  // mLattice size in Z/2+1
__device__ __constant__ uint32_t dcBoxZ_HalfM3;  // mLattice size in Z/2-3
__device__ __constant__ uint32_t dcBoxZM2     ;  // mLattice size in Z-2
__device__ __constant__ double dLookUpShearForce[2];
/* used for the movement of the bonds 
 * here we make use of the alternating coloring of the chain monomers and that 
 * there should be the same number of zero and one colored monomers */
__device__ __constant__ uint32_t dMatrixOffsetElementsZero;
__device__ __constant__ uint32_t dMatrixOffsetElementsOne ;
__device__ __constant__ uint32_t dMatrixOffsetElementsTwo ;

__device__ __constant__ uint32_t diSubGroupOffsetZero;
__device__ __constant__ uint32_t diSubGroupOffsetOne ;
__device__ __constant__ uint32_t diSubGroupOffsetTwo ;

__device__ __constant__ uint32_t dChainPitch       ;  // pitch for the new indecies for species 0 and 1 (pitch is the same, because they are assinged in alternating form )

using T_Flags = UpdaterGPUScBFMGPR_AB_Type::T_Flags;
/*
 *default check is for periodicity in all directions. 
 */
struct BoxCheck
{
  enum mode {  periodic111, periodic000, 
	       periodic100, periodic010, periodic001, 
	       periodic110, periodic011, periodic101, 
	       periodic=0, nonperiodic=1 };
	       
  int myperiodicmode;
  bool pX;
  bool pY;
  bool pZ;

  BoxCheck():myperiodicmode(0) {}
  BoxCheck(int myperiodicmode_):myperiodicmode(myperiodicmode_) {}
  BoxCheck( bool pX_,  bool pY_,  bool pZ_ ):pX(pX_),pY(pY_),pZ(pZ_)
  {
    if      (   pX &&   pY &&   pZ )  //111 periodic 
	myperiodicmode=0;
    else if ( ! pX && ! pY && ! pZ )  //000 nonperiodic
	myperiodicmode=1;
    else if (   pX && ! pY && ! pZ )  //100 mixed 
	myperiodicmode=2;
    else if ( ! pX &&   pY && ! pZ )  //010
	myperiodicmode=3;
    else if ( ! pX && ! pY &&   pZ )  //001
	myperiodicmode=4;
    else if (   pX &&   pY && ! pZ )  //110
	myperiodicmode=5;
    else if ( ! pX &&   pY &&   pZ )  //011
	myperiodicmode=6;
    else if (   pX && ! pY &&   pZ )  //101
	myperiodicmode=7;
  }
  
  __device__ bool operator()(const int32_t x,const int32_t y, const int32_t z)
  {
    switch(myperiodicmode)
    { 
	case periodic111: return  true                                ;
	case periodic000: return (uint32_t(0) <= x && x < dcBoxXM1 &&
				  uint32_t(0) <= y && y < dcBoxYM1 &&
				  uint32_t(0) <= z && z < dcBoxZM1 )  ;  
	case periodic100: return (uint32_t(0) <= y && y < dcBoxYM1 &&
				  uint32_t(0) <= z && z < dcBoxZM1 )  ;
	case periodic010: return (uint32_t(0) <= x && x < dcBoxXM1 &&
				  uint32_t(0) <= z && z < dcBoxZM1 )  ;
	case periodic001: return (uint32_t(0) <= x && x < dcBoxXM1 &&
				  uint32_t(0) <= y && y < dcBoxYM1 )  ;
	case periodic110: return (uint32_t(0) <= z && z < dcBoxZM1 )  ;
	case periodic011: return (uint32_t(0) <= x && x < dcBoxXM1 )  ;
	case periodic101: return (uint32_t(0) <= y && y < dcBoxYM1 )  ;
	default         : return false                                ; //maybe throw an error?! 
    };
  }
};

// template <typename MoveType>
class Move
{
public:    
    Move(int StandardOn_):StandardOn(StandardOn_) {}
    template <class T> inline __device__ T getDirectionID( T rn ) {
      switch (StandardOn)
      { case Standard: return rn % 6 ;
	case Diagonal: return rn % 18;  
	case Basic   : return 0      ;
      };
    }
    template <class T> inline __device__ T setDirectionID( T properties ) {
       switch (StandardOn)
      { case Standard: return ( properties >> T_Flags(2) ) & T_Flags(7) ;
	case Diagonal: return ( properties >> T_Flags(2) ) & T_Flags(31); 
	case Basic   : return 0      ;
      };
      
    }    
private: 
      enum mode { Standard, Diagonal, Basic };
      int StandardOn;
//   MoveType move;
};
struct StandardScMove
{ 
  template <class T>
  inline __device__ T getDirectionID( T rn ) {return rn % 6;}
  template <class T>
  inline __device__ T setDirectionID( T properties ) { return ( properties >> T_Flags(2) ) & T_Flags(7);}
};
struct DiagonalScMove
{
  template <class T>
  inline __device__ T getDirectionID( T rn         ) { return rn % 18;}
  template <class T>
  inline __device__ T setDirectionID( T properties ) { return ( properties >> T_Flags(2) ) & T_Flags(31);}
};

template <typename Criterion >
class LocalMetropolis
{
public: 
  LocalMetropolis():LocalMetropolisOn(false),criterion() {}
  LocalMetropolis(bool LocalMetropolisOn_):LocalMetropolisOn(LocalMetropolisOn_) {}
  
  inline void setLocalMetropolisOn(bool LocalMetropolisOn_){LocalMetropolisOn=LocalMetropolisOn_;}

  inline __host__ __device__ double operator() () {return 0; }

  template <class T>
  inline __host__ __device__ double operator() (T input)
  {
    switch (LocalMetropolisOn)
    {
      case 0: return 1;
      case 1: return criterion(input);
    }
  }
  template <class T1, class T2>
  inline __host__ __device__ double operator() (T1 input1, T2 input2)
  {
    switch (LocalMetropolisOn)
    {
      case 0: return 1;
      case 1: return criterion(input1, input2);
    };
  }
private:
  bool LocalMetropolisOn;  
  Criterion criterion;
};

struct ShearForce
{
// doing the Metropolis-criterion for movement
// dU = gamma*dr
// slice z=0 && z=1 -> right gamma=(1,0,0) -> dU = dx
// slice z is [Box/2-3;Box/2+1] -> left gamma=(-1,0,0) -> dU = -dx
// slice z=Box-1 && z=Box-2 -> right gamma=(1,0,0) -> dU = dx
// otherwise -> always allowed  
//   ShearForce(){}
  template <class T1, class T2>
  inline __device__ double  operator() (T1 ZPosition, T2 dx)
  {
    uint32_t id(0);
    if      ( ZPosition < 2                                              ) id += 1;
    else if ((ZPosition <= dcBoxZ_HalfP1) && (ZPosition > dcBoxZ_HalfM3) ) id += 2;
    else if ( ZPosition >= dcBoxZM2                                      ) id += 3;
    else return 1;
    //  Metropolis-criterion is only necessary
    if ( dx < 0 )
      id+=3;
    return dLookUpShearForce[id | 2];

  }
};

#define USE_BIT_PACKING
#ifdef USE_BIT_PACKING
    template< typename T > __device__ __host__ inline
    T bitPackedGet( T const * const & p, uint32_t const & i )
    {
        /**
         * >> 3, because 3 bits = 2^3=8 numbers are used for sub-byte indexing,
         * i.e. we divide the index i by 8 which is equal to the space we save
         * by bitpacking.
         * & 7, because 7 = 0b111, i.e. we are only interested in the last 3
         * bits specifying which subbyte element we want
         */
        return ( p[ i >> 3 ] >> ( i & T(7) ) ) & T(1);
    }

    template< typename T > __device__ inline
    T bitPackedTextureGet( cudaTextureObject_t const & p, uint32_t const & i )
    {
        return ( tex1Dfetch<T>( p, i >> 3 ) >> ( i & T(7) ) ) & T(1);
    }

    /**
     * Because the smalles atomic is for int (4x uint8_t) we need to
     * cast the array to that and then do a bitpacking for the whole 32 bits
     * instead of 8 bits
     * I.e. we need to address 32 subbits, i.e. >>3 becomes >>5
     * and &7 becomes &31 = 0b11111 = 0x1F
     * __host__ __device__ function with differing code
     * @see https://codeyarns.com/2011/03/14/cuda-common-function-for-both-host-and-device-code/
     */
    template< typename T > __device__ __host__ inline
    void bitPackedSet( T * const __restrict__ p, uint32_t const & i )
    {
        static_assert( sizeof(int) == 4, "" );
        #ifdef __CUDA_ARCH__
            atomicOr ( (int*) p + ( i >> 5 ),    T(1) << ( i & T( 0x1F ) )   );
        #else
            p[ i >> 3 ] |= T(1) << ( i & T(7) );
        #endif
    }

    template< typename T > __device__ __host__ inline
    void bitPackedUnset( T * const __restrict__ p, uint32_t const & i )
    {
        #ifdef __CUDA_ARCH__
            atomicAnd( (int*) p + ( i >> 5 ), ~( T(1) << ( i & T( 0x1F ) ) ) );
        #else
            p[ i >> 3 ] &= ~( T(1) << ( i & T(7) ) );
        #endif
    }
#else
    template< typename T > __device__ __host__ inline
    T bitPackedGet( T const * const & p, uint32_t const & i ){ return p[i]; }
    
    template< typename T > __device__ inline
    T bitPackedTextureGet( cudaTextureObject_t const & p, uint32_t const & i ) {return tex1Dfetch<T>(p,i); }
    
    template< typename T > __device__ __host__ inline
    void bitPackedSet  ( T * const __restrict__ p, uint32_t const & i ){ p[i] = 1; }
    
    template< typename T > __device__ __host__ inline
    void bitPackedUnset( T * const __restrict__ p, uint32_t const & i ){ p[i] = 0; }
#endif

__device__ inline uint32_t linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #if defined ( USE_ZCURVE_FOR_LATTICE )
        return   diluteBits< uint32_t, 2 >( ix & dcBoxXM1 )        +
               ( diluteBits< uint32_t, 2 >( iy & dcBoxYM1 ) << 1 ) +
               ( diluteBits< uint32_t, 2 >( iz & dcBoxZM1 ) << 2 );
    #else
        return   ( ix & dcBoxXM1 ) +
               ( ( iy & dcBoxYM1 ) << dcBoxXLog2  ) +
               ( ( iz & dcBoxZM1 ) << dcBoxXYLog2 );
    #endif
}
uint32_t UpdaterGPUScBFMGPR_AB_Type::linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #if defined ( USE_ZCURVE_FOR_LATTICE )
        return   diluteBits< uint32_t, 2 >( ix & mBoxXM1 )        +
               ( diluteBits< uint32_t, 2 >( iy & mBoxYM1 ) << 1 ) +
               ( diluteBits< uint32_t, 2 >( iz & mBoxZM1 ) << 2 );
//     #elif defined (USE_ZCURVE_FOR_NONCUBIC_LATTICE) 
    #else
        return   ( ix & mBoxXM1 ) +
               ( ( iy & mBoxYM1 ) << mBoxXLog2  ) +
               ( ( iz & mBoxZM1 ) << mBoxXYLog2 );
    #endif
}

//   Why do not declare a host-device function so that only one functions exits and no confusion are created !!!???
//   __device__ inline uint32_t linearizeBoxVectorIndex
// (
//     uint32_t  ix,
//     uint32_t  iy,
//     uint32_t  iz
// )
//  {
// 	ix &= dcBoxXM1;
// 	iy &= dcBoxYM1;
// 	iz &= dcBoxZM1;
// // 	uint32_t const nBitsZCurve = std::min( std::min( log2( dcBoxX ), log2(
// // 	dcBoxY ) ), log2( dcBoxZ ) );
// // 	uint32_t const nBitsZCurve = fmin( fmin( log2( dcBoxX ), log2(
// // 	dcBoxY ) ), log2( dcBoxZ ) );
// 	uint32_t const nBitsZCurve =4;
// 	auto const iZCurve =
// 	  diluteBits< T_Id, 2 >( ix & ( ( 1 << nBitsZCurve ) - 1 ) )        +
// 	( diluteBits< T_Id, 2 >( iy & ( ( 1 << nBitsZCurve ) - 1 ) ) << 1 ) +
// 	( diluteBits< T_Id, 2 >( iz & ( ( 1 << nBitsZCurve ) - 1 ) ) << 2 );
// 	auto const iNormal =
// 	    ix >> nBitsZCurve                    + 
// 	( ( iy >> nBitsZCurve ) << dcBoxXLog2  ) +
// 	( ( iz >> nBitsZCurve ) << dcBoxXYLog2 );
// 	
// 	return ( iNormal << ( 3*nBitsZCurve ) ) + iZCurve;
//  }

/**
 * Checks the 3x3 grid one in front of the new position in the direction of the
 * move given by axis.
 *
 * @verbatim
 *           ____________
 *         .'  .'  .'  .'|
 *        +---+---+---+  +     y
 *        | 6 | 7 | 8 |.'|     ^ z
 *        +---+---+---+  +     |/
 *        | 3/| 4/| 5 |.'|     +--> x
 *        +-/-+-/-+---+  +
 *   0 -> |+---+1/| 2 |.'  ^          ^
 *        /|/-/|/-+---+   /          / axis direction +z (axis = 0b101)
 *       / +-/-+         /  2 (*dz) /                              ++|
 *      +---+ /         /                                         /  +/-
 *      |/X |/         L                                        xyz
 *      +---+  <- X ... current position of the monomer
 * @endverbatim
 *
 * @param[in] axis +-x, +-y, +-z in that order from 0 to 5, or put in another
 *                 equivalent way: the lowest bit specifies +(1) or -(0) and the
 *                 Bit 2 and 1 specify the axis: 0b00=x, 0b01=y, 0b10=z
 * @return Returns true if any of that is occupied, i.e. if there
 *         would be a problem with the excluded volume condition.
 */
__device__ inline bool checkFront
(
    cudaTextureObject_t const & texLattice,
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    intCUDA             const & axis 
)
{
    #if defined( USE_ZCURVE_FOR_LATTICE )
        auto const x0Abs  = diluteBits< uint32_t, 2 >( ( x0               ) & dcBoxXM1 );
        auto const x0PDX  = diluteBits< uint32_t, 2 >( ( x0 + uint32_t(1) ) & dcBoxXM1 );
        auto const x0MDX  = diluteBits< uint32_t, 2 >( ( x0 - uint32_t(1) ) & dcBoxXM1 );
        auto const y0Abs  = diluteBits< uint32_t, 2 >( ( y0               ) & dcBoxYM1 ) << 1;
        auto const y0PDY  = diluteBits< uint32_t, 2 >( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << 1;
        auto const y0MDY  = diluteBits< uint32_t, 2 >( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << 1;
        auto const z0Abs  = diluteBits< uint32_t, 2 >( ( z0               ) & dcBoxZM1 ) << 2;
        auto const z0PDZ  = diluteBits< uint32_t, 2 >( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << 2;
        auto const z0MDZ  = diluteBits< uint32_t, 2 >( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << 2;
    #else
        auto const x0Abs  =   ( x0               ) & dcBoxXM1;
        auto const x0PDX  =   ( x0 + uint32_t(1) ) & dcBoxXM1;
        auto const x0MDX  =   ( x0 - uint32_t(1) ) & dcBoxXM1;
        auto const y0Abs  = ( ( y0               ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const y0PDY  = ( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const y0MDY  = ( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const z0Abs  = ( ( z0               ) & dcBoxZM1 ) << dcBoxXYLog2;
        auto const z0PDZ  = ( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;
        auto const z0MDZ  = ( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;
    #endif

    auto const dx = DXTable_d[ axis ];   // 2*axis-1
    auto const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    auto const dz = DZTable_d[ axis ];   // 2*(axis&1)-1

    uint32_t is[15];

    #if defined( USE_ZCURVE_FOR_LATTICE )
        switch ( axis >> intCUDA(1) )
        {
            case 0: is[7] = diluteBits< uint32_t, 2 >(( x0 + decltype(dx)(2) * dx ) & dcBoxXM1)     ; break;
            case 1: is[7] = diluteBits< uint32_t, 2 >(( y0 + decltype(dy)(2) * dy ) & dcBoxYM1) << 1; break;
            case 2: is[7] = diluteBits< uint32_t, 2 >(( z0 + decltype(dz)(2) * dz ) & dcBoxZM1) << 2; break;
	    case 3:
	    case 4: is[7]  = diluteBits< uint32_t, 2 >(( x0 + decltype(dx)(2) * dx ) & dcBoxXM1)     ; 
	            is[10] = diluteBits< uint32_t, 2 >(( y0 + decltype(dy)(2) * dy ) & dcBoxYM1) << 1; 
		    break;
	    case 5:
	    case 6: is[7]  = diluteBits< uint32_t, 2 >(( y0 + decltype(dy)(2) * dy ) & dcBoxYM1) << 1; 
	            is[10] = diluteBits< uint32_t, 2 >(( z0 + decltype(dz)(2) * dz ) & dcBoxZM1) << 2; 
		    break;
	    case 7:
	    case 8: is[7]  = diluteBits< uint32_t, 2 >(( z0 + decltype(dz)(2) * dz ) & dcBoxZM1) << 2; 
		    is[10] = diluteBits< uint32_t, 2 >(( x0 + decltype(dx)(2) * dx ) & dcBoxXM1)     ; 
		    break;
        }
    #else
        switch ( axis >> intCUDA(1) )
        {
            case 0: is[7]  =   ( x0 + decltype(dx)(2) * dx ) & dcBoxXM1                 ; break;
            case 1: is[7]  = ( ( y0 + decltype(dy)(2) * dy ) & dcBoxYM1 ) << dcBoxXLog2 ; break;
            case 2: is[7]  = ( ( z0 + decltype(dz)(2) * dz ) & dcBoxZM1 ) << dcBoxXYLog2; break;
	    case 3:
	    case 4: is[7]  = diluteBits< uint32_t, 2 >(( x0 + decltype(dx)(2) * dx ) & dcBoxXM1)              ; 
		    is[10] = diluteBits< uint32_t, 2 >(( y0 + decltype(dy)(2) * dy ) & dcBoxYM1) << dcBoxXLog2; 
		    break;
	    case 5:
	    case 6: is[7]  = diluteBits< uint32_t, 2 >(( y0 + decltype(dy)(2) * dy ) & dcBoxYM1) << dcBoxXLog2 ; 
		    is[10] = diluteBits< uint32_t, 2 >(( z0 + decltype(dz)(2) * dz ) & dcBoxZM1) << dcBoxXYLog2; 
		    break;
	    case 7:
	    case 8: is[7]  = diluteBits< uint32_t, 2 >(( z0 + decltype(dz)(2) * dz ) & dcBoxZM1) << dcBoxXYLog2; 
		    is[10] = diluteBits< uint32_t, 2 >(( x0 + decltype(dx)(2) * dx ) & dcBoxXM1)      	       ; 
		    break;
	}
    #endif
    switch ( axis >> intCUDA(1) )
    {
        case 0: //-+x
            is[2]  = is[7] | z0Abs; is[5]  = is[7] | z0MDZ; is[8]  = is[7] | z0PDZ;
            is[0]  = is[2] | y0MDY; is[1]  = is[2] | y0Abs; is[2] |=         y0PDY;
            is[3]  = is[5] | y0MDY; is[4]  = is[5] | y0Abs; is[5] |=         y0PDY;
            is[6]  = is[8] | y0MDY; is[7]  = is[8] | y0Abs; is[8] |=         y0PDY;
            break;
        case 1: //-+y
            is[2]  = is[7] | z0MDZ; is[5]  = is[7] | z0Abs; is[8]  = is[7] | z0PDZ;
            is[0]  = is[2] | x0MDX; is[1]  = is[2] | x0Abs; is[2] |=         x0PDX;
            is[3]  = is[5] | x0MDX; is[4]  = is[5] | x0Abs; is[5] |=         x0PDX;
            is[6]  = is[8] | x0MDX; is[7]  = is[8] | x0Abs; is[8] |=         x0PDX;
            break;
        case 2: //-+z
            is[2]  = is[7] | y0MDY; is[5]  = is[7] | y0Abs; is[8]  = is[7] | y0PDY;
            is[0]  = is[2] | x0MDX; is[1]  = is[2] | x0Abs; is[2] |=         x0PDX;
            is[3]  = is[5] | x0MDX; is[4]  = is[5] | x0Abs; is[5] |=         x0PDX;
            is[6]  = is[8] | x0MDX; is[7]  = is[8] | x0Abs; is[8] |=         x0PDX;
            break;
	case 3: //+x-+y
	case 4: //-x-+y
	    is[2]  = is[10] | x0Abs; is[0]  =  is[2]  | z0MDZ; is[1]  =  is[2]  | z0Abs; is[2]  |=          z0PDZ;
	    is[5]  = is[10] | x0PDX; is[3]  =  is[5]  | z0MDZ; is[4]  =  is[5]  | z0Abs; is[5]  |=          z0PDZ;
	    is[8]  = is[10] | is[7]; is[6]  =  is[8]  | z0MDZ; is[7]  =  is[8]  | z0Abs; is[8]  |=          z0PDZ;
	    is[11] = is[7]  | y0Abs; is[9]  =  is[11] | z0MDZ; is[10] =  is[11] | z0Abs; is[11] |=          z0PDZ;
	    is[14] = is[7]  | y0PDY; is[12] =  is[14] | z0MDZ; is[13] =  is[14] | z0Abs; is[14] |=          z0PDZ;
	    break;
	case 5: //+y-+z
	case 6: //-y-+z
	    is[2]  = is[10] | y0Abs; is[0]  =  is[2]  | x0MDX; is[1]  =  is[2]  | x0Abs; is[2]  |=          x0PDX;
	    is[5]  = is[10] | y0PDY; is[3]  =  is[5]  | x0MDX; is[4]  =  is[5]  | x0Abs; is[5]  |=          x0PDX;
	    is[8]  = is[10] | is[7]; is[6]  =  is[8]  | x0MDX; is[7]  =  is[8]  | x0Abs; is[8]  |=          x0PDX;
	    is[11] = is[7]  | z0Abs; is[9]  =  is[11] | x0MDX; is[10] =  is[11] | x0Abs; is[11] |=          x0PDX;
	    is[14] = is[7]  | z0PDZ; is[12] =  is[14] | x0MDX; is[13] =  is[14] | x0Abs; is[14] |=          x0PDX;
	    break;
	case 7: //+z-+x
	case 8: //-z-+x
	    is[2]  = is[10] | z0Abs; is[0]  =  is[2]  | y0MDY; is[1]  =  is[2]  | y0Abs; is[2]  |=          y0PDY;
	    is[5]  = is[10] | z0PDZ; is[3]  =  is[5]  | y0MDY; is[4]  =  is[5]  | y0Abs; is[5]  |=          y0PDY;
	    is[8]  = is[10] | is[7]; is[6]  =  is[8]  | y0MDY; is[7]  =  is[8]  | y0Abs; is[8]  |=          y0PDY;
	    is[11] = is[7]  | x0Abs; is[9]  =  is[11] | y0MDY; is[10] =  is[11] | y0Abs; is[11] |=          y0PDY;
	    is[14] = is[7]  | x0PDX; is[12] =  is[14] | y0MDY; is[13] =  is[14] | y0Abs; is[14] |=          y0PDY;
	    break;
    }
    return ( tex1Dfetch< uint8_t >( texLattice, is[0] ) |
                        tex1Dfetch< uint8_t >( texLattice, is[1] ) |
			tex1Dfetch< uint8_t >( texLattice, is[2] ) |
			tex1Dfetch< uint8_t >( texLattice, is[3] ) |
			tex1Dfetch< uint8_t >( texLattice, is[4] ) |
			tex1Dfetch< uint8_t >( texLattice, is[5] ) |
			tex1Dfetch< uint8_t >( texLattice, is[6] ) |
			tex1Dfetch< uint8_t >( texLattice, is[7] ) |
			tex1Dfetch< uint8_t >( texLattice, is[8] ) );
    
//     switch ( axis >> intCUDA(1) )
//     {
//       case 0:
//       case 1:
//       case 2:  return ( tex1Dfetch< uint8_t >( texLattice, is[0] ) |
//                         tex1Dfetch< uint8_t >( texLattice, is[1] ) |
// 			tex1Dfetch< uint8_t >( texLattice, is[2] ) |
// 			tex1Dfetch< uint8_t >( texLattice, is[3] ) |
// 			tex1Dfetch< uint8_t >( texLattice, is[4] ) |
// 			tex1Dfetch< uint8_t >( texLattice, is[5] ) |
// 			tex1Dfetch< uint8_t >( texLattice, is[6] ) |
// 			tex1Dfetch< uint8_t >( texLattice, is[7] ) |
// 			tex1Dfetch< uint8_t >( texLattice, is[8] ) );
//       case 3:
//       case 4:
//       case 5:
//       case 6:
//       case 7:
//       case 8: return ( tex1Dfetch< uint8_t >( texLattice, is[0]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[1]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[2]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[3]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[4]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[5]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[6]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[7]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[8]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[9]  ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[10] ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[11] ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[12] ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[13] ) |
// 		       tex1Dfetch< uint8_t >( texLattice, is[14] ) );
//     }
}

#ifdef USE_BIT_PACKING
/*TODO: Test!!
 * I dont know if it works or not -_-
 */
__device__ inline bool checkFrontBitPacked
(
    cudaTextureObject_t const & texLattice,
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    intCUDA             const & axis
)
{
    auto const x0Abs  = diluteBits< uint32_t, 2 >( ( x0               ) & dcBoxXM1 );
    auto const x0PDX  = diluteBits< uint32_t, 2 >( ( x0 + uint32_t(1) ) & dcBoxXM1 );
    auto const x0MDX  = diluteBits< uint32_t, 2 >( ( x0 - uint32_t(1) ) & dcBoxXM1 );
    auto const y0Abs  = diluteBits< uint32_t, 2 >( ( y0               ) & dcBoxYM1 ) << 1;
    auto const y0PDY  = diluteBits< uint32_t, 2 >( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << 1;
    auto const y0MDY  = diluteBits< uint32_t, 2 >( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << 1;
    auto const z0Abs  = diluteBits< uint32_t, 2 >( ( z0               ) & dcBoxZM1 ) << 2;
    auto const z0PDZ  = diluteBits< uint32_t, 2 >( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << 2;
    auto const z0MDZ  = diluteBits< uint32_t, 2 >( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << 2;
    
    auto const dx = DXTable_d[ axis ];   // 2*axis-1
    auto const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    auto const dz = DZTable_d[ axis ];   // 2*(axis&1)-1

//     uint32_t is[9];
    uint32_t is[15];
    switch ( axis >> intCUDA(1) )
    {
        case 0: is[7] = diluteBits< uint32_t, 2 >(( x0 + decltype(dx)(2) * dx ) & dcBoxXM1)   ; break;
        case 1: is[7] = diluteBits< uint32_t, 2 >(( y0 + decltype(dy)(2) * dy ) & dcBoxYM1)<<1; break;
        case 2: is[7] = diluteBits< uint32_t, 2 >(( z0 + decltype(dz)(2) * dz ) & dcBoxZM1)<<2; break;
	case 3:
	case 4: is[7] = diluteBits< uint32_t, 2 >(( x0 + decltype(dx)(2) * dx ) & dcBoxXM1)   ; is[10] = diluteBits< uint32_t, 2 >(( y0 + decltype(dy)(2) * dy ) & dcBoxYM1)<<1; break;
	case 5:
	case 6: is[7] = diluteBits< uint32_t, 2 >(( y0 + decltype(dy)(2) * dy ) & dcBoxYM1)<<1; is[10] = diluteBits< uint32_t, 2 >(( z0 + decltype(dz)(2) * dz ) & dcBoxZM1)<<2; break;
	case 7:
	case 8: is[7] = diluteBits< uint32_t, 2 >(( z0 + decltype(dz)(2) * dz ) & dcBoxZM1)<<2; is[10] = diluteBits< uint32_t, 2 >(( x0 + decltype(dx)(2) * dx ) & dcBoxXM1)   ; break;
    }
//     is[7] = diluteBits< uint32_t, 2 >( is[7] ) << ( axis >> intCUDA(1) );
    switch ( axis >> intCUDA(1) )
    {
        case 0: //-+x
            is[2]  = is[7] + z0Abs; is[5]  = is[7] + z0MDZ; is[8]  = is[7] + z0PDZ;
            is[0]  = is[2] + y0MDY; is[1]  = is[2] + y0Abs; is[2] +=         y0PDY;
            is[3]  = is[5] + y0MDY; is[4]  = is[5] + y0Abs; is[5] +=         y0PDY;
            is[6]  = is[8] + y0MDY; is[7]  = is[8] + y0Abs; is[8] +=         y0PDY;
            break;
        case 1: //-+y
            is[2]  = is[7] + z0MDZ; is[5]  = is[7] + z0Abs; is[8]  = is[7] + z0PDZ;
            is[0]  = is[2] + x0MDX; is[1]  = is[2] + x0Abs; is[2] +=         x0PDX;
            is[3]  = is[5] + x0MDX; is[4]  = is[5] + x0Abs; is[5] +=         x0PDX;
            is[6]  = is[8] + x0MDX; is[7]  = is[8] + x0Abs; is[8] +=         x0PDX;
            break;
        case 2: //-+z
            is[2]  = is[7] + y0MDY; is[5]  = is[7] + y0Abs; is[8]  = is[7] + y0PDY;
            is[0]  = is[2] + x0MDX; is[1]  = is[2] + x0Abs; is[2] +=         x0PDX;
            is[3]  = is[5] + x0MDX; is[4]  = is[5] + x0Abs; is[5] +=         x0PDX;
            is[6]  = is[8] + x0MDX; is[7]  = is[8] + x0Abs; is[8] +=         x0PDX;
            break;
	case 3: //+x-+y
	case 4: //-x-+y
	    is[2]  = is[10] + x0Abs; is[0]  =  is[2]  + z0MDZ; is[1]  =  is[2]  + z0Abs; is[2]  +=          z0PDZ;
	    is[5]  = is[10] + x0PDX; is[3]  =  is[5]  + z0MDZ; is[4]  =  is[5]  + z0Abs; is[5]  +=          z0PDZ;
	    is[8]  = is[10] + is[7]; is[6]  =  is[8]  + z0MDZ; is[7]  =  is[8]  + z0Abs; is[8]  +=          z0PDZ;
	    is[11] = is[7]  + y0Abs; is[9]  =  is[11] + z0MDZ; is[10] =  is[11] + z0Abs; is[11] +=          z0PDZ;
	    is[14] = is[7]  + y0PDY; is[12] =  is[14] + z0MDZ; is[13] =  is[14] + z0Abs; is[14] +=          z0PDZ;
	    break;
	case 5: //+y-+z
	case 6: //-y-+z
	    is[2]  = is[10] + y0Abs; is[0]  =  is[2]  + x0MDX; is[1]  =  is[2]  + x0Abs; is[2]  +=          x0PDX;
	    is[5]  = is[10] + y0PDY; is[3]  =  is[5]  + x0MDX; is[4]  =  is[5]  + x0Abs; is[5]  +=          x0PDX;
	    is[8]  = is[10] + is[7]; is[6]  =  is[8]  + x0MDX; is[7]  =  is[8]  + x0Abs; is[8]  +=          x0PDX;
	    is[11] = is[7]  + z0Abs; is[9]  =  is[11] + x0MDX; is[10] =  is[11] + x0Abs; is[11] +=          x0PDX;
	    is[14] = is[7]  + z0PDZ; is[12] =  is[14] + x0MDX; is[13] =  is[14] + x0Abs; is[14] +=          x0PDX;
	    break;
	case 7: //+z-+x
	case 8: //-z-+x
	    is[2]  = is[10] + z0Abs; is[0]  =  is[2]  + y0MDY; is[1]  =  is[2]  + y0Abs; is[2]  +=          y0PDY;
	    is[5]  = is[10] + z0PDZ; is[3]  =  is[5]  + y0MDY; is[4]  =  is[5]  + y0Abs; is[5]  +=          y0PDY;
	    is[8]  = is[10] + is[7]; is[6]  =  is[8]  + y0MDY; is[7]  =  is[8]  + y0Abs; is[8]  +=          y0PDY;
	    is[11] = is[7]  + x0Abs; is[9]  =  is[11] + y0MDY; is[10] =  is[11] + y0Abs; is[11] +=          y0PDY;
	    is[14] = is[7]  + x0PDX; is[12] =  is[14] + y0MDY; is[13] =  is[14] + y0Abs; is[14] +=          y0PDY;
	    break;
    }
    switch ( axis >> intCUDA(1) ) 
    {
      case 0:
      case 1:
      case 2: return bitPackedTextureGet< uint8_t >( texLattice, is[0] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[1] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[2] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[3] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[4] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[5] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[6] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[7] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[8] );
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8: return bitPackedTextureGet< uint8_t >( texLattice, is[0]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[1]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[2]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[3]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[4]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[5]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[6]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[7]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[8]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[9]  ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[10] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[11] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[12] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[13] ) +
		     bitPackedTextureGet< uint8_t >( texLattice, is[14] );
    }
}
#endif

__device__ __host__ inline uintCUDA linearizeBondVectorIndex
(
    intCUDA const x,
    intCUDA const y,
    intCUDA const z
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
    return   ( x & intCUDA(7) /* 0b111 */ ) +
           ( ( y & intCUDA(7) /* 0b111 */ ) << intCUDA(3) ) +
           ( ( z & intCUDA(7) /* 0b111 */ ) << intCUDA(6) );
}


/**
 * Recheck whether the move is possible without collision, using the
 * temporarily parallel executed moves saved in texLatticeTmp. If so,
 * do the move in dpLattice. (Still not applied in dpPolymerSystem!)
 */
__global__ void kernelSimulationScBFMPerformSpecies
(
    Move                                     move           , 
    vecIntCUDA    const * const __restrict__ dpPolymerSystem,
    T_Flags             * const __restrict__ dpPolymerFlags ,
    uint8_t             * const __restrict__ dpLattice      ,
    uint32_t              const              nMonomers      ,
    cudaTextureObject_t   const              texLatticeTmp
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        auto const r0 = dpPolymerSystem[ iMonomer ];
        //uint3 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z }; // slower
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111
//         auto const direction = ( properties >> T_Flags(2) ) & T_Flags(31); // 31=0b11111
// 	auto const direction = move.getDirectionID(properties);
	#ifdef USE_BIT_PACKING_TMP_LATTICE
        if ( checkFrontBitPacked( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #else
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #endif
            continue;

        /* If possible, perform move now on normal lattice */
        dpPolymerFlags[ iMonomer ] = properties | T_Flags(2); // indicating allowed move
        dpLattice[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
        dpLattice[ linearizeBoxVectorIndex( r0.x + DXTable_d[ direction ],
                                            r0.y + DYTable_d[ direction ],
                                            r0.z + DZTable_d[ direction ] ) ] = 1;
        /* We can't clean the temporary lattice in here, because it still is being
         * used for checks. For cleaning we need only the new positions.
         * But we can't use the applied positions, because we also need to clean
         * those particles which couldn't move in this second kernel but where
         * still set in the lattice by the first kernel! */
    }
}

/**
 * Apply move to dpPolymerSystem and clean the temporary lattice of moves
 * which seemed like they would work, but did clash with another parallel
 * move, unfortunately.
 * @todo it might be better to just use a cudaMemset to clean the lattice,
 *       that way there wouldn't be any memory dependencies and calculations
 *       needed, even though we would have to clean everything, instead of
 *       just those set. But that doesn't matter, because most of the threads
 *       are idling anyway ...
 *       This kind of kernel might give some speedup after stream compaction
 *       has been implemented though.
 *    -> print out how many percent of cells need to be cleaned .. I need
 *       many more statistics anyway for evaluating performance benefits a bit
 *       better!
 */
__global__ void kernelSimulationScBFMZeroArraySpecies
(   
    Move                                     move           , 
    vecIntCUDA          * const __restrict__ dpPolymerSystem,
    T_Flags       const * const __restrict__ dpPolymerFlags ,
    uint8_t             * const __restrict__ dpLatticeTmp   ,
    uint32_t              const              nMonomers
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(3) ) == T_Flags(0) )    // impossible move
            continue;

        auto r0 = dpPolymerSystem[ iMonomer ];
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111
//         auto const direction = ( properties >> T_Flags(2) ) & T_Flags(31); // 31=0b11111
// 	auto const direction = move.getDirectionID(properties);

        r0.x += DXTableIntCUDA_d[ direction ];
        r0.y += DYTableIntCUDA_d[ direction ];
        r0.z += DZTableIntCUDA_d[ direction ];
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        //bitPackedUnset( dpLatticeTmp, linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) );
        dpLatticeTmp[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) >> 3 ] = 0;
    #else
        dpLatticeTmp[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
    #endif
        if ( ( properties & T_Flags(3) ) == T_Flags(3) )  // 3=0b11
            dpPolymerSystem[ iMonomer ] = r0;
    }
}



/**
 * Goes over all monomers of a species given specified by texSpeciesIndices
 * draws a random direction for them and checks whether that move is possible
 * with the box size and periodicity as well as the monomers at the target
 * location (excluded volume) and the new bond lengths to all neighbors.
 * If so, then the new position is set to 1 in dpLatticeTmp and encode the
 * possible movement direction in the property tag of the corresponding monomer
 * in dpPolymerSystem.
 * Note that the old position is not removed in order to correctly check for
 * excluded volume a second time.
 *
 * @param[in] rn a random number used as a kind of seed for the RNG
 * @param[in] nMonomers number of max. monomers to work on, this is for
 *            filtering out excessive threads and was prior a __constant__
 *            But it is only used one(!) time in the kernel so the caching
 *            of constant memory might not even be used.
 *            @see https://web.archive.org/web/20140612185804/http://www.pixel.io/blog/2013/5/9/kernel-arguments-vs-__constant__-variables.html
 *            -> Kernel arguments are even put into constant memory it seems:
 *            @see "Section E.2.5.2 Function Parameters" in the "CUDA 5.5 C Programming Guide"
 *            __global__ function parameters are passed to the device:
 *             - via shared memory and are limited to 256 bytes on devices of compute capability 1.x,
 *             - via constant memory and are limited to 4 KB on devices of compute capability 2.x and higher.
 *            __device__ and __global__ functions cannot have a variable number of arguments.
 * @param[in] iOffset Offste to curent species we are supposed to work on.
 *            We can't simply adjust dpPolymerSystem before calling the kernel,
 *            because we are accessing the neighbors, therefore need all the
 *            polymer data, especially for other species.
 *
 * Note: all of the three kernels do quite few work. They basically just fetch
 *       data, and check one condition and write out again. There isn't even
 *       a loop and most of the work seems to be boiler plate initialization
 *       code which could be cut if the kernels could be merged together.
 *       Why are there three kernels instead of just one
 *        -> for global synchronization
 */
template<typename RNG, typename CheckBox, typename LocalMetropolis, typename Move>
__global__ void kernelSimulationScBFMCheckSpecies
(
    Move 				   move                    ,
    LocalMetropolis 			   localMetropolis         ,  
    CheckBox                               checkBox                ,
    vecIntCUDA  const * const __restrict__ dpPolymerSystem         ,
    T_Flags           * const __restrict__ dpPolymerFlags          ,
    uint32_t            const              iOffset                 ,
    T_Flags           * const __restrict__ dpLatticeTmp            ,
    uint32_t    const * const __restrict__ dpNeighbors             ,
    uint32_t            const              rNeighborsPitchElements ,
    uint8_t     const * const __restrict__ dpNeighborsSizes        ,
    uint32_t            const              nMonomers               ,
    uint64_t            const              rSeed                   ,
    uint64_t            const              global_iteration        ,
    typename RNG::GlobalState *            global_rng_states       ,
    cudaTextureObject_t const              texLatticeRefOut
)
{

    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
        /* upcast int3 to int4 in preparation to use PTX SIMD instructions */
        //int4 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z, 0 }; // not faster nor slower
        //select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
	RNG generator;
        if ( iGrid % 1 == 0 ){
	      RNG generator;
	      if( RNG::needsGlobalState() ) generator.setGlobalState( global_rng_states );
	      if( RNG::needsIteration() ) generator.setIteration( global_iteration );
	      if( RNG::needsSubsequence() ) generator.setSubsequence(iMonomer);
	      if( RNG::needsSeed() ) generator.setSeed( rSeed );

	    rn = generator.rng32();
	}
        T_Flags const direction = rn % T_Flags(6); rn /= T_Flags(6);
// 	T_Flags const direction = rn % T_Flags(18);
// 	T_Flags const direction = move.getDirectionID(rn);
	T_Flags properties = 0;

         /* select random direction. Do this with bitmasking instead of lookup ??? */
        uint3 const r1 = { r0.x + DXTable_d[ direction ],
                           r0.y + DYTable_d[ direction ],
                           r0.z + DZTable_d[ direction ] };
	  if ( checkBox(r1.x, r1.y, r1.z) )
	  {
		/* check whether the new position would result in invalid bonds
		* between this monomer and its neighbors */
		auto const nNeighbors = dpNeighborsSizes[ iOffset + iMonomer ];
		bool forbiddenBond = false;
		for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
		{
		    auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
		    auto const data2 = dpPolymerSystem[ iGlobalNeighbor ];
		    if ( dpForbiddenBonds[ linearizeBondVectorIndex( data2.x - r1.x, data2.y - r1.y, data2.z - r1.z ) ] )
		    {
			forbiddenBond = true;
			break;
		    }
		}
		if ( ! forbiddenBond 
		  && ! checkFront( texLatticeRefOut, r0.x, r0.y, r0.z, direction ) )
// 		  && localMetropolis(r1.z, DZTableIntCUDA_d[ direction ]) > generator.rng_d() )
		{   
			/* everything fits so perform move on temporary lattice */
			/* can I do this ??? dpPolymerSystem is the device pointer to the read-only
			* texture used above. Won't this result in read-after-write race-conditions?
			* Then again the written / changed bits are never used in the above code ... */
			properties = ( direction << T_Flags(2) ) + T_Flags(1) /* can-move-flag */;
		    #ifdef USE_BIT_PACKING_TMP_LATTICE
			bitPackedSet( dpLatticeTmp, linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) );
		    #else
			dpLatticeTmp[ linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
		    #endif
		}
	  }
        dpPolymerFlags[ iOffset + iMonomer ] = properties;
    }
}

__global__ void kernelSimulationScBFMPerformSpeciesAndApply
(
    Move 				     move           ,
    vecIntCUDA          * const __restrict__ dpPolymerSystem,
    T_Flags             * const __restrict__ dpPolymerFlags ,
    uint8_t             * const __restrict__ dpLattice      ,
    uint32_t              const              nMonomers      ,
    cudaTextureObject_t   const              texLatticeTmp
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        auto const r0 = ( (CudaVec4< intCUDA >::value_type *) dpPolymerSystem )[ iMonomer ];
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111
// 	auto const direction = ( properties >> T_Flags(2) ) & T_Flags(31); // 31=0b11111
// 	auto const direction = move.setDirectionID(properties);
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        if ( checkFrontBitPacked( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #else
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #endif
            continue;

        CudaVec4< intCUDA >::value_type const r1 = {
            intCUDA( r0.x + DXTableIntCUDA_d[ direction ] ),
            intCUDA( r0.y + DYTableIntCUDA_d[ direction ] ),
            intCUDA( r0.z + DZTableIntCUDA_d[ direction ] ), 0
        };
        /* If possible, perform move now on normal lattice */
        dpLattice[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
        dpLattice[ linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
        dpPolymerSystem[ iMonomer ] = r1;
    }
}

template<typename RNG, typename CheckBox, typename LocalMetropolis>
__global__ void kernelSimulationScBFMwoEVCheckPerformApplySpecies
(
    Move 				   move                    ,
    LocalMetropolis 			   localMetropolis         ,  
    CheckBox                               checkBox                ,
    vecIntCUDA        * const __restrict__ dpPolymerSystem         ,
    uint32_t            const              iOffset                 ,
    uint32_t    const * const __restrict__ dpNeighbors             ,
    uint32_t            const              rNeighborsPitchElements ,
    uint8_t     const * const __restrict__ dpNeighborsSizes        ,
    uint32_t            const              nMonomers               ,
    uint64_t            const              rSeed                   ,
    uint64_t            const              global_iteration        ,
    typename RNG::GlobalState *            global_rng_states       
)
{
//     uint32_t rn;
//     auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
	  auto r0 = dpPolymerSystem[ iOffset + iMonomer ];
	  RNG generator;
	  if ( iGrid % 1 == 0 ){
		RNG generator;
		if( RNG::needsGlobalState() ) generator.setGlobalState( global_rng_states );
		if( RNG::needsIteration() ) generator.setIteration( global_iteration );
		if( RNG::needsSubsequence() ) generator.setSubsequence(iMonomer);
		if( RNG::needsSeed() ) generator.setSeed( rSeed );
	    rn = generator.rng32();
	  }
	  T_Flags const direction = rn % T_Flags(6); rn /= T_Flags(6);
// 	  T_Flags const direction = rn % T_Flags(18);
// 	  T_Flags const direction = move.getDirectionID(rn);
	    /* select random direction. Do this with bitmasking instead of lookup ??? */
	  uint3 const r1 = { r0.x + DXTable_d[ direction ],
			      r0.y + DYTable_d[ direction ],
			      r0.z + DZTable_d[ direction ] };
	  if ( checkBox(r1.x, r1.y, r1.z) )
	  {
		/* check whether the new position would result in invalid bonds
		* between this monomer and its neighbors */
		auto const nNeighbors = dpNeighborsSizes[ iOffset + iMonomer ];
		bool forbiddenBond = false;
		for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
		{
		    auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
		    auto const data2 = dpPolymerSystem[ iGlobalNeighbor ];
		    if ( dpForbiddenBonds[ linearizeBondVectorIndex( data2.x - r1.x, data2.y - r1.y, data2.z - r1.z ) ] )
		    {
			forbiddenBond = true;
			break;
		    }
		}
		if ( ! forbiddenBond  && (localMetropolis(r1.z, DZTableIntCUDA_d[ direction ]) < generator.rng_d()) ) 
		{
		      /* If possible, perform move now */
		      r0.x += DXTableIntCUDA_d[direction];
		      r0.y += DYTableIntCUDA_d[direction];
		      r0.z += DZTableIntCUDA_d[direction];
		      dpPolymerSystem[ iOffset + iMonomer ] = r0;
		}
	  }
    }
    
}
/**
 * Goes over all ring monomers and make an attempt to connect to another monomer
 * of the host chain. 
 * @param[in] rn a random number used as a kind of seed for the RNG
 * @param[in] nMonomers number of max. monomers to work on, this is for
 *            filtering out excessive threads and was prior a __constant__
 *            But it is only used one(!) time in the kernel so the caching
 *            of constant memory might not even be used.
 *            @see https://web.archive.org/web/20140612185804/http://www.pixel.io/blog/2013/5/9/kernel-arguments-vs-__constant__-variables.html
 *            -> Kernel arguments are even put into constant memory it seems:
 *            @see "Section E.2.5.2 Function Parameters" in the "CUDA 5.5 C Programming Guide"
 *            __global__ function parameters are passed to the device:
 *             - via shared memory and are limited to 256 bytes on devices of compute capability 1.x,
 *             - via constant memory and are limited to 4 KB on devices of compute capability 2.x and higher.
 *            __device__ and __global__ functions cannot have a variable number of arguments.
 * @param[in] iOffset Offste to curent species we are supposed to work on.
 *            We can't simply adjust dpPolymerSystem before calling the kernel,
 *            because we are accessing the neighbors, therefore need all the
 *            polymer data, especially for other species.

 */    

template<typename RNG>
__global__ void kernelMoveRingBond
(
    vecIntCUDA  const * const __restrict__ dpPolymerSystem         ,
    uint8_t     const * const __restrict__ dStructureTag	   ,  
    uint32_t            const              iOffset                 ,
    uint32_t          * const __restrict__ dpNeighbors             ,
    uint32_t            const              rNeighborsPitchElements ,
    uint8_t           * const __restrict__ dpNeighborsSizes        ,
    uint32_t          * const __restrict__ dAllNeighbors           ,
    uint32_t            const              nMonomers               ,
    uint64_t            const              rSeed                   ,
    uint64_t            const              global_iteration        ,
    typename RNG::GlobalState *            global_rng_states       
)
{

    double rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
	auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
	/* check whether the new position would result in invalid bonds
	  * between this monomer and its neighbors */
	auto const nNeighbors = dpNeighborsSizes[ iOffset + iMonomer ];
	for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
	{
	      //get neighbor
	      auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
	      // get the species offset and check that neighbor is a chain monomer and not another ring 
	      auto OffsetSpeciesNeighbor     =  dMatrixOffsetElementsOne  - diSubGroupOffsetOne ; 
	      auto OffsetSpeciesNextNeighbor =  dMatrixOffsetElementsZero - diSubGroupOffsetZero;  // is 0
	      if ( iGlobalNeighbor <  diSubGroupOffsetOne ) //the neighbor id belongs to the species 0 
		    { 
		      OffsetSpeciesNeighbor     = dMatrixOffsetElementsZero - diSubGroupOffsetZero;
		      OffsetSpeciesNextNeighbor = dMatrixOffsetElementsOne  - diSubGroupOffsetOne ; 
		    }
	      else if ( iGlobalNeighbor >= diSubGroupOffsetTwo ) //the neighbor id belongs to the species greater 1 which are the ring species 
		    continue;
	      //if still here then the chain neighbor is found 
	      auto const nOfNextNeighbors = dpNeighborsSizes[ iGlobalNeighbor ]; 
	      uint32_t counter(0); 
	      uint32_t neighbor[2];
	      
	      for ( auto iNextNeighbor = decltype( nOfNextNeighbors )(0); iNextNeighbor < nOfNextNeighbors; ++iNextNeighbor )
	      {
		  auto const iGlobalNextNeighbor = dAllNeighbors[ iNextNeighbor * dChainPitch + iGlobalNeighbor + OffsetSpeciesNeighbor ];
		  if ( iGlobalNextNeighbor < diSubGroupOffsetTwo  )
		  {
		    neighbor[counter]=iGlobalNextNeighbor;
		    counter++; 
		  }
	      }
	      //  randomly choosen neighbors
  	      if ( counter > 1  )
  	      {         
		RNG generator;
		if( RNG::needsGlobalState() )
		    generator.setGlobalState( global_rng_states );
		if( RNG::needsIteration() )
		    generator.setIteration( global_iteration );
		if( RNG::needsSubsequence() )
		    generator.setSubsequence(iMonomer);
		if( RNG::needsSeed() )
		    generator.setSeed( rSeed );

		rn = generator.rng_d();
  		if( rn < 0.5 ) counter=0;
  		else           counter=1;
  	      }else  counter = 0; 
	      auto NewChainMonomer = neighbor[counter]; 
	      /* check whether the new potential bond partner has already a ring monomer attached ( means 3 bonds or 2 for the end monomers of the chain)*/
	      if ( dpNeighborsSizes[ NewChainMonomer ] >= dStructureTag[NewChainMonomer] ) break;

	      // check whether the new bond is allowed
	      auto const Position = dpPolymerSystem [ NewChainMonomer ];
	      auto const X =  Position.x - r0.x;
	      auto const Y =  Position.y - r0.y;
	      auto const Z =  Position.z - r0.z; 
	      if ( X < -3 || X > 3 || Y < -3 || Y > 3 || Z < -3 || Z > 3 ) break; 
	      if ( dpForbiddenBonds[ linearizeBondVectorIndex( X, Y, Z ) ] ) break;
	      
	      //still here, then the bond is allowed and can be registered! 
	      //ring neighbors
	      dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ] = NewChainMonomer;
	      //old ring neighbor 
	      dpNeighborsSizes[ iGlobalNeighbor ]--;
	      for ( uint32_t iNextNeighbor = 0;  iNextNeighbor <  dpNeighborsSizes[ iGlobalNeighbor ]; ++iNextNeighbor )
		    dAllNeighbors[ iNextNeighbor * dChainPitch + iGlobalNeighbor + OffsetSpeciesNeighbor ] = neighbor[iNextNeighbor]; 
	      //new ring neighbor 
	      dpNeighborsSizes[ NewChainMonomer ]++;
	      dAllNeighbors[ (dpNeighborsSizes[ NewChainMonomer ]-1) * dChainPitch + NewChainMonomer + OffsetSpeciesNextNeighbor ] = (iOffset + iMonomer); 
	}
      }
}


/********************************************
 * Start declaring the functions of the class
 ********************************************/


UpdaterGPUScBFMGPR_AB_Type::UpdaterGPUScBFMGPR_AB_Type()
 : mStream              ( 0 ),
   nAllMonomers         ( 0 ),
   nRingMonomers        ( 0 ),
   nChains        	( 0 ),
   nMonomersPerChain    ( 0 ),
   mLattice             ( NULL ),
   mLatticeOut          ( NULL ),
   mLatticeTmp          ( NULL ),
   mPolymerSystemSorted ( NULL ),
   mMonomerStructureTag ( NULL ),
   mPolymerFlags        ( NULL ),
   mNeighborsSorted     ( NULL ),
   mNeighborsSortedSizes( NULL ),
   mNeighborsSortedInfo ( nBytesAlignment ),
   mAttributeSystem     ( NULL ),
   mBoxX                ( 0 ),
   mBoxY                ( 0 ),
   mBoxZ                ( 0 ),
   mBoxXM1              ( 0 ),
   mBoxYM1              ( 0 ),
   mBoxZM1              ( 0 ),
   mBoxXLog2            ( 0 ),
   mBoxXYLog2           ( 0 )
{
    /**
     * Log control.
     * Note that "Check" controls not the output, but the actualy checks
     * If a checks needs to always be done, then do that check and declare
     * the output as "Info" log level
     */
    mLog.file( __FILENAME__ );
    mLog.  activate( "Benchmark" );
    mLog.  activate( "Check"     );
    mLog.  activate( "Error"     );
    mLog.  activate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
}

/**
 * Deletes everything which could and is allocated
 */
void UpdaterGPUScBFMGPR_AB_Type::destruct()
{
    if ( mLattice              != NULL ){ delete[] mLattice             ; mLattice              = NULL; }  // setLatticeSize
    if ( mLatticeOut           != NULL ){ delete   mLatticeOut          ; mLatticeOut           = NULL; }  // initialize
    if ( mLatticeTmp           != NULL ){ delete   mLatticeTmp          ; mLatticeTmp           = NULL; }  // initialize
    if ( mPolymerSystemSorted  != NULL ){ delete   mPolymerSystemSorted ; mPolymerSystemSorted  = NULL; }  // initialize
    if ( mMonomerStructureTag  != NULL ){ delete   mMonomerStructureTag ; mMonomerStructureTag  = NULL; }  // initialize
    if ( mPolymerFlags         != NULL ){ delete   mPolymerFlags        ; mPolymerFlags         = NULL; }  // initialize
    if ( mNeighborsSorted      != NULL ){ delete   mNeighborsSorted     ; mNeighborsSorted      = NULL; }  // initialize
    if ( mNeighborsSortedSizes != NULL ){ delete   mNeighborsSortedSizes; mNeighborsSortedSizes = NULL; }  // initialize
    if ( mAttributeSystem      != NULL ){ delete[] mAttributeSystem     ; mAttributeSystem      = NULL; }  // setNrOfAllMonomers
}

UpdaterGPUScBFMGPR_AB_Type::~UpdaterGPUScBFMGPR_AB_Type()
{
    this->destruct();
}

void UpdaterGPUScBFMGPR_AB_Type::setGpu( int iGpuToUse )
{
    int nGpus;
    getCudaDeviceProperties( NULL, &nGpus, true /* print GPU information */ );
    if ( ! ( 0 <= iGpuToUse && iGpuToUse < nGpus ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setGpu] "
            << "GPU with ID " << iGpuToUse << " not present. "
            << "Only " << nGpus << " GPUs are available.\n";
        mLog( "Error" ) << msg.str();
        throw std::invalid_argument( msg.str() );
    }
    CUDA_ERROR( cudaSetDevice( iGpuToUse ) );
    miGpuToUse = iGpuToUse;
}


void UpdaterGPUScBFMGPR_AB_Type::initialize( void )
{
    if ( mLog( "Stats" ).isActive() )
    {
        // this is called in parallel it seems, therefore need to buffer it
        std::stringstream msg; msg
        << "[" << __FILENAME__ << "::initialize] The "
        << "(" << mBoxX << "," << mBoxY << "," << mBoxZ << ")"
        << " lattice is populated by " << nAllMonomers
        << " resulting in a filling rate of "
        << nAllMonomers / double( mBoxX * mBoxY * mBoxZ ) << "\n";
        mLog( "Stats" ) << msg.str();
    }

    if ( mLatticeOut != NULL || mLatticeTmp != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "Initialize was already called and may not be called again "
            << "until cleanup was called!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

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
        msg << "[" << __FILENAME__ << "::initialize] "
            << "Wrong bond-set! Expected 108 allowed bonds, but got " << nAllowedBonds << "\n";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }
    CUDA_ERROR( cudaMemcpyToSymbol( dpForbiddenBonds, tmpForbiddenBonds, sizeof(bool)*512 ) );
    free( tmpForbiddenBonds );

    /* create a table mapping the random int to directions whereto move the
     * monomers. We can use negative numbers, because (0u-1u)+1u still is 0u */
    uint32_t tmp_DXTable[6] = { 0u-1u,1,  0,0,  0,0 };
    uint32_t tmp_DYTable[6] = {  0,0, 0u-1u,1,  0,0 };
    uint32_t tmp_DZTable[6] = {  0,0,  0,0, 0u-1u,1 };
//     uint32_t tmp_DXTable[18] = { 0u-1u, 1,     0, 0,     0, 0, 1,     1, 0u-1u, 0u-1u, 0,     0,     0,     0, 1, 0u-1u,     1, 0u-1u };
//     uint32_t tmp_DYTable[18] = {     0, 0, 0u-1u, 1,     0, 0, 1, 0u-1u,     1, 0u-1u, 1,     1, 0u-1u, 0u-1u, 0,     0,     0,     0 };
//     uint32_t tmp_DZTable[18] = {     0, 0,     0, 0, 0u-1u, 1, 0,     0,     0,     0, 1, 0u-1u,     1, 0u-1u, 1,     1, 0u-1u, 0u-1u };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTable_d, tmp_DXTable, sizeof( tmp_DXTable ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTable_d, tmp_DYTable, sizeof( tmp_DXTable ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTable_d, tmp_DZTable, sizeof( tmp_DXTable ) ) );
    intCUDA tmp_DXTableIntCUDA[6] = { -1, 1,  0, 0,  0, 0 };
    intCUDA tmp_DYTableIntCUDA[6] = {  0, 0, -1, 1,  0, 0 };
    intCUDA tmp_DZTableIntCUDA[6] = {  0, 0,  0, 0, -1, 1 };
//     intCUDA tmp_DXTableIntCUDA[18] = { -1, 1,  0, 0,  0, 0, 1,  1, -1, -1, 0,  0,  0,  0, 1, -1,  1, -1 };
//     intCUDA tmp_DYTableIntCUDA[18] = {  0, 0, -1, 1,  0, 0, 1, -1,  1, -1, 1,  1, -1, -1, 0,  0,  0,  0 };
//     intCUDA tmp_DZTableIntCUDA[18] = {  0, 0,  0, 0, -1, 1, 0,  0,  0,  0, 1, -1,  1, -1, 1,  1, -1, -1 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTableIntCUDA_d, tmp_DXTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTableIntCUDA_d, tmp_DYTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTableIntCUDA_d, tmp_DZTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );

    /*************************** start of grouping ***************************/

   mLog( "Info" ) << "Coloring graph ...\n";
    bool const bUniformColors = true; // setting this to true should yield more performance as the kernels are uniformly utilized
    
    if (RingSlidingON){   
	/* use the graphColoring for the virtual connectected rings */
	mRingGroupIds = graphColoring< MonomerEdges const *, uint32_t, uint8_t >(
	    &mNeighborsRings[0], mNeighborsRings.size(), bUniformColors,
	    []( MonomerEdges const * const & x, uint32_t const & i ){ return x[i].size; },
	    []( MonomerEdges const * const & x, uint32_t const & i, size_t const & j ){ return x[i].neighborIds[j]; }
	);
	/*increase color to get free colors for the chains*/
	uint8_t offset(2);
	for ( size_t i = 0u; i < mRingGroupIds.size(); ++i) mRingGroupIds[i]+=offset; 
	/*color the chains by hand */
	for ( size_t i = 0u; i < nChains*nMonomersPerChain; ++i )
	  mGroupIds.push_back(static_cast<uint8_t>( (i%2)) ); //label with color 1 and 2 in alternating form 
	mGroupIds.insert(mGroupIds.end(),mRingGroupIds.begin(),mRingGroupIds.end());
    }else 
    {
	mGroupIds = graphColoring< MonomerEdges const *, uint32_t, uint8_t >(
	    &mNeighbors[0], mNeighbors.size(), bUniformColors,
	    []( MonomerEdges const * const & x, uint32_t const & i ){ return x[i].size; },
	    []( MonomerEdges const * const & x, uint32_t const & i, size_t const & j ){ return x[i].neighborIds[j]; }
    );
    }
//     if ( mLog( "Check" ).isActive() )
//     { 
//       std::stringstream msg;
//       msg << "[" << __FILENAME__ << "::initialize] "
// 	  << " Coloring of the graph (ID, color) : \n";
//       for ( size_t i = 0u; i < mGroupIds.size(); ++i) msg << i << " " << (uint32_t)mGroupIds[i]<<"\n";
//       mLog( "Check" ) << msg.str();     
//     }
    /* count monomers per species before allocating per species arrays and
     * sorting the monomers into them */
    mnElementsInGroup.resize(0);
    for ( size_t i = 0u; i < mGroupIds.size(); ++i )
    {
        if ( mGroupIds[i] >= mnElementsInGroup.size() )
            mnElementsInGroup.resize( mGroupIds[i]+1, 0 );
        ++mnElementsInGroup[ mGroupIds[i] ];
    }
    /**
     * Generate new array which contains all sorted monomers aligned
     * @verbatim
     * ABABABABABA
     * A A A A A A
     *  B B B B B
     * AAAAAA  BBBBB
     *        ^ alignment
     * @endverbatim
     * in the worst case we are only one element ( 4*intCUDA ) over the
     * alignment with each group and need to fill up to nBytesAlignment for
     * all of them */
    /* virtual number of monomers which includes the additional alignment padding */
    auto const nMonomersPadded = nAllMonomers + ( nElementsAlignment - 1u ) * mnElementsInGroup.size();
    assert( mPolymerFlags == NULL );
    mPolymerFlags = new MirroredVector< T_Flags >( nMonomersPadded, mStream );
    CUDA_ERROR( cudaMemset( mPolymerFlags->gpu, 0, mPolymerFlags->nBytes ) );
    /* Calculate offsets / prefix sum including the alignment */
    assert( mPolymerSystemSorted == NULL );
    mPolymerSystemSorted = new MirroredVector< vecIntCUDA >( nMonomersPadded, mStream );
    #ifndef NDEBUG
        std::memset( mPolymerSystemSorted->host, 0, mPolymerSystemSorted->nBytes );
    #endif

    /* calculate offsets to each aligned subgroup vector */
    iSubGroupOffset.resize( mnElementsInGroup.size() );
    iSubGroupOffset.at(0) = 0;
    for ( size_t i = 1u; i < mnElementsInGroup.size(); ++i )
    {
        iSubGroupOffset[i] = iSubGroupOffset[i-1] +
        ceilDiv( mnElementsInGroup[i-1], nElementsAlignment ) * nElementsAlignment;
        assert( iSubGroupOffset[i] - iSubGroupOffset[i-1] >= mnElementsInGroup[i-1] );
    }

    /* virtually sort groups into new array and save index mappings */
    iToiNew.resize( nAllMonomers   , UINT32_MAX );
    iNewToi.resize( nMonomersPadded, UINT32_MAX );
    std::vector< size_t > iSubGroup = iSubGroupOffset;   /* stores the next free index for each subgroup */
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        iToiNew[i] = iSubGroup[ mGroupIds[i] ]++;
        iNewToi[ iToiNew[i] ] = i;
    }

    
    /* adjust neighbor IDs to new sorted PolymerSystem and also sort that array.
     * Bonds are not supposed to change, therefore we don't need to push and
     * pop them each time we do something on the GPU! */
    assert( mNeighborsSorted == NULL );
    assert( mNeighborsSortedInfo.getRequiredBytes() == 0 );
    for ( size_t i = 0u; i < mnElementsInGroup.size(); ++i )
        mNeighborsSortedInfo.newMatrix( MAX_CONNECTIVITY, mnElementsInGroup[i] );
    for ( size_t i = 0u; i < mnElementsInGroup.size(); ++i )
	std::cout << "mNeighborsSortedInfo.getMatrixPitchElements( species ):" << i << " " << mNeighborsSortedInfo.getMatrixPitchElements( i ) <<std::endl;
    mNeighborsSorted = new MirroredVector< uint32_t >( mNeighborsSortedInfo.getRequiredElements(), mStream );
    std::memset( mNeighborsSorted->host, 0, mNeighborsSorted->nBytes );
    mNeighborsSortedSizes = new MirroredVector< uint8_t >( nMonomersPadded, mStream );
    std::memset( mNeighborsSortedSizes->host, 0, mNeighborsSortedSizes->nBytes );
    mMonomerStructureTag = new MirroredVector< uint8_t >( nMonomersPadded, mStream );
    std::memset( mMonomerStructureTag->host, 0, mMonomerStructureTag->nBytes );
	
    if (RingSlidingON)
    {
	uint32_t mMatrixOffsetElementsZero( mNeighborsSortedInfo.getMatrixOffsetElements( 0 ) );
	uint32_t mMatrixOffsetElementsOne ( mNeighborsSortedInfo.getMatrixOffsetElements( 1 ) );
	uint32_t mMatrixOffsetElementsTwo ( mNeighborsSortedInfo.getMatrixOffsetElements( 2 ) );
      
	uint32_t miSubGroupOffsetZero( iSubGroupOffset[ 0 ] );
	uint32_t miSubGroupOffsetOne ( iSubGroupOffset[ 1 ] );
	uint32_t miSubGroupOffsetTwo ( iSubGroupOffset[ 2 ] );
	std::cout << "mMatrixOffsetElementsZero: " << mMatrixOffsetElementsZero << " miSubGroupOffsetZero: " << miSubGroupOffsetZero << "\n";
	std::cout << "mMatrixOffsetElementsOne : " << mMatrixOffsetElementsOne  << " miSubGroupOffsetOne : " << miSubGroupOffsetOne  << "\n";
	std::cout << "mMatrixOffsetElementsTwo : " << mMatrixOffsetElementsTwo  << " miSubGroupOffsetTwo : " << miSubGroupOffsetTwo  << "\n";
	
	if ( mNeighborsSortedInfo.getMatrixPitchElements( 0 ) != mNeighborsSortedInfo.getMatrixPitchElements( 1 ) )
	{
	    std::stringstream msg;
	    msg << "[" << __FILENAME__ << "::initialize] "
		<< " pitch element for species 0 " << mNeighborsSortedInfo.getMatrixPitchElements( 0 ) 
		<< " pitch element for species 1 " << mNeighborsSortedInfo.getMatrixPitchElements( 1 ) << ")\n";
	    throw std::runtime_error( msg.str() );
	} 
	uint32_t mChainPitch(mNeighborsSortedInfo.getMatrixPitchElements( 0 ));
	CUDA_ERROR( cudaMemcpyToSymbol( dMatrixOffsetElementsZero  , &mMatrixOffsetElementsZero, sizeof( mMatrixOffsetElementsZero ) ) );
	CUDA_ERROR( cudaMemcpyToSymbol( dMatrixOffsetElementsOne   , &mMatrixOffsetElementsOne , sizeof( mMatrixOffsetElementsOne  ) ) );
	CUDA_ERROR( cudaMemcpyToSymbol( dMatrixOffsetElementsTwo   , &mMatrixOffsetElementsTwo , sizeof( mMatrixOffsetElementsTwo  ) ) );
	
	CUDA_ERROR( cudaMemcpyToSymbol( diSubGroupOffsetZero, &miSubGroupOffsetZero, sizeof( miSubGroupOffsetZero ) ) );
	CUDA_ERROR( cudaMemcpyToSymbol( diSubGroupOffsetOne , &miSubGroupOffsetOne , sizeof( miSubGroupOffsetOne  ) ) );
	CUDA_ERROR( cudaMemcpyToSymbol( diSubGroupOffsetTwo , &miSubGroupOffsetTwo , sizeof( miSubGroupOffsetTwo ) ) );
	
	CUDA_ERROR( cudaMemcpyToSymbol( dChainPitch         , &mChainPitch       , sizeof( mChainPitch        ) ) );
    }
    {
        size_t iSpecies = 0u;
        /* iterate over sorted instead of unsorted array so that calculating
         * the current species we are working on is easier */
        for ( size_t i = 0u; i < iNewToi.size(); ++i )
        {
            /* check if we are already working on a new species */
            if ( iSpecies+1 < iSubGroupOffset.size() &&
                 i >= iSubGroupOffset[ iSpecies+1 ] )
            {
                ++iSpecies;
            }
            /* skip over padded indices */
            if ( iNewToi[i] >= nAllMonomers )
                continue;
	    /** hand over the tag for the new monomer IDs
	     **/
	    if (RingSlidingON)
	    {
		if ( ( (iNewToi[i] % nMonomersPerChain) == 0 || ( (iNewToi[i] % nMonomersPerChain) + 1 ) == (nMonomersPerChain) ) 
		    && (iNewToi[i] < nMonomersPerChain*nChains)
		  ) mMonomerStructureTag->host[i] = 2;
		else if ( iNewToi[i] >= nMonomersPerChain*nChains) mMonomerStructureTag->host[i] = 2;
		else mMonomerStructureTag->host[i] = 3;
	    }
	    /* actually to the sorting / copying and conversion */
            mNeighborsSortedSizes->host[i] = mNeighbors[ iNewToi[i] ].size;
            auto const pitch = mNeighborsSortedInfo.getMatrixPitchElements( iSpecies );
            for ( size_t j = 0u; j < mNeighbors[  iNewToi[i] ].size; ++j )
                mNeighborsSorted->host[ mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) + j * pitch + ( i - iSubGroupOffset[ iSpecies ] ) ] = iToiNew[ mNeighbors[ iNewToi[i] ].neighborIds[j] ];
        }
    }

    mMonomerStructureTag->pushAsync();
    mNeighborsSorted->pushAsync();
    mNeighborsSortedSizes->pushAsync();

    /************************** end of group sorting **************************/

    /* sort groups into new array and save index mappings */
    mLog( "Info" ) << "[UpdaterGPUScBFMGPR_AB_Type::runSimulationOnGPU] sort mPolymerSystem -> mPolymerSystemSorted ... ";
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        if ( i < 20 )
            mLog( "Info" ) << "Write " << i << " to " << this->iToiNew[i] << "\n";
        auto const pTarget = mPolymerSystemSorted->host + iToiNew[i];
        pTarget->x = mPolymerSystem[ 4*i+0 ];
        pTarget->y = mPolymerSystem[ 4*i+1 ];
        pTarget->z = mPolymerSystem[ 4*i+2 ];
        pTarget->w = mNeighbors[i].size;
    }
    mPolymerSystemSorted->pushAsync();

    checkSystem();

    /* creating lattice */
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &mBoxXM1   , sizeof( mBoxXM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &mBoxYM1   , sizeof( mBoxYM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &mBoxZM1   , sizeof( mBoxZM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &mBoxXLog2 , sizeof( mBoxXLog2  ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &mBoxXYLog2, sizeof( mBoxXYLog2 ) ) );

    mLatticeOut = new MirroredTexture< uint8_t >( mBoxX * mBoxY * mBoxZ, mStream );
    mLatticeTmp = new MirroredTexture< uint8_t >( mBoxX * mBoxY * mBoxZ, mStream );
    CUDA_ERROR( cudaMemsetAsync( mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream ) );
    /* populate latticeOut with monomers from mPolymerSystem */
    std::memset( mLatticeOut->host, 0, mLatticeOut->nBytes );
    for ( uint32_t t = 0; t < nAllMonomers; ++t )
    {
        mLatticeOut->host[ linearizeBoxVectorIndex( mPolymerSystem[ 4*t+0 ],
                                                    mPolymerSystem[ 4*t+1 ],
                                                    mPolymerSystem[ 4*t+2 ] ) ] = 1;
    }
    mLatticeOut->pushAsync();

    CUDA_ERROR( cudaGetDevice( &miGpuToUse ) );
    CUDA_ERROR( cudaGetDeviceProperties( &mCudaProps, miGpuToUse ) );


}
void UpdaterGPUScBFMGPR_AB_Type::setEVON(bool excludedVolumeON_){excludedVolumeON=excludedVolumeON_;}
void UpdaterGPUScBFMGPR_AB_Type::setRingSliding(bool RingSlidingON_){RingSlidingON=RingSlidingON_;}
void UpdaterGPUScBFMGPR_AB_Type::setDiagonalMovesON(bool DiagonalMovesON_){DiagonalMovesON=DiagonalMovesON_;}
/* Set the frequently used constants for the shear force
 * prob = expf(-shearForce_d*prefactorPot*dx);
 **/
void UpdaterGPUScBFMGPR_AB_Type::setForce(double shearForce)
{ 
    //!frequently used constant to check monomer positions  
    uint32_t mBoxZ_HalfP1((mBoxZ>>1) +1);
    uint32_t mBoxZ_HalfM3((mBoxZ>>1) -3);
    uint32_t mBoxZM2     ( mBoxZ -2    );
    double tmp_LookUpShearForce[2] = { expf(-shearForce), expf( shearForce) };
    CUDA_ERROR( cudaMemcpyToSymbol( dLookUpShearForce, tmp_LookUpShearForce, 2*sizeof( tmp_LookUpShearForce ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ_HalfP1, &mBoxZ_HalfP1, sizeof( mBoxZ_HalfP1 ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ_HalfM3, &mBoxZ_HalfM3, sizeof( mBoxZ_HalfM3 ) ) );  
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM2     , &mBoxZM2     , sizeof( mBoxZM2      ) ) );  
}

void UpdaterGPUScBFMGPR_AB_Type::copyBondSet
( int dx, int dy, int dz, bool bondForbidden )
{
    mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;
}

void UpdaterGPUScBFMGPR_AB_Type::setNrOfAllMonomers( uint32_t const rnAllMonomers )
{
    if ( this->nAllMonomers != 0 || mAttributeSystem != NULL ||
         mPolymerSystemSorted != NULL || mNeighborsSorted != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] "
            << "Number of Monomers already set to " << nAllMonomers << "!\n"
            << "Or some arrays were already allocated "
            << "(mAttributeSystem=" << (void*) mAttributeSystem
            << ", mPolymerSystemSorted" << (void*) mPolymerSystemSorted
            << ", mNeighborsSorted" << (void*) mNeighborsSorted << ")\n";
        throw std::runtime_error( msg.str() );
    }

    this->nAllMonomers = rnAllMonomers;
    mAttributeSystem = new int32_t[ nAllMonomers ];
    if ( mAttributeSystem == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] mAttributeSystem is still NULL after call to 'new int32_t[ " << nAllMonomers << " ]!\n";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }
    mPolymerSystem .resize( nAllMonomers*4 );
    mNeighbors     .resize( nAllMonomers   );
    std::memset( &mNeighbors[0], 0, mNeighbors.size() * sizeof( mNeighbors[0] ) );
}
void UpdaterGPUScBFMGPR_AB_Type::setNrOfRingMonomers( uint32_t const rnRingMonomers )
{
    if ( this->nRingMonomers != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfRingMonomers] "
            << "Number of rings already set to " << nRingMonomers << "!\n";
        throw std::runtime_error( msg.str() );
    }
    this->nRingMonomers = rnRingMonomers;
    mNeighborsRings.resize( nRingMonomers   );
    std::memset( &mNeighborsRings[0], 0, mNeighborsRings.size() * sizeof( mNeighborsRings[0] ) );
}
void UpdaterGPUScBFMGPR_AB_Type::setNrOfChains( uint32_t const rnChains )
{
    if ( this->nChains != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfChains] "
            << "Number of chains already set to " << nChains << "!\n";
        throw std::runtime_error( msg.str() );
    }
    this->nChains = rnChains;
}
void UpdaterGPUScBFMGPR_AB_Type::setNrOfMonomersPerChain( uint32_t const rnMonomersPerChain )
{
    if ( this->nMonomersPerChain != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfMonomersPerChain] "
            << "Number of Monomers per chain already set to " << nMonomersPerChain << "!\n";
        throw std::runtime_error( msg.str() );
    }
    this->nMonomersPerChain = rnMonomersPerChain;
}
void UpdaterGPUScBFMGPR_AB_Type::setPeriodicity
(
    bool const isPeriodicX,
    bool const isPeriodicY,
    bool const isPeriodicZ
)
{
  mBoxXIsPeriodic = isPeriodicX; 
  mBoxYIsPeriodic = isPeriodicY; 
  mBoxZIsPeriodic = isPeriodicZ; 
}

void UpdaterGPUScBFMGPR_AB_Type::setAttribute( uint32_t i, int32_t attribute )
{
    mAttributeSystem[i] = attribute;
}

void UpdaterGPUScBFMGPR_AB_Type::setMonomerCoordinates
(
    uint32_t const i,
    int32_t  const x,
    int32_t  const y,
    int32_t  const z
)
{
    mPolymerSystem.at( 4*i+0 ) = x;
    mPolymerSystem.at( 4*i+1 ) = y;
    mPolymerSystem.at( 4*i+2 ) = z;
}

int32_t UpdaterGPUScBFMGPR_AB_Type::getMonomerPositionInX( uint32_t i ){ return mPolymerSystem[ 4*i+0 ]; }
int32_t UpdaterGPUScBFMGPR_AB_Type::getMonomerPositionInY( uint32_t i ){ return mPolymerSystem[ 4*i+1 ]; }
int32_t UpdaterGPUScBFMGPR_AB_Type::getMonomerPositionInZ( uint32_t i ){ return mPolymerSystem[ 4*i+2 ]; }

void UpdaterGPUScBFMGPR_AB_Type::setConnectivity
(
    uint32_t const iMonomer1,
    uint32_t const iMonomer2
)
{
    /* @todo add check whether the bond already exists */
    /* Could also add the inversio, but the bonds are a non-directional graph */
    auto const iNew = mNeighbors[ iMonomer1 ].size++;
    if ( iNew > MAX_CONNECTIVITY-1 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setConnectivity" << "] "
            << "The maximum amount of bonds per monomer (" << MAX_CONNECTIVITY
            << ") has been exceeded!\n";
        throw std::invalid_argument( msg.str() );
    }
    mNeighbors[ iMonomer1 ].neighborIds[ iNew ] = iMonomer2;
}

uint32_t UpdaterGPUScBFMGPR_AB_Type::getConnectivity
(
    uint32_t const iMonomer,
    uint32_t const iNeighbor
)
{
    if ( iNeighbor > MAX_CONNECTIVITY-1 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::getConnectivity" << "] "
            << "The maximum amount of bonds per monomer (" << MAX_CONNECTIVITY
            << ") has been exceeded!\n";
        throw std::invalid_argument( msg.str() );
    }
    return mNeighbors[ iMonomer ].neighborIds[ iNeighbor ];
}

uint32_t UpdaterGPUScBFMGPR_AB_Type::getMaximumConnectivity
(
    uint32_t const iMonomer
)
{
    if ( iMonomer > nAllMonomers )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::getMaximumConnectivity" << "] "
            << "The monomer index " << iMonomer
            << " exceedes the maximum number " << nAllMonomers
            << " of all monomers!\n";
        throw std::invalid_argument( msg.str() );
    }
    return mNeighbors[ iMonomer ].size;
}


void UpdaterGPUScBFMGPR_AB_Type::setRingConnectivity
(
    uint32_t const iMonomer1,
    uint32_t const iMonomer2
)
{
    /* @todo add check whether the bond already exists */
    /* Could also add the inversio, but the bonds are a non-directional graph */
    auto const iNew = mNeighborsRings[ iMonomer1 ].size++;
    if ( iNew > MAX_CONNECTIVITY-1 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setRingConnectivity" << "] "
            << "The maximum amount of bonds per monomer (" << MAX_CONNECTIVITY
            << ") has been exceeded for monomer" << iMonomer1
	    << "!\n ";
        throw std::invalid_argument( msg.str() );
    }
    mNeighborsRings[ iMonomer1 ].neighborIds[ iNew ] = iMonomer2;
}
void UpdaterGPUScBFMGPR_AB_Type::setLatticeSize
(
    uint32_t const boxX,
    uint32_t const boxY,
    uint32_t const boxZ
)
{
    if ( mBoxX == boxX && mBoxY == boxY && mBoxZ == boxZ )
        return;

    if ( ! ( inRange< intCUDA >( boxX ) &&
             inRange< intCUDA >( boxY ) &&
             inRange< intCUDA >( boxZ )    ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "The box size (" << boxX << "," << boxY << "," << boxZ
            << ") is larger than the internal integer data type for "
            << "representing positions allow! (" << std::numeric_limits< intCUDA >::min()
            << " <= size <= " << std::numeric_limits< intCUDA >::max() << ")";
        throw std::invalid_argument( msg.str() );
    }

    mBoxX   = boxX;
    mBoxY   = boxY;
    mBoxZ   = boxZ;
    mBoxXM1 = boxX-1;
    mBoxYM1 = boxY-1;
    mBoxZM1 = boxZ-1;

    /* determine log2 for mBoxX and mBoxX * mBoxY to be used for bitshifting
     * the indice instead of multiplying ... WHY??? I don't think it is faster,
     * but much less readable */
    mBoxXLog2  = 0; uint32_t dummy = mBoxX; while ( dummy >>= 1 ) ++mBoxXLog2;
    mBoxXYLog2 = 0; dummy = mBoxX*mBoxY;    while ( dummy >>= 1 ) ++mBoxXYLog2;
    if ( mBoxX != ( 1u << mBoxXLog2 ) || mBoxX * boxY != ( 1u << mBoxXYLog2 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "Could not determine value for bit shift. "
            << "Check whether the box size is a power of 2! ( "
            << "boxX=" << mBoxX << " =? 2^" << mBoxXLog2 << " = " << ( 1 << mBoxXLog2 )
            << ", boxX*boY=" << mBoxX * mBoxY << " =? 2^" << mBoxXYLog2
            << " = " << ( 1 << mBoxXYLog2 ) << " )\n";
        throw std::runtime_error( msg.str() );
    }

    if ( mLattice != NULL )
        delete[] mLattice;
    mLattice = new uint8_t[ mBoxX * mBoxY * mBoxZ ];
    std::memset( (void *) mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
}

void UpdaterGPUScBFMGPR_AB_Type::populateLattice()
{
    std::memset( mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
    for ( size_t i = 0; i < nAllMonomers; ++i )
    {
        auto const j = linearizeBoxVectorIndex( mPolymerSystem[ 4*i+0 ],
                                                mPolymerSystem[ 4*i+1 ],
                                                mPolymerSystem[ 4*i+2 ] );
        if ( j >= mBoxX * mBoxY * mBoxZ )
        {
            std::stringstream msg;
            msg
            << "[populateLattice] " << i << " -> ("
            << mPolymerSystem[ 4*i+0 ] << ","
            << mPolymerSystem[ 4*i+1 ] << ","
            << mPolymerSystem[ 4*i+2 ] << ") -> " << j << " is out of range "
            << "of (" << mBoxX << "," << mBoxY << "," << mBoxZ << ") = "
            << mBoxX * mBoxY * mBoxZ << "\n";
            throw std::runtime_error( msg.str() );
        }
        mLattice[ j ] = 1;
    }
}


void UpdaterGPUScBFMGPR_AB_Type::runSimulationOnGPU
(
    int32_t const nMonteCarloSteps
)
{
    std::clock_t const t0 = std::clock();

    auto const nSpecies = mnElementsInGroup.size();

    /**
     * Statistics (min, max, mean, stddev) on filtering. Filtered because of:
     *   0: bonds, 1: volume exclusion, 2: volume exclusion (parallel)
     * These statistics are done for each group separately
     */
    std::vector< std::vector< double > > sums, sums2, mins, maxs, ns;
    std::vector< unsigned long long int > vFiltered;
    unsigned long long int * dpFiltered = NULL;
    auto constexpr nFilters = 5;
    if ( mLog.isActive( "Stats" ) )
    {
        sums .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        sums2.resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        mins .resize( nSpecies, std::vector< double >( nFilters, nAllMonomers ) );
        maxs .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        ns   .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        /* ns needed because we need to know how often each group was advanced */
        vFiltered.resize( nFilters );
        CUDA_ERROR( cudaMalloc( &dpFiltered, nFilters * sizeof( *dpFiltered ) ) );
        CUDA_ERROR( cudaMemsetAsync( (void*) dpFiltered, 0, nFilters * sizeof( *dpFiltered ), mStream ) );
    }

    /**
     * Logic for determining the best threadsPerBlock configuration
     *
     * This might be dependent on the species, therefore for each species
     * store the current best thread count and all timings.
     * As the cudaEventSynchronize timings are expensive, stop benchmarking
     * after having found the best configuration.
     * Only try out power two multiples of warpSize up to maxThreadsPerBlock,
     * e.g. 32, 64, 128, 256, 512, 1024, because smaller than warp size
     * should never lead to a speedup and non power of twos, e.g. 196 even,
     * won't be able to perfectly fill out the shared multi processor.
     * Also, automatically determine whether cudaMemset is faster or not (after
     * we found the best threads per block configuration
     * note: test example best configuration was 128 threads per block and use
     *       the cudaMemset version instead of the third kernel
     */
    std::vector< int > vnThreadsToTry;
    for ( auto nThreads = mCudaProps.warpSize; nThreads <= mCudaProps.maxThreadsPerBlock; nThreads *= 2 )
        vnThreadsToTry.push_back( nThreads );
    assert( vnThreadsToTry.size() > 0 );
    struct SpeciesBenchmarkData
    {
        /* 2 vectors of double for measurements with and without cudaMemset */
        std::vector< std::vector< float > > timings;
        std::vector< std::vector< int   > > nRepeatTimings;
        int iBestThreadCount;
        bool useCudaMemset;
        bool decidedAboutThreadCount;
        bool decidedAboutCudaMemset;
    };
    std::vector< SpeciesBenchmarkData > benchmarkInfo( nSpecies, SpeciesBenchmarkData{
        std::vector< std::vector< float > >( 2 /* true or false */,
            std::vector< float >( vnThreadsToTry.size(),
                std::numeric_limits< float >::infinity() ) ),
        std::vector< std::vector< int   > >( 2 /* true or false */,
            std::vector< int   >( vnThreadsToTry.size(),
            2 /* repeat 2 time, i.e. execute three times */ ) ),
#ifdef AUTO_CONFIGURE_BEST_SETTINGS_FOR_PSCBFM_ALGORITHM
        0, true, vnThreadsToTry.size() <= 1, false
#else
        2, true, true, true
#endif
    } );
    cudaEvent_t tOneGpuLoop0, tOneGpuLoop1;
    cudaEventCreate( &tOneGpuLoop0 );
    cudaEventCreate( &tOneGpuLoop1 );

    cudaEvent_t tGpu0, tGpu1;
    if ( mLog.isActive( "Benchmark" ) )
    {
        cudaEventCreate( &tGpu0 );
        cudaEventCreate( &tGpu1 );
        cudaEventRecord( tGpu0, mStream );
    }

    BoxCheck boxcheckmethod(mBoxXIsPeriodic, mBoxYIsPeriodic, mBoxZIsPeriodic);
    LocalMetropolis<ShearForce> MyMetropolisCheck(true);
    Move MyMove( DiagonalMovesON );
    
    /* draw seed for the interval */
    auto const seed     = randomNumbers.r250_rand32();
    /* run simulation */
    for ( int32_t iStep = 1; iStep <= nMonteCarloSteps; ++iStep )
    {
        /* one Monte-Carlo step:
         *  - tries to move on average all particles one time
         *  - each particle could be touched, not just one group */
        for ( uint32_t iSubStep = 0; iSubStep < nSpecies; ++iSubStep )
        {
            /* randomly choose which monomer group to advance */
            auto const iSpecies = randomNumbers.r250_rand32() % nSpecies;
            auto const nThreads = vnThreadsToTry.at( benchmarkInfo[ iSpecies ].iBestThreadCount );
            auto const nBlocks  = ceilDiv( mnElementsInGroup[ iSpecies ], nThreads );
            auto const needToBenchmark = ! (
                benchmarkInfo[ iSpecies ].decidedAboutThreadCount &&
                benchmarkInfo[ iSpecies ].decidedAboutCudaMemset );
            auto const useCudaMemset = benchmarkInfo[ iSpecies ].useCudaMemset;
            if ( needToBenchmark )
                cudaEventRecord( tOneGpuLoop0, mStream );

	    if ( excludedVolumeON )
	    {
		kernelSimulationScBFMCheckSpecies<Rngs::Saru>
		  <<< nBlocks, nThreads, 0, mStream >>>(
		      MyMove,
		      MyMetropolisCheck,
		      boxcheckmethod,
		      mPolymerSystemSorted->gpu,
		      mPolymerFlags->gpu,
		      iSubGroupOffset[ iSpecies ],
		      mLatticeTmp->gpu,
		      mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
		      mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
		      mNeighborsSortedSizes->gpu,
		      mnElementsInGroup[ iSpecies ], 
		      seed,
		      mGlobalIterator,
		      NULL,
		      mLatticeOut->texture
		  );
		  mGlobalIterator++;

		  if ( useCudaMemset )
		  {
		      kernelSimulationScBFMPerformSpeciesAndApply
		      <<< nBlocks, nThreads, 0, mStream >>>(
			  MyMove,
			  mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
			  mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
			  mLatticeOut->gpu,
			  mnElementsInGroup[ iSpecies ],
			  mLatticeTmp->texture
		      );
		  }
		  else
		  {
		      kernelSimulationScBFMPerformSpecies
		      <<< nBlocks, nThreads, 0, mStream >>>(
			  MyMove,
			  mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
			  mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
			  mLatticeOut->gpu,
			  mnElementsInGroup[ iSpecies ],
			  mLatticeTmp->texture
		      );
		  }

		  if ( useCudaMemset )
		  {
		      #ifdef USE_THRUST_FILL
			  thrust::fill( thrust::system::cuda::par, (uint64_t*)  mLatticeTmp->gpu,
					(uint64_t*)( mLatticeTmp->gpu + mLatticeTmp->nElements ), 0 );
		      #else
			  #ifdef USE_BIT_PACKING_TMP_LATTICE
			      cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes / CHAR_BIT, mStream );
			  #else
			      cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream );
			  #endif
		      #endif
		  }
		  else
		  {
		      std::cout << "Start simulation kernelSimulationScBFMZeroArraySpecies " << std::endl;
		      kernelSimulationScBFMZeroArraySpecies
		      <<< nBlocks, nThreads, 0, mStream >>>(
			  MyMove,
			  mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
			  mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
			  mLatticeTmp->gpu,
			  mnElementsInGroup[ iSpecies ]
		      );
		  }
	    }
	    else //no excluded volume 
	    {
		  kernelSimulationScBFMwoEVCheckPerformApplySpecies<Rngs::Saru>
		  <<< nBlocks, nThreads, 0, mStream >>>(
		      MyMove,
		      MyMetropolisCheck,
		      boxcheckmethod,
		      mPolymerSystemSorted->gpu,
		      iSubGroupOffset[ iSpecies ],
		      mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
		      mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
		      mNeighborsSortedSizes->gpu,
		      mnElementsInGroup[ iSpecies ],
		      seed,
		      mGlobalIterator,
		      NULL
		  );
		  mGlobalIterator++;
	    }
	      
            if ( needToBenchmark )
            {
                auto & info = benchmarkInfo[ iSpecies ];
                cudaEventRecord( tOneGpuLoop1, mStream );
                cudaEventSynchronize( tOneGpuLoop1 );
                float milliseconds = 0;
                cudaEventElapsedTime( & milliseconds, tOneGpuLoop0, tOneGpuLoop1 );
                auto const iThreadCount = info.iBestThreadCount;
                auto & oldTiming = info.timings.at( useCudaMemset ).at( iThreadCount );
                oldTiming = std::min( oldTiming, milliseconds );

                mLog( "Info" )
                << "Using " << nThreads << " threads (" << nBlocks << " blocks)"
                << " and using " << ( useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
                << " for species " << iSpecies << " took " << milliseconds << "ms\n";

                auto & nStillToRepeat = info.nRepeatTimings.at( useCudaMemset ).at( iThreadCount );
                if ( nStillToRepeat > 0 )
                    --nStillToRepeat;
                else if ( ! info.decidedAboutThreadCount )
                {
                    /* if not the first timing, then decide whether we got slower,
                     * i.e. whether we found the minimum in the last step and
                     * have to roll back */
                    if ( iThreadCount > 0 )
                    {
                        if ( info.timings.at( useCudaMemset ).at( iThreadCount-1 ) < milliseconds )
                        {
                            --info.iBestThreadCount;
                            info.decidedAboutThreadCount = true;
                        }
                        else
                            ++info.iBestThreadCount;
                    }
                    else
                        ++info.iBestThreadCount;
                    /* if we can't increment anymore, then we are finished */
                    assert( (size_t) info.iBestThreadCount <= vnThreadsToTry.size() );
                    if ( (size_t) info.iBestThreadCount == vnThreadsToTry.size() )
                    {
                        --info.iBestThreadCount;
                        info.decidedAboutThreadCount = true;
                    }
                    if ( info.decidedAboutThreadCount )
                    {
                        /* then in the next term try out changing cudaMemset
                         * version to custom kernel version (or vice-versa) */
                        if ( ! info.decidedAboutCudaMemset )
                            info.useCudaMemset = ! info.useCudaMemset;
                        mLog( "Info" )
                        << "Using " << vnThreadsToTry.at( info.iBestThreadCount )
                        << " threads per block is the fastest for species "
                        << iSpecies << ".\n";
                    }
                }
                else if ( ! info.decidedAboutCudaMemset )
                {
                    info.decidedAboutCudaMemset = true;
                    if ( info.timings.at( ! useCudaMemset ).at( iThreadCount ) < milliseconds )
                        info.useCudaMemset = ! useCudaMemset;
                    if ( info.decidedAboutCudaMemset )
                    {
                        mLog( "Info" )
                        << "Using " << ( info.useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
                        << " is the fastest for species " << iSpecies << ".\n";
                    }
                }
            }
        }
	mPolymerSystemSorted->pop( false ); // sync
       
	// the ring is allowed to slide along the chain backbone
        // try to move the bond from the ring to the chain to a neighbor of the chain monomer
        if(RingSlidingON)
	{
	      for ( uint32_t iSubStep = 2; iSubStep < nSpecies; ++iSubStep )
	      {
		  auto const iSpecies = iSubStep;
		  auto const nThreads = vnThreadsToTry.at( benchmarkInfo[ iSpecies ].iBestThreadCount );
		  auto const nBlocks  = ceilDiv( mnElementsInGroup[ iSpecies ], nThreads );
		  
		  
		  kernelMoveRingBond<Rngs::Saru>
		  <<< nBlocks, nThreads, 0, mStream >>>(
		      mPolymerSystemSorted->gpu,
		      mMonomerStructureTag->gpu ,
		      iSubGroupOffset[ iSpecies ],
		      mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
		      mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
		      mNeighborsSortedSizes->gpu,
		      mNeighborsSorted->gpu + 0 ,
		      mnElementsInGroup[ iSpecies ], seed,
		      mGlobalIterator,NULL
		  );
		  mGlobalIterator++;
	      } // iSubstep
	}
    } // iStep
    
    if ( mLog.isActive( "Benchmark" ) )
    {
        // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/#disqus_thread
        cudaEventRecord( tGpu1, mStream );
        cudaEventSynchronize( tGpu1 );  // basically a StreamSynchronize
        float milliseconds = 0;
        cudaEventElapsedTime( & milliseconds, tGpu0, tGpu1 );
        std::stringstream sBuffered;
        sBuffered << "tGpuLoop = " << milliseconds / 1000. << "s\n";
        mLog( "Benchmark" ) << sBuffered.str();
    }

    mtCopyBack0 = std::chrono::high_resolution_clock::now();

    /* copy into mPolymerSystem and drop the property tag while doing so.
     * would be easier and probably more efficient if mPolymerSystem->gpu/host
     * would be a struct of arrays instead of an array of structs !!! */
    mPolymerSystemSorted->pop( false ); // sync
    /*  copy into */
    mNeighborsSorted->pop( false ); //sync
    mNeighborsSortedSizes->pop( false ); //sync
    
    if ( mLog.isActive( "Benchmark" ) )
    {
        std::clock_t const t1 = std::clock();
        double const dt = float(t1-t0) / CLOCKS_PER_SEC;
        mLog( "Benchmark" )
        << "run time (GPU): " << nMonteCarloSteps << "\n"
        << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
        << nMonteCarloSteps * ( nAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    }

    /* untangle reordered array so that LeMonADE can use it again */
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        auto const pTarget = mPolymerSystemSorted->host + iToiNew[i];
        if ( i < 10 )
            mLog( "Info" ) << "Copying back " << i << " from " << iToiNew[i] << "\n";
        mPolymerSystem[ 4*i+0 ] = pTarget->x;
        mPolymerSystem[ 4*i+1 ] = pTarget->y;
        mPolymerSystem[ 4*i+2 ] = pTarget->z;
        mPolymerSystem[ 4*i+3 ] = pTarget->w;
    }
    /* write the new bonds into mNeighbors for further use in LeMonADE*/
    if (RingSlidingON)
    {
	size_t iSpecies = 0u;
	/* iterate over sorted instead of unsorted array so that calculating
	  * the current species we are working on is easier */
	for ( size_t i = 0u; i < iNewToi.size(); ++i )
	{
	    /* check if we are already working on a new species */
	    if ( iSpecies+1 < iSubGroupOffset.size() &&
		  i >= iSubGroupOffset[ iSpecies+1 ] )
	    {
		++iSpecies;
	    }
	    /* skip over padded indices */
	    if ( iNewToi[i] >= nAllMonomers )
		continue;
	    /* actually to the sorting / copying and conversion */
	    auto const pitch = mNeighborsSortedInfo.getMatrixPitchElements( iSpecies );
	    mNeighbors[ iNewToi[i] ].size = mNeighborsSortedSizes->host[i];
   
	    for ( size_t j = 0u; j < mNeighborsSortedSizes->host[i]; ++j )
	    {
		std::stringstream outMoveState;  
		bool writeout(false);
		if ( mLog.isActive( "Check" ) && nChains*nMonomersPerChain <= iNewToi[i] ) 
		{
		  outMoveState << "NEW: " << i << " Old: " << iNewToi[i] << " Bond ID:" << j << " OldNeigh: "  << mNeighbors[ iNewToi[i] ].neighborIds[j];
		  writeout=true;
		}
		  
		mNeighbors[ iNewToi[i] ].neighborIds[j] = iNewToi[ mNeighborsSorted->host[ mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) + j * pitch + ( i - iSubGroupOffset[ iSpecies ] ) ] ];
		
		if ( writeout)  
		{
		    outMoveState  << " NewNeigh: " << mNeighbors[ iNewToi[i] ].neighborIds[j] 
				  << " mNeighSo: " << mNeighborsSorted->host[ mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) + j * pitch + ( i - iSubGroupOffset[ iSpecies ] ) ]
				  << " \n";
		    mLog( "Check" ) << outMoveState.str(); 
		}
	    }
	}
    }
    
    checkSystem(); // no-op if "Check"-level deactivated
}

/**
 * Checks for excluded volume condition and for correctness of all monomer bonds
 * Beware, it useses and thereby thrashes mLattice. Might be cleaner to declare
 * as const and malloc and free some temporary buffer, but the time ...
 * https://randomascii.wordpress.com/2014/12/10/hidden-costs-of-memory-allocation/
 * "In my tests, for sizes ranging from 8 MB to 32 MB, the cost for a new[]/delete[] pair averaged about 7.5 μs (microseconds), split into ~5.0 μs for the allocation and ~2.5 μs for the free."
 *  => ~40k cycles
 */
void UpdaterGPUScBFMGPR_AB_Type::checkSystem()
{
    if ( ! mLog.isActive( "Check" ) )
        return;

    if ( mLattice == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkSystem" << "] "
            << "mLattice is not allocated. You need to call "
            << "setNrOfAllMonomers and initialize before calling checkSystem!\n";
        mLog( "Error" ) << msg.str();
        throw std::invalid_argument( msg.str() );
    }

    /**
     * Test for excluded volume by setting all lattice points and count the
     * toal lattice points occupied. If we have overlap this will be smaller
     * than calculated for zero overlap!
     * mPolymerSystem only stores the lower left front corner of the 2x2x2
     * monomer cube. Use that information to set all 8 cells in the lattice
     * to 'occupied'
     */
    /*
     Lattice is an array of size Box_X*Box_Y*Box_Z. PolymerSystem holds the monomer positions which I strongly guess are supposed to be in the range 0<=x<Box_X. If I see correctly, then this part checks for excluded volume by occupying a 2x2x2 cube for each monomer in Lattice and then counting the total occupied cells and compare it to the theoretical value of nMonomers * 8. But Lattice seems to be too small for that kinda usage! I.e. for two particles, one being at x=0 and the other being at x=Box_X-1 this test should return that the excluded volume condition is not met! Therefore the effective box size is actually (Box_X-1,Box_X-1,Box_Z-1) which in my opinion should be a bug ??? */
    std::memset( mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        int32_t const & x = mPolymerSystem[ 4*i   ];
        int32_t const & y = mPolymerSystem[ 4*i+1 ];
        int32_t const & z = mPolymerSystem[ 4*i+2 ];
        /**
         * @verbatim
         *           ...+---+---+
         *     ...'''   | 6 | 7 |
         *    +---+---+ +---+---+    y
         *    | 2 | 3 | | 4 | 5 |    ^ z
         *    +---+---+ +---+---+    |/
         *    | 0 | 1 |   ...'''     +--> x
         *    +---+---+'''
         * @endverbatim
         */
        mLattice[ linearizeBoxVectorIndex( x  , y  , z   ) ] = 1; /* 0 */
        mLattice[ linearizeBoxVectorIndex( x+1, y  , z   ) ] = 1; /* 1 */
        mLattice[ linearizeBoxVectorIndex( x  , y+1, z   ) ] = 1; /* 2 */
        mLattice[ linearizeBoxVectorIndex( x+1, y+1, z   ) ] = 1; /* 3 */
        mLattice[ linearizeBoxVectorIndex( x  , y  , z+1 ) ] = 1; /* 4 */
        mLattice[ linearizeBoxVectorIndex( x+1, y  , z+1 ) ] = 1; /* 5 */
        mLattice[ linearizeBoxVectorIndex( x  , y+1, z+1 ) ] = 1; /* 6 */
        mLattice[ linearizeBoxVectorIndex( x+1, y+1, z+1 ) ] = 1; /* 7 */
    }
    /* check total occupied cells inside lattice to ensure that the above
     * transfer went without problems. Note that the number will be smaller
     * if some monomers overlap!
     * Could also simply reduce mLattice with +, I think, because it only
     * cotains 0 or 1 ??? */
    unsigned nOccupied = 0;
    for ( unsigned i = 0u; i < mBoxX * mBoxY * mBoxZ; ++i )
        nOccupied += mLattice[i] != 0;
    if ( ! ( nOccupied == nAllMonomers * 8 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::~checkSystem" << "] "
            << "Occupation count in mLattice is wrong! Expected 8*nMonomers="
            << 8 * nAllMonomers << " occupied cells, but got " << nOccupied;
        throw std::runtime_error( msg.str() );
    }

    /**
     * Check bonds i.e. that |dx|<=3 and whether it is allowed by the given
     * bond set
     */
    for ( unsigned i = 0; i < nAllMonomers; ++i )
    {
	if (RingSlidingON) 
	{
	  if (mNeighbors[i].size > mMonomerStructureTag->host[iToiNew[i]]) 
	  {
	    std::stringstream msg;
	    msg << "[" << __FILENAME__ << "::checkSystem] "
		<< "Exceeds the number of maximum bonds "
		<< mMonomerStructureTag->host[iToiNew[i]]
		<< ". The current number is "
		<< mNeighbors[i].size  << std::endl;
	    throw std::runtime_error( msg.str() );
	  }
	}
	for ( unsigned iNeighbor = 0; iNeighbor < mNeighbors[i].size; ++iNeighbor )
	{
	    /* calculate the bond vector between the neighbor and this particle
	    * neighbor - particle = ( dx, dy, dz ) */
	    intCUDA * const neighbor = & mPolymerSystem[ 4*mNeighbors[i].neighborIds[ iNeighbor ] ];
	    int32_t  dx = neighbor[0] - mPolymerSystem[ 4*i+0 ];
	    int32_t  dy = neighbor[1] - mPolymerSystem[ 4*i+1 ];
	    int32_t  dz = neighbor[2] - mPolymerSystem[ 4*i+2 ];
	    /*
	    * * -4 &rarr; (-4 &7) = 4
	    * * -3 &rarr; (-3 &7) = 5
	    * * -2 &rarr; (-2 &7) = 6
	    * * -1 &rarr; (-1 &7) = 7
	    * *  0 &rarr; ( 0 &7) = 0
	    * *  1 &rarr; ( 1 &7) = 1
	    * *  2 &rarr; ( 2 &7) = 2
	    * *  3 &rarr; ( 3 &7) = 3
	    * *  4 &rarr; ( 4 &7) = 4
	    */
	    dx = dx & 7;
	    dy = dy & 7;
	    dz = dz & 7;
	    int erroneousAxis = -1;
	    if ( ! ( 0 <= dx && dx <= 7 && dx != 4 ) ) erroneousAxis = 0;
	    if ( ! ( 0 <= dy && dy <= 7 && dy != 4 ) ) erroneousAxis = 1;
	    if ( ! ( 0 <= dz && dz <= 7 && dz != 4 ) ) erroneousAxis = 2;
// 	    if ( ! ( -3 <= dx && dx <= 3 ) ) erroneousAxis = 0;
// 	    if ( ! ( -3 <= dy && dy <= 3 ) ) erroneousAxis = 1;
// 	    if ( ! ( -3 <= dz && dz <= 3 ) ) erroneousAxis = 2;

	    if ( erroneousAxis >= 0 || mForbiddenBonds[ linearizeBondVectorIndex( dx, dy, dz ) ] )
	    {
		std::stringstream msg;
		msg << "[" << __FILENAME__ << "::checkSystem] ";
		if ( erroneousAxis > 0 )
		    msg << "Invalid " << 'X' + erroneousAxis << " Bond: ";
		if ( mForbiddenBonds[ linearizeBondVectorIndex( dx, dy, dz ) ] )
		    msg << "This particular bond is forbidden: ";
		msg << "(" << dx << "," << dy<< "," << dz << ") between monomer "
		    << i+1 << " at (" << mPolymerSystem[ 4*i+0 ] << ","
				      << mPolymerSystem[ 4*i+1 ] << ","
				      << mPolymerSystem[ 4*i+2 ] << ") and monomer "
		    << mNeighbors[i].neighborIds[ iNeighbor ]+1 << " at ("
		    << neighbor[0] << "," << neighbor[1] << "," << neighbor[2] << ")"
		    << std::endl;
		throw std::runtime_error( msg.str() );
	    }
	}
    }
}

/**
 * GPUScBFM_AB_Type::initialize and run and cleanup should be usable on
 * repeat. Which means we need to destruct everything created in
 * GPUScBFM_AB_Type::initialize, which encompasses setLatticeSize,
 * setNrOfAllMonomers and initialize. Currently this includes all allocs,
 * so we can simply call destruct.
 */
void UpdaterGPUScBFMGPR_AB_Type::cleanup()
{
    this->destruct();
    cudaDeviceSynchronize();
    cudaProfilerStop();
}
