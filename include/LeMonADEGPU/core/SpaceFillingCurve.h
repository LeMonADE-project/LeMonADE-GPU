/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo--GPU--oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADEGPU.

LeMonADE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LeMonADE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LeMonADE.  If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------*/
#ifndef LEMONADE_CORE_SPACE_FILLING_CURVE
#define LEMONADE_CORE_SPACE_FILLING_CURVE
// #include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Connection.h>

#include <extern/Fundamental/BitsCompileTime.hpp>
// #include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/utility/cudacommon.hpp>
// #include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>

/*****************************************************************************/
/**
 *@brief abstract template function for the space filling curve 
 */
/*****************************************************************************/

/** @todo  Check if this type could/should be the coordinate type 
 */
typedef uint64_t T_LinVector  ;  
  
template <class specializedCurve>
class AbstractSpaceFillingCurve
{

public:

template< typename T >
__host__ __device__ inline T_LinVector linearizeBoxVectorIndex
( T const & ix, T const & iy, T const & iz) const 
{ return static_cast<specializedCurve*>(this)->linearizeBoxVectorIndex(ix,iy,iz); }

template< typename T >
__host__ __device__ inline T_LinVector linearizeBoxVectorIndexX
( T const & ix ) const 
{ return static_cast<specializedCurve*>(this)->linearizeBoxVectorIndexX(ix);}

template< typename T >
__host__ __device__ inline T_LinVector linearizeBoxVectorIndexY
( T const & iy ) const 
{ return static_cast<specializedCurve*>(this)->linearizeBoxVectorIndexY(iy);}

template< typename T >
__host__ __device__ inline T_LinVector linearizeBoxVectorIndexZ
( T const & iz ) const 
{ return static_cast<specializedCurve*>(this)->linearizeBoxVectorIndexZ(iz);}

template < class IngredientsType >
void initialize(const IngredientsType& ing)
{ static_cast<specializedCurve*>(this)->initialize(ing);}

template < typename T >
void initialize(T mBoxX_, T mBoxY_, T mBoxZ_)
{ static_cast<specializedCurve*>(this)->initialize(mBoxX_,mBoxY_,mBoxZ_);} 
};


/*****************************************************************************/
/**
  * The z curve in 3D is:
  * @verbatim
  *   i -> bin  -> (z,y,x)
  *   0 -> 000b -> (0,0,0)
  *   1 -> 001b -> (0,0,1)
  *   2 -> 010b -> (0,1,0)
  *   3 -> 011b -> (0,1,1)
  *   4 -> 100b -> (1,0,0)
  *   5 -> 101b -> (1,0,1)
  *   6 -> 110b -> (1,1,0)
  *   7 -> 111b -> (1,1,1)
  *
  *       .'+---+---+
  *     .'  | 0 | 1 |       y
  *   .'    +---+---+       ^
  *  +---+---+2 | 3 |       |
  *  | 4 | 5 |--+---+       +--> x
  *  +---+---+    .'      .'
  *  | 6 | 7 |  .'       L z
  *  +---+---+.'
  * @endverbatim
  */
class ZOrderCurve:public AbstractSpaceFillingCurve<ZOrderCurve>
{
  
public:
  ZOrderCurve()
  :mBoxXM1(0),mBoxYM1(0),mBoxZM1(0){};
  template <class IngredientsType >
  void initialize(const IngredientsType& ing)
  {
    mBoxXM1 = ing.getBoxX()-1;
    mBoxYM1 = ing.getBoxY()-1;
    mBoxZM1 = ing.getBoxZ()-1;
  }
  template < typename T >
  void initialize(T mBoxX_, T mBoxY_, T mBoxZ_)
  {
    mBoxXM1 = mBoxX_-1;
    mBoxYM1 = mBoxY_-1;
    mBoxZM1 = mBoxZ_-1;
  }
  
  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexX
  ( T const & ix ) const
  {
    #ifdef __CUDA_ARCH__
	    return diluteBits< T_LinVector, 2 >( ix & dcBoxXM1 ) ;
    #else 
	    return diluteBits< T_LinVector, 2 >( T_LinVector( ix ) & mBoxXM1 ) ;
    #endif
  }
  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexY
  ( T const & iy ) const 
  {
  #ifdef __CUDA_ARCH__
	  return diluteBits< T_LinVector, 2 >( iy & dcBoxYM1 ) << 1 ;
  #else 
	  return diluteBits< T_LinVector, 2 >( T_LinVector( iy ) & mBoxYM1 ) << 1 ;
  #endif
  }

  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexZ
  ( T const & iz ) const 
  {
  #ifdef __CUDA_ARCH__
	  return diluteBits< T_LinVector, 2 >( iz & dcBoxZM1 ) << 2 ;
  #else 
	  return diluteBits< T_LinVector, 2 >( T_LinVector( iz ) & mBoxZM1 ) << 2 ;
  #endif
  }
  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz) const 
  { 
    return linearizeBoxVectorIndexX(ix) +
	   linearizeBoxVectorIndexY(iy) +
	   linearizeBoxVectorIndexZ(iz);
  }
private: 
  size_t mBoxXM1, mBoxYM1, mBoxZM1; 

};
/*****************************************************************************/
/**
 * @brief
 */
class LinearCurvePowOfTwo:public AbstractSpaceFillingCurve<LinearCurvePowOfTwo>{

public:
  LinearCurvePowOfTwo()
  :mBoxXM1(0),mBoxYM1(0),mBoxZM1(0),mBoxXLog2(0),mBoxXYLog2(0){};
  template <class IngredientsType >
  void initialize(const IngredientsType& ing)
  {
    auto mBoxX =  ing.getBoxX();
    auto mBoxY =  ing.getBoxY();
    auto mBoxZ =  ing.getBoxZ();
    mBoxXM1 = mBoxX-1;
    mBoxYM1 = mBoxY-1;
    mBoxZM1 = mBoxZ-1;
    /* determine log2 for mBoxX and mBoxX * mBoxY to be used for bitshifting
    * the indice instead of multiplying ... WHY??? I don't think it is faster,
    * but much less readable */
    mBoxXLog2  = 0; auto dummy = mBoxX ; while ( dummy >>= 1 ) ++mBoxXLog2;
    mBoxXYLog2 = 0; dummy = mBoxX*mBoxY; while ( dummy >>= 1 ) ++mBoxXYLog2;
    if ( mBoxX != ( 1u << mBoxXLog2 ) || mBoxX * mBoxY != ( 1u << mBoxXYLog2 ) )
    {
	std::stringstream msg;
	msg << "[" << __FILENAME__ << "::initialize" << "] "
	    << "Could not determine value for bit shift. "
	    << "Check whether the box size is a power of 2! ( "
	    << "boxX=" << mBoxX << " =? 2^" << mBoxXLog2 << " = " << ( 1 << mBoxXLog2 )
	    << ", boxX*boY=" << mBoxX * mBoxY << " =? 2^" << mBoxXYLog2
	    << " = " << ( 1 << mBoxXYLog2 ) << " )\n";
	throw std::runtime_error( msg.str() );
    }
    std::cout << "LinearCurvePowOfTwo::initialize uses mBoxXLog2 of "<< mBoxXLog2 << " and mBoxXYLog2 "<< mBoxXYLog2 <<std::endl; 
  }
  template < typename T >
  void initialize(T mBoxX_, T mBoxY_, T mBoxZ_)
  {
    mBoxXM1 = mBoxX_-1;
    mBoxYM1 = mBoxY_-1;
    mBoxZM1 = mBoxZ_-1;
    /* determine log2 for mBoxX and mBoxX * mBoxY to be used for bitshifting
    * the indice instead of multiplying ... WHY??? I don't think it is faster,
    * but much less readable */
    mBoxXLog2  = 0; auto dummy = mBoxX_ ; while ( dummy >>= 1 ) ++mBoxXLog2;
    mBoxXYLog2 = 0; dummy = mBoxX_*mBoxY_; while ( dummy >>= 1 ) ++mBoxXYLog2;
    if ( mBoxX_ != ( 1u << mBoxXLog2 ) || mBoxX_ * mBoxY_ != ( 1u << mBoxXYLog2 ) )
    {
      std::stringstream msg;
      msg << "[" << __FILENAME__ << "::initialize" << "] "
          << "Could not determine value for bit shift. "
          << "Check whether the box size is a power of 2! ( "
          << "boxX=" << mBoxX_ << " =? 2^" << mBoxXLog2 << " = " << ( 1 << mBoxXLog2 )
          << ", boxX*boY=" << mBoxX_ * mBoxY_ << " =? 2^" << mBoxXYLog2
          << " = " << ( 1 << mBoxXYLog2 ) << " )\n";
      throw std::runtime_error( msg.str() );
    }
    std::cout << "LinearCurvePowOfTwo::initialize uses mBoxXLog2 of "<< mBoxXLog2 << " and mBoxXYLog2 "<< mBoxXYLog2 <<std::endl;
  }
  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexX
  ( T const & ix ) const 
  {
  #ifdef __CUDA_ARCH__
	  return    ( ix & dcBoxXM1 ) ;
  #else 
	  return    ( T_LinVector( ix ) & mBoxXM1 ) ;
  #endif
  }
  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexY
  ( T const & iy ) const 
  {
  #ifdef __CUDA_ARCH__
	  return  ( iy & dcBoxYM1 ) << dcBoxXLog2 ;
  #else 
	  return  ( T_LinVector( iy ) & mBoxYM1 ) << mBoxXLog2 ;
  #endif
  }

  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexZ
  ( T const & iz ) const 
  { 
  #ifdef __CUDA_ARCH__
	  return  ( ( iz & dcBoxZM1 ) << dcBoxXYLog2 );
  #else 
	  return  ( ( T_LinVector( iz ) & mBoxZM1 ) << mBoxXYLog2 );	  
  #endif
  }
  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz) const 
  { 
    return linearizeBoxVectorIndexX(ix) +
	   linearizeBoxVectorIndexY(iy) +
	   linearizeBoxVectorIndexZ(iz);
  }
private:
  uint64_t mBoxXM1, mBoxYM1, mBoxZM1; 
  uint64_t mBoxXLog2, mBoxXYLog2;
};
/*****************************************************************************/
/**
 * @brief sometimes referred as NoMagic
 * @todo write a test and check functions
 */
class LinearCurve:public AbstractSpaceFillingCurve<LinearCurve>{

public:
  
  LinearCurve()
  :mBoxX(0),mBoxY(0),mBoxZ(0){};
  
  template <class IngredientsType >
  void initialize(const IngredientsType& ing)
  {
    mBoxX =  ing.getBoxX();
    mBoxY =  ing.getBoxY();
    mBoxZ =  ing.getBoxZ();
  }
  template < typename T >
  void initialize(T mBoxX_, T mBoxY_, T mBoxZ_)
  {
    mBoxX = mBoxX_;
    mBoxY = mBoxY_;
    mBoxZ = mBoxZ_;
  }
  
  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexX
  ( T const & ix ) const 
  {
    #ifdef __CUDA_ARCH__
	  return ( ix % dcBoxX );
    #else 
	  return ( ix % mBoxX ) ;
    #endif
  }
  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexY
  ( T const & iy ) const 
  { 
    #ifdef __CUDA_ARCH__
	    return ( iy % dcBoxY ) * dcBoxX;
    #else 
	    return ( iy % mBoxY ) * mBoxX;
    #endif
  }

  template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndexZ
  ( T const & iz ) const
  {
    #ifdef __CUDA_ARCH__
	    return ( iz % dcBoxZ ) * dcBoxX * dcBoxY;
    #else 
	    return ( iz % mBoxZ ) * mBoxX * mBoxY;
    #endif
  }
    template< typename T >
  __host__ __device__ inline T_LinVector linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz) const 
  { 
    return linearizeBoxVectorIndexX(ix) +
	   linearizeBoxVectorIndexY(iy) +
	   linearizeBoxVectorIndexZ(iz);
  }
private:
  uint64_t mBoxX, mBoxY, mBoxZ; 
};

/**
 * @todo We maybe could optimize this approach by using operators and 
 */
class SpaceFillingCurve {

public:
  enum Curvemode {  ZOrderCurveMode=0,
	            LinearMode=1,
	            LinearPowOfTwoMode=2
  };
  SpaceFillingCurve();
  SpaceFillingCurve(uint64_t mBoxX_, uint64_t mBoxY_, uint64_t mBoxZ_, int mode_=2 );
  void setBox(uint64_t mBoxX_, uint64_t mBoxY_, uint64_t mBoxZ_);
 
private:
  int mode;
  ZOrderCurve zCurve;
  LinearCurve lCurve;
  LinearCurvePowOfTwo lP2Curve;
  
  inline bool IsPowerOfTwo(uint64_t x)
  {
      return (x != 0) && ((x & (x - 1)) == 0);
  }
public:  
  int getMode() const ;
//   {return mode;}
  void setMode( int mode_);
//   { mode=mode_; }
  
template< typename T >
__host__ __device__ inline T_LinVector linearizeBoxVectorIndex
( T const & ix, T const & iy, T const & iz ) const {
  switch(mode){
    case ZOrderCurveMode    : return zCurve.linearizeBoxVectorIndex(ix,iy,iz);
    case LinearMode         : return lCurve.linearizeBoxVectorIndex(ix,iy,iz);
    case LinearPowOfTwoMode : return lP2Curve.linearizeBoxVectorIndex(ix,iy,iz);

  };
  return T_LinVector(); // to supress warnings 
}

template< typename T >
__host__ __device__ inline T_LinVector linearizeBoxVectorIndexX
( T const & ix ) const {
  switch(mode){
    case ZOrderCurveMode    : return zCurve.linearizeBoxVectorIndexX(ix);
    case LinearMode         : return lCurve.linearizeBoxVectorIndexX(ix);
    case LinearPowOfTwoMode : return lP2Curve.linearizeBoxVectorIndexX(ix);
  };
  return T_LinVector(); // to supress warnings 
}

template< typename T >
__host__ __device__ inline T_LinVector linearizeBoxVectorIndexY
( T const & iy ) const {
  switch(mode){
    case ZOrderCurveMode    : return zCurve.linearizeBoxVectorIndexY(iy);
    case LinearMode         : return lCurve.linearizeBoxVectorIndexY(iy);
    case LinearPowOfTwoMode : return lP2Curve.linearizeBoxVectorIndexY(iy);
  };
  return T_LinVector(); // to supress warnings 
}

template< typename T >
__host__ __device__ inline T_LinVector linearizeBoxVectorIndexZ
( T const & iz ) const {
  switch(mode){
    case ZOrderCurveMode    : return zCurve.linearizeBoxVectorIndexZ(iz);
    case LinearMode         : return lCurve.linearizeBoxVectorIndexZ(iz);
    case LinearPowOfTwoMode : return lP2Curve.linearizeBoxVectorIndexZ(iz);
  };
  return T_LinVector(); // to supress warnings 
}

};

#endif 
