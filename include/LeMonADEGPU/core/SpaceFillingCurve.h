/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo---0---oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADE.

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
#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.h>

#include <extern/Fundamental/BitsCompileTime.hpp>
#include <LeMonADEGPU/core/constants.cuh>

/*****************************************************************************/
/**
 *@brief abstract template function for the space filling curve 
 */
/*****************************************************************************/

typedef uint32_t T_Id  ;  
  
template <class specializedCurve>
class AbstractSpaceFillingCurve
{

public:

template< typename T >
__host__ __device__ inline T_Id linearizeBoxVectorIndex
( T const & ix, T const & iy, T const & iz) const { static_cast<specializedCurve*>(this)->linearizeBoxVectorIndex(ix,iy,iz);}

template < class IngredientsType >
void initialize(const IngredientsType& ing){ static_cast<specializedCurve*>(this)->initialize(ing);}

template < typename T >
void initialize(T mBoxX_, T mBoxY_, T mBoxZ_){ static_cast<specializedCurve*>(this)->initialize(mBoxX_,mBoxY_,mBoxZ_);}

private:

  
};


/*****************************************************************************/
/**
 * @brief
 */
class ZOrderCurve:public AbstractSpaceFillingCurve<ZOrderCurve>
{
  
public:
  template <class IngredientsType >
  void initialize(const IngredientsType& ing)
  {
    mBoxXM1 = ing.getBoxX()-1;
    mBoxYM1 = ing.getBoxY()-1;
    mBoxZM1 = ing.getBoxZ()-1;
  }
  template < typename T >
  void initialize(T mBoxX_, T mBoxY_, T mBoxZ_){
    mBoxXM1 = mBoxX_-1;
    mBoxYM1 = mBoxY_-1;
    mBoxZM1 = mBoxZ_-1;
  }

      
  template <typename T >  
  __host__ __device__ inline T_Id linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz ) const { 
  #ifdef __CUDA_ARCH__
	  return  diluteBits< T_Id, 2 >( ix & dcBoxXM1 )        +
		( diluteBits< T_Id, 2 >( iy & dcBoxYM1 ) << 1 ) +
		( diluteBits< T_Id, 2 >( iz & dcBoxZM1 ) << 2 );
  #else 
	  return diluteBits< T_Id, 2 >( T_Id( ix ) & mBoxXM1 )        +
	       ( diluteBits< T_Id, 2 >( T_Id( iy ) & mBoxYM1 ) << 1 ) +
	       ( diluteBits< T_Id, 2 >( T_Id( iz ) & mBoxZM1 ) << 2 );
  #endif
  }
private: 
  uint32_t mBoxXM1, mBoxYM1, mBoxZM1; 

};
/*****************************************************************************/
/**
 * @brief
 */
class LinearCurvePowOfTwo:public AbstractSpaceFillingCurve<LinearCurvePowOfTwo>{

public:
  template <class IngredientsType >
  void initialize(const IngredientsType& ing){
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
  }
  template < typename T >
  void initialize(T mBoxX_, T mBoxY_, T mBoxZ_){
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
  }
  
  template <typename T>  
  __host__ __device__ inline T_Id linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz ) const { 
  #ifdef __CUDA_ARCH__
	  return    ( ix & dcBoxXM1 ) +
		  ( ( iy & dcBoxYM1 ) << dcBoxXLog2  ) +
		  ( ( iz & dcBoxZM1 ) << dcBoxXYLog2 );
  #else 
	  return    ( T_Id( ix ) & mBoxXM1 ) +
		  ( ( T_Id( iy ) & mBoxYM1 ) << mBoxXLog2  ) +
		  ( ( T_Id( iz ) & mBoxZM1 ) << mBoxXYLog2 );	  
  #endif
  }

private:
  uint32_t mBoxXM1, mBoxYM1, mBoxZM1; 
  uint32_t mBoxXLog2, mBoxXYLog2;
};
/*****************************************************************************/
/**
 * @brief sometimes referred as NoMagic
 * @todo write a test and check functions
 */
class LinearCurve:public AbstractSpaceFillingCurve<LinearCurve>{

public:
  template <class IngredientsType >
  void initialize(const IngredientsType& ing){
    mBoxX =  ing.getBoxX();
    mBoxY =  ing.getBoxY();
    mBoxZ =  ing.getBoxZ();
  }
  template < typename T >
  void initialize(T mBoxX_, T mBoxY_, T mBoxZ_){
    mBoxX = mBoxX_;
    mBoxY = mBoxY_;
    mBoxZ = mBoxZ_;
  }
  
  template <typename T >  
  __host__ __device__ inline T_Id linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz ) const { 
  #ifdef __CUDA_ARCH__
	  return ( ix % dcBoxX ) +
		 ( iy % dcBoxY ) * dcBoxX +
		 ( iz % dcBoxZ ) * dcBoxX * dcBoxY;
  #else 
	  return ( ix % mBoxX ) +
		 ( iy % mBoxY ) * mBoxX +
		 ( iz % mBoxZ ) * mBoxX * mBoxY;
  #endif
  }
private:
  uint32_t mBoxX, mBoxY, mBoxZ; 
};


class SpaceFillingCurve {

public:
  enum Curvemode {  ZOrderCurveMode=0,
	            LinearMode=1,
	            LinearPowOfTwoMode=2
  };
  SpaceFillingCurve(){};
  template <class IngredientsType >
  SpaceFillingCurve(IngredientsType& ing)
  {
    zCurve.initialize(ing);
    lCurve.initialize(ing);
    lP2Curve.initialize(ing);
  }
  template <class T >
  SpaceFillingCurve(T mBoxX_, T mBoxY_, T mBoxZ_)
  {
    zCurve.initialize(mBoxX_,mBoxY_,mBoxZ_);
    lCurve.initialize(mBoxX_,mBoxY_,mBoxZ_);
    lP2Curve.initialize(mBoxX_,mBoxY_,mBoxZ_);
  }
private:
  int mode;
  ZOrderCurve zCurve;
  LinearCurve lCurve;
  LinearCurvePowOfTwo lP2Curve;
public:  
  int getCurve() const {return mode;}
  void setCurve( int mode_) { mode=mode_; }
  
template< typename T >
__host__ __device__ inline T_Id linearizeBoxVectorIndex
( T const & ix, T const & iy, T const & iz) const {
  switch(mode){
    case ZOrderCurveMode    : return zCurve.linearizeBoxVectorIndex(ix,iy,iz);
    case LinearMode         : return lCurve.linearizeBoxVectorIndex(ix,iy,iz);
    case LinearPowOfTwoMode : return lP2Curve.linearizeBoxVectorIndex(ix,iy,iz);
  };
}
};

#endif 