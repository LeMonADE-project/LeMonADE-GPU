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
#ifndef LEMONADE_CORE_SPACE_FILLING_CURVE_CU
#define LEMONADE_CORE_SPACE_FILLING_CURVE_CU

#include <extern/Fundamental/BitsCompileTime.hpp>
#include <LeMonADEGPU/core/SpaceFillingCurve.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <stdio.h>

/*****************************************************************************/
/**
 *@brief abstract template function for the space filling curve 
 */
/*****************************************************************************/
 __constant__ uint32_t dcBoxX     ;  // lattice size in X
 __constant__ uint32_t dcBoxY     ;  // lattice size in Y
 __constant__ uint32_t dcBoxZ     ;  // lattice size in Z
 __constant__ uint32_t dcBoxXM1   ;  // lattice size in X-1
 __constant__ uint32_t dcBoxYM1   ;  // lattice size in Y-1
 __constant__ uint32_t dcBoxZM1   ;  // lattice size in Z-1
 __constant__ uint32_t dcBoxXLog2 ;  // lattice shift in X
 __constant__ uint32_t dcBoxXYLog2;  // lattice shift in X*Y
__global__ void CheckBoxDimensionsSpaceFilling()
{
printf("CheckBoxDimensionsSpaceFilling: %d %d %d %d %d %d  %d %d",dcBoxX,dcBoxY, dcBoxZ,dcBoxXM1, dcBoxYM1,dcBoxZM1, dcBoxXLog2, dcBoxXYLog2 );
}
  SpaceFillingCurve::SpaceFillingCurve(){};
  template <class IngredientsType >
  SpaceFillingCurve::SpaceFillingCurve(IngredientsType& ing)
  {
    zCurve.initialize(ing);
    lCurve.initialize(ing);
    lP2Curve.initialize(ing);
  }
  
  SpaceFillingCurve::SpaceFillingCurve(uint32_t mBoxX_, uint32_t mBoxY_, uint32_t mBoxZ_)
  {
    zCurve.initialize(mBoxX_,mBoxY_,mBoxZ_);
    lCurve.initialize(mBoxX_,mBoxY_,mBoxZ_);
    lP2Curve.initialize(mBoxX_,mBoxY_,mBoxZ_);
    uint32_t mBoxXLog2  = 0; auto dummy = mBoxX_ ; while ( dummy >>= 1 ) ++mBoxXLog2;
    uint32_t mBoxXYLog2 = 0; dummy = mBoxX_*mBoxY_; while ( dummy >>= 1 ) ++mBoxXYLog2;
    { decltype( dcBoxX      ) x = mBoxX_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
    { decltype( dcBoxY      ) x = mBoxY_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
    { decltype( dcBoxZ      ) x = mBoxZ_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
    { decltype( dcBoxXM1    ) x = mBoxX_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxYM1    ) x = mBoxY_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxZM1    ) x = mBoxZ_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxXLog2  ) x = mBoxXLog2  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
    { decltype( dcBoxXYLog2 ) x = mBoxXYLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); }
    CheckBoxDimensionsSpaceFilling<<<1,1>>>();
  }

  void SpaceFillingCurve::setBox(uint32_t mBoxX_, uint32_t mBoxY_, uint32_t mBoxZ_)
  {
    zCurve.initialize(mBoxX_,mBoxY_,mBoxZ_);
    lCurve.initialize(mBoxX_,mBoxY_,mBoxZ_);
    lP2Curve.initialize(mBoxX_,mBoxY_,mBoxZ_);
        uint32_t mBoxXLog2  = 0; auto dummy = mBoxX_ ; while ( dummy >>= 1 ) ++mBoxXLog2;
    uint32_t mBoxXYLog2 = 0; dummy = mBoxX_*mBoxY_; while ( dummy >>= 1 ) ++mBoxXYLog2;
    { decltype( dcBoxX      ) x = mBoxX_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
    { decltype( dcBoxY      ) x = mBoxY_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
    { decltype( dcBoxZ      ) x = mBoxZ_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
    { decltype( dcBoxXM1    ) x = mBoxX_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxYM1    ) x = mBoxY_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxZM1    ) x = mBoxZ_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxXLog2  ) x = mBoxXLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
    { decltype( dcBoxXYLog2 ) x = mBoxXYLog2; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); }
    CheckBoxDimensionsSpaceFilling<<<1,1>>>();
  }

  int SpaceFillingCurve::getMode() const {return mode;}
  void SpaceFillingCurve::setMode( int mode_) { mode=mode_; }
 /*
 
typedef uint32_t T_Id  ;  
template <class specializedCurve>
template< typename T >
__host__ __device__   T_Id AbstractSpaceFillingCurve<specializedCurve>::linearizeBoxVectorIndex
( T const & ix, T const & iy, T const & iz) const { return static_cast<specializedCurve*>(this)->linearizeBoxVectorIndex(ix,iy,iz); }
template <class specializedCurve>
template< typename T >
__host__ __device__   T_Id AbstractSpaceFillingCurve<specializedCurve>::linearizeBoxVectorIndexX
( T const & ix ) const { return static_cast<specializedCurve*>(this)->linearizeBoxVectorIndexX(ix);}
template <class specializedCurve>
template< typename T >
__host__ __device__   T_Id AbstractSpaceFillingCurve<specializedCurve>::linearizeBoxVectorIndexY
( T const & iy ) const { return static_cast<specializedCurve*>(this)->linearizeBoxVectorIndexY(iy);}
template <class specializedCurve>
template< typename T >
__host__ __device__   T_Id AbstractSpaceFillingCurve<specializedCurve>::linearizeBoxVectorIndexZ
( T const & iz ) const { return static_cast<specializedCurve*>(this)->linearizeBoxVectorIndexZ(iz);}
template <class specializedCurve>
template < class IngredientsType >
void AbstractSpaceFillingCurve<specializedCurve>::initialize(const IngredientsType& ing){ static_cast<specializedCurve*>(this)->initialize(ing);}
template <class specializedCurve>
template < typename T >
void AbstractSpaceFillingCurve<specializedCurve>::initialize(T mBoxX_, T mBoxY_, T mBoxZ_){ static_cast<specializedCurve*>(this)->initialize(mBoxX_,mBoxY_,mBoxZ_);}


  ZOrderCurve::ZOrderCurve():mBoxXM1(0),mBoxYM1(0),mBoxZM1(0){};
  template <class IngredientsType >
  void   ZOrderCurve::initialize(const IngredientsType& ing)
  {
    mBoxXM1 = ing.getBoxX()-1;
    mBoxYM1 = ing.getBoxY()-1;
    mBoxZM1 = ing.getBoxZ()-1;
  }
  template < typename T >
  void   ZOrderCurve::initialize(T mBoxX_, T mBoxY_, T mBoxZ_){
    mBoxXM1 = mBoxX_-1;
    mBoxYM1 = mBoxY_-1;
    mBoxZM1 = mBoxZ_-1;
  }
  
    template< typename T >
  __host__ __device__   T_Id ZOrderCurve::linearizeBoxVectorIndexX
  ( T const & ix ) const
  {
    #ifdef __CUDA_ARCH__
	    return diluteBits< T_Id, 2 >( ix & dcBoxXM1 ) ;
    #else 
	    return diluteBits< T_Id, 2 >( T_Id( ix ) & mBoxXM1 ) ;
    #endif
  }
  template< typename T >
  __host__ __device__   T_Id ZOrderCurve::linearizeBoxVectorIndexY
  ( T const & iy ) const 
  {
  #ifdef __CUDA_ARCH__
	  return diluteBits< T_Id, 2 >( iy & dcBoxYM1 ) << 1 ;
  #else 
	  return diluteBits< T_Id, 2 >( T_Id( iy ) & mBoxYM1 ) << 1 ;
  #endif
  }

  template< typename T >
  __host__ __device__   T_Id ZOrderCurve::linearizeBoxVectorIndexZ
  ( T const & iz ) const 
  {
  #ifdef __CUDA_ARCH__
	  return diluteBits< T_Id, 2 >( iz & dcBoxZM1 ) << 2 ;
  #else 
	  return diluteBits< T_Id, 2 >( T_Id( iz ) & mBoxZM1 ) << 2 ;
  #endif
  }
  template< typename T >
  __host__ __device__   T_Id ZOrderCurve::linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz) const 
  { 
    return linearizeBoxVectorIndexX(ix) +
	   linearizeBoxVectorIndexY(iy) +
	   linearizeBoxVectorIndexZ(iz);
  }
 

  LinearCurvePowOfTwo::LinearCurvePowOfTwo():mBoxXM1(0),mBoxYM1(0),mBoxZM1(0),mBoxXLog2(0),mBoxXYLog2(0){};
  template <class IngredientsType >
  void LinearCurvePowOfTwo::initialize(const IngredientsType& ing){
    auto mBoxX =  ing.getBoxX();
    auto mBoxY =  ing.getBoxY();
    auto mBoxZ =  ing.getBoxZ();
    mBoxXM1 = mBoxX-1;
    mBoxYM1 = mBoxY-1;
    mBoxZM1 = mBoxZ-1;
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
  void LinearCurvePowOfTwo::initialize(T mBoxX_, T mBoxY_, T mBoxZ_){
    mBoxXM1 = mBoxX_-1;
    mBoxYM1 = mBoxY_-1;
    mBoxZM1 = mBoxZ_-1;
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
  
    template< typename T >
  __host__ __device__    T_Id LinearCurvePowOfTwo::linearizeBoxVectorIndexX
  ( T const & ix ) const 
  {
  #ifdef __CUDA_ARCH__
	  return    ( ix & dcBoxXM1 ) ;
  #else 
	  return    ( T_Id( ix ) & mBoxXM1 ) ;
  #endif
  }
  template< typename T >
  __host__ __device__   T_Id LinearCurvePowOfTwo::linearizeBoxVectorIndexY
  ( T const & iy ) const 
  {
  #ifdef __CUDA_ARCH__
	  return  ( iy & dcBoxYM1 ) << dcBoxXLog2 ;
  #else 
	  return  ( T_Id( iy ) & mBoxYM1 ) << mBoxXLog2 ;
  #endif
  }

  template< typename T >
  __host__ __device__   T_Id LinearCurvePowOfTwo::linearizeBoxVectorIndexZ
  ( T const & iz ) const 
  { 
  #ifdef __CUDA_ARCH__
	  return  ( ( iz & dcBoxZM1 ) << dcBoxXYLog2 );
  #else 
	  return  ( ( T_Id( iz ) & mBoxZM1 ) << mBoxXYLog2 );	  
  #endif
  }
  template< typename T >
  __host__ __device__   T_Id LinearCurvePowOfTwo::linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz) const 
  { 
    return linearizeBoxVectorIndexX(ix) +
	   linearizeBoxVectorIndexY(iy) +
	   linearizeBoxVectorIndexZ(iz);
  }
 
  LinearCurve::LinearCurve():mBoxX(0),mBoxY(0),mBoxZ(0){};
  
  template <class IngredientsType >
  void LinearCurve::initialize(const IngredientsType& ing){
    mBoxX =  ing.getBoxX();
    mBoxY =  ing.getBoxY();
    mBoxZ =  ing.getBoxZ();
  }
  template < typename T >
  void LinearCurve::initialize(T mBoxX_, T mBoxY_, T mBoxZ_){
    mBoxX = mBoxX_;
    mBoxY = mBoxY_;
    mBoxZ = mBoxZ_;
  }
  template< typename T >
  __host__ __device__   T_Id LinearCurve::linearizeBoxVectorIndexX
  ( T const & ix ) const 
  {
    #ifdef __CUDA_ARCH__
	  return ( ix % dcBoxX );
    #else 
	  return ( ix % mBoxX ) ;
    #endif
  }
  template< typename T >
  __host__ __device__   T_Id LinearCurve::linearizeBoxVectorIndexY
  ( T const & iy ) const 
  { 
    #ifdef __CUDA_ARCH__
	    return ( iy % dcBoxY ) * dcBoxX;
    #else 
	    return ( iy % mBoxY ) * mBoxX;
    #endif
  }

  template< typename T >
  __host__ __device__   T_Id LinearCurve::linearizeBoxVectorIndexZ
  ( T const & iz ) const 
  {
    #ifdef __CUDA_ARCH__
	    return ( iz % dcBoxZ ) * dcBoxX * dcBoxY;
    #else 
	    return ( iz % mBoxZ ) * mBoxX * mBoxY;
    #endif
  }
    template< typename T >
  __host__ __device__   T_Id LinearCurve::linearizeBoxVectorIndex
  ( T const & ix, T const & iy, T const & iz) const 
  { 
    return linearizeBoxVectorIndexX(ix) +
	   linearizeBoxVectorIndexY(iy) +
	   linearizeBoxVectorIndexZ(iz);
  }
*/

#endif 
