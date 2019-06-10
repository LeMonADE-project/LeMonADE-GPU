#include <extern/Fundamental/BitsCompileTime.hpp>
/**
 * @brief abstract class which needs to be specialized 
 */
template < class SpecializedClass >
class SpaceFillingCurve{
  
public:
template <class T >  
__device__ inline  T getNodeXdir( T const & coord, T const & shift ){return static_cast<SpecializedClass*>(this)->getNodeXDir(coord,shift);}

template <class T >  
__device__ inline  T getNodeYdir( T const & coord, T const & shift ){return static_cast<SpecializedClass*>(this)->getNodeYDir(coord,shift);}

template <class T >  
__device__ inline  T getNodeZdir( T const & coord, T const & shift ){return static_cast<SpecializedClass*>(this)->getNodeZDir(coord,shift);}

template< typename T_Id >
__host__ __device__  inline T_Id  linearizeBoxVectorIndex(
				    uint32_t const & ix,
				    uint32_t const & iy,
				    uint32_t const & iz) const
{ return static_cast<SpecializedClass*>(this)->linearizeBoxVectorIndex( ix, iy, iz); };

template <class T , class T2 >
__device__ inline T const calcX0MDX( T2 x ){ return static_cast<SpecializedClass*>(this)->calcX0MDX(  x ); }
template <class T , class T2 >
__device__ inline T const calcX0Abs( T2 x ){ return static_cast<SpecializedClass*>(this)->calcX0Abs(  x ); }
template <class T , class T2 >
__device__ inline T const calcX0PDX( T2 x ){ return static_cast<SpecializedClass*>(this)->calcX0PDX(  x ); }

template <class T , class T2 >
__device__ inline T const calcY0MDY( T2 y ){ return static_cast<SpecializedClass*>(this)->calcY0MDY(  x ); }
template <class T , class T2 >
__device__ inline T const calcY0Abs( T2 y ){ return static_cast<SpecializedClass*>(this)->calcY0Abs(  x ); }
template <class T , class T2 >
__device__ inline T const calcY0PDZ( T2 y ){ return static_cast<SpecializedClass*>(this)->calcY0PDZ(  x ); }

template <class T , class T2 >
__device__ inline T const calcZ0MDZ( T2 z ){ return static_cast<SpecializedClass*>(this)->calcZ0MDZ(  x ); }
template <class T , class T2 >
__device__ inline T const calcZ0Abs( T2 z ){ return static_cast<SpecializedClass*>(this)->calcZ0Abs(  x ); }
template <class T , class T2 >
__device__ inline T const calcZ0PDZ( T2 z ){ return static_cast<SpecializedClass*>(this)->calcZ0PDZ(  x ); }
  
  
  
private:
  bool inline isPowerOfTwo( uint32_t n ){return (n & (n - 1)) == 0;}
  
  const uint32_t mBoxX, mBoxZ, mBoxXLog2, mBoxXYLog2, mBoxXM1, mBoxYM1, mBoxZM1;
  
};

/**
 * @details use bit shifts and thus works only for power of two lattices 
 * 	      is only applicable for cubic lattices 
 * @todo  at an implementation for non-cubic lattices
 */
class ZOrderCurve:public SpaceFillingCurve< ZOrderCurve >{
  
public:
//   ZOrderCurve(paramTATAT):SpaceFillingCurve(paramTATAT){};
template< typename T_Id >
__host__ __device__  inline T_Id  linearizeBoxVectorIndex(
			    uint32_t const & ix,
			    uint32_t const & iy,
			    uint32_t const & iz) const {
  #ifdef __CUDA_ARCH__
       return diluteBits< T_Id, 2 >( ix & dcBoxXM1 )        +
	    ( diluteBits< T_Id, 2 >( iy & dcBoxYM1 ) << 1 ) +
	    ( diluteBits< T_Id, 2 >( iz & dcBoxZM1 ) << 2 );
  #else
       return diluteBits< T_Id, 2 >( T_Id( ix ) & mBoxXM1 )        +
	    ( diluteBits< T_Id, 2 >( T_Id( iy ) & mBoxYM1 ) << 1 ) +
	    ( diluteBits< T_Id, 2 >( T_Id( iz ) & mBoxZM1 ) << 2 );
  #endif
  }
  
template <class T , class T2 >
__device__ inline T const calcX0MDX( T2 x ){ return diluteBits< uint32_t, 2 >( ( x0 - uint32_t(1) ) & dcBoxXM1 ); }
template <class T , class T2 >
__device__ inline T const calcX0Abs( T2 x ){ return diluteBits< uint32_t, 2 >( ( x0               ) & dcBoxXM1 ) }
template <class T , class T2 >
__device__ inline T const calcX0PDX( T2 x ){ return diluteBits< uint32_t, 2 >( ( x0 + uint32_t(1) ) & dcBoxXM1 ); }

template <class T , class T2 >
__device__ inline T const calcY0MDY( T2 y ){ return diluteBits< uint32_t, 2 >( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << 1; }
template <class T , class T2 >
__device__ inline T const calcY0Abs( T2 y ){ return diluteBits< uint32_t, 2 >( ( y0               ) & dcBoxYM1 ) << 1; }
template <class T , class T2 >
__device__ inline T const calcY0PDZ( T2 y ){ return diluteBits< uint32_t, 2 >( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << 1; }

template <class T , class T2 >
__device__ inline T const calcZ0MDZ( T2 z ){ return diluteBits< uint32_t, 2 >( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << 2; }
template <class T , class T2 >
__device__ inline T const calcZ0Abs( T2 z ){ return diluteBits< uint32_t, 2 >( ( z0               ) & dcBoxZM1 ) << 2; }
template <class T , class T2 >
__device__ inline T const calcZ0PDZ( T2 z ){ return diluteBits< uint32_t, 2 >( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << 2; }
  
  
template <class T >  
__device__ inline  T getNodeXdir( T const & coord, T const & shift ){return diluteBits< T, 2 >( ( coord + shift + shift ) & dcBoxXM1 )      ;}
template <class T >  
__device__ inline  T getNodeYdir( T const & coord, T const & shift ){return diluteBits< T, 2 >( ( coord + shift + shift ) & dcBoxXM1 ) << 1 ;}
template <class T >  
__device__ inline  T getNodeZdir( T const & coord, T const & shift ){return diluteBits< T, 2 >( ( coord + shift + shift ) & dcBoxXM1 ) << 1 ;}

};

/**
 * @brief use linearized coordinates 
 * @details use bit shifts and thus works only for power of two lattices 
 */

class LinearCurve:public SpaceFillingCurve<LinearCurve>{
public: 
template< typename T_Id >
__host__ __device__ inline T_Id  linearizeBoxVectorIndex(
			    uint32_t const & ix,
			    uint32_t const & iy,
			    uint32_t const & iz) const 
  {
#ifdef __CUDA_ARCH__
    return    ( ix & dcBoxXM1 )                  +
	    ( ( iy & dcBoxYM1 ) << dcBoxXLog2  ) +
            ( ( iz & dcBoxZM1 ) << dcBoxXYLog2 );
#else
    return    ( T_Id( ix ) & mBoxXM1 ) +
	    ( ( T_Id( iy ) & mBoxYM1 ) << mBoxXLog2  ) +
	    ( ( T_Id( iz ) & mBoxZM1 ) << mBoxXYLog2 );
#endif
  };

template <class T , class T2 >
__device__ inline T const calcX0MDX( T2 x ){ return ( x0 - uint32_t(1) ) & dcBoxXM1;                  }
template <class T , class T2 >
__device__ inline T const calcX0Abs( T2 x ){ return ( x0               ) & dcBoxXM1;                  }
template <class T , class T2 >
__device__ inline T const calcX0PDX( T2 x ){ return ( x0 + uint32_t(1) ) & dcBoxXM1;                  }

template <class T , class T2 >
__device__ inline T const calcY0MDY( T2 y ){ return ( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;}
template <class T , class T2 >
__device__ inline T const calcY0Abs( T2 y ){ return ( ( y0               ) & dcBoxYM1 ) << dcBoxXLog2;}
template <class T , class T2 >
__device__ inline T const calcY0PDZ( T2 y ){ return ( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;}

template <class T , class T2 >
__device__ inline T const calcZ0MDZ( T2 z ){ return ( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;}
template <class T , class T2 >
__device__ inline T const calcZ0Abs( T2 z ){ return ( ( z0               ) & dcBoxZM1 ) << dcBoxXYLog2;}
template <class T , class T2 >
__device__ inline T const calcZ0PDZ( T2 z ){ return ( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;}
  
template <class T >  
__device__ inline  T getNodeXdir( T const & coord, T const & shift ){return ( ( coord + 2*shift ) & dcBoxXM1 )                ;}
template <class T >  
__device__ inline  T getNodeYdir( T const & coord, T const & shift ){return ( ( coord + 2*shift ) & dcBoxXM1 ) << dcBoxXLog2  ;}
template <class T >  
__device__ inline  T getNodeZdir( T const & coord, T const & shift ){return ( ( coord + 2*shift ) & dcBoxXM1 ) << dcBoxXYLog2 ;}

};


/**
 * @brief use linearized coordinates 
 * @details can be used also for box sizes not power of two!  
 */

class LinearCurveNoMagic:public SpaceFillingCurve<LinearCurveNoMagic>{
public: 
template< typename T_Id >
__host__ __device__ inline T_Id  linearizeBoxVectorIndex(
			    uint32_t const & ix,
			    uint32_t const & iy,
			    uint32_t const & iz) const 
  {
#ifdef __CUDA_ARCH__
	return  ( ix % dcBoxXM1 )           +
		( iy % dcBoxYM1 ) * dcBoxX  +
		( iz % dcBoxZM1 ) * dcBoxX * dcBoxY;
#else
        return ( ix % mBoxX ) +
               ( iy % mBoxY ) * mBoxX +
               ( iz % mBoxZ ) * mBoxX * mBoxY;
#endif
  };

template <class T , class T2 >
__device__ inline T const calcX0MDX( T2 x ){ return ( x0 - uint32_t(1) ) % dcBoxXM1;                  }
template <class T , class T2 >
__device__ inline T const calcX0Abs( T2 x ){ return ( x0               ) % dcBoxXM1;                  }
template <class T , class T2 >
__device__ inline T const calcX0PDX( T2 x ){ return ( x0 + uint32_t(1) ) % dcBoxXM1;                  }

template <class T , class T2 >
__device__ inline T const calcY0MDY( T2 y ){ return ( ( y0 - uint32_t(1) ) % dcBoxYM1 ) * dcBoxX;}
template <class T , class T2 >
__device__ inline T const calcY0Abs( T2 y ){ return ( ( y0               ) % dcBoxYM1 ) * dcBoxX;}
template <class T , class T2 >
__device__ inline T const calcY0PDZ( T2 y ){ return ( ( y0 + uint32_t(1) ) % dcBoxYM1 ) * dcBoxX;}

template <class T , class T2 >
__device__ inline T const calcZ0MDZ( T2 z ){ return ( ( z0 - uint32_t(1) ) & dcBoxZM1 ) * dcBoxX * dcBoxY;}
template <class T , class T2 >
__device__ inline T const calcZ0Abs( T2 z ){ return ( ( z0               ) & dcBoxZM1 ) * dcBoxX * dcBoxY;}
template <class T , class T2 >
__device__ inline T const calcZ0PDZ( T2 z ){ return ( ( z0 + uint32_t(1) ) & dcBoxZM1 ) * dcBoxX * dcBoxY;}
  
template <class T >  
__device__ inline  T getNodeXdir( T const & coord, T const & shift ){return ( ( coord + 2*shift ) & dcBoxXM1 )                   ;}
template <class T >  
__device__ inline  T getNodeYdir( T const & coord, T const & shift ){return ( ( coord + 2*shift ) & dcBoxXM1 ) * dcBoxX          ;}
template <class T >  
__device__ inline  T getNodeZdir( T const & coord, T const & shift ){return ( ( coord + 2*shift ) & dcBoxXM1 ) * dcBoxX * dcBoxY ;}

private: 

};