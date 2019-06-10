#include <iostream>
#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.h>

/**
 * @brief class for checking the coordinates for periodic boundary conditions
 * @todo hand over a pointer to the box sizes on the device with cudaGetSymbolAdress(void** devPtr, const void**  symbol)
 */
struct BoxCheck
{
private:
  /**
   * @brief convinience enum for the periodicity
   */
  enum mode {  periodic111, periodic000, 
	       periodic100, periodic010, periodic001, 
	       periodic110, periodic011, periodic101, 
	       periodic=0, nonperiodic=1 };
	       
  int myperiodicmode;
  bool pX, pY, pZ;
  
public:
  /*standard constructor*/
  BoxCheck():myperiodicmode(0) {}
  /**
   * @brief constructor 
   * @param myperiodicmode_ periodicityas enum 
   * 
   */
  BoxCheck(int myperiodicmode_ ):myperiodicmode(myperiodicmode_)
  {  }
  
  /**
   * @brief constructor 
   * @param pX_ periodicity in x-direction
   * @param pY_ periodicity in y-direction
   * @param pZ_ periodicity in z-direction
   */
  BoxCheck( bool pX_,  bool pY_,  bool pZ_):pX(pX_),pY(pY_),pZ(pZ_)
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
  template <class T >
  __device__ bool operator()(const T & x, const T &  y, const T & z)
  {
    switch(myperiodicmode)
    { 
	case periodic111: return  true                         ;
	case periodic000: return ((0) <= x && x <  dcBoxXM1 &&
				  (0) <= y && y <  dcBoxYM1 &&
				  (0) <= z && z <  dcBoxZM1 )  ;  
	case periodic100: return ((0) <= y && y <  dcBoxYM1 &&
				  (0) <= z && z <  dcBoxZM1 )  ;
	case periodic010: return ((0) <= x && x <  dcBoxXM1 &&
				  (0) <= z && z <  dcBoxZM1 )  ;
	case periodic001: return ((0) <= x && x <  dcBoxXM1 &&
				  (0) <= y && y <  dcBoxYM1 )  ;
	case periodic110: return ((0) <= z && z <  dcBoxZM1 )  ;
	case periodic011: return ((0) <= x && x <  dcBoxXM1 )  ;
	case periodic101: return ((0) <= y && y <  dcBoxYM1 )  ;
	default         : return false                                ; //maybe throw an error?! 
    };
  }

};