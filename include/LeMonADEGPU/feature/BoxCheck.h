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
#ifndef LEMONADEGPU_FEATURE_BOXCHECK_H_
#define LEMONADEGPU_FEATURE_BOXCHECK_H_
#include <iostream>
#include <LeMonADEGPU/core/constants.cuh>

/**
 * @brief class for checking the coordinates for periodic boundary conditions
 * @todo hand over a pointer to the box sizes on the device with cudaGetSymbolAdress(void** devPtr, const void**  symbol)
 * @todo think about to define device box constants  for only this file ?!
 * 
 * check whether the new location of the particle would be inside the box
 * if the box is not periodic, if not, then don't move the particle
 * r1 is unsigned so we don't have to check whether it's < 0 as that
 * would mean it -1 wraps around to UINTN_MAX
 * But in order for this to work with 256-sized boxes on uint8_t with
 * non-periodic boundary conditions, we have to check for wrap arounds
 * wrap arounds can only happen if the monomer was at 0 or dcBoxXM1
 * and moved outside
 *   0 <= x1 <= dcBoxX is useless, we need to replace, not add to it!
 *   0 < x0 <= dxBoxXM1 || ( x0 == 0 && x1 <= 1 ) || ( x0 == 255 && x1 >= 254 )
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
  BoxCheck( bool pX,  bool pY,  bool pZ)
  {
    initialize( pX,  pY,  pZ);
  }
  inline void initialize(int myperiodicmode_){myperiodicmode=myperiodicmode_;}
  inline void initialize(bool pX,  bool pY,  bool pZ)
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
//   template <class T >
// __device__ bool operator()(const T & x, const T &  y, const T & z) //throws a lot of warnings....because uint compares with 0 
  __device__ bool operator()(const int & x, const int &  y, const int & z)
  {
    // printf("%d %d %d %d \n",dcBoxXM1,dcBoxYM1,dcBoxZM1, myperiodicmode);
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
  
  __device__ bool operator()(const uint & x, const uint &  y, const uint & z)
  {
    // printf("%d %d %d %d \n",dcBoxXM1,dcBoxYM1,dcBoxZM1, myperiodicmode);
    switch(myperiodicmode)
    { 
	case periodic111: return  true              ;
	case periodic000: return ( x <  dcBoxXM1 &&
				   y <  dcBoxYM1 &&
				   z <  dcBoxZM1 )  ;  
	case periodic100: return ( y <  dcBoxYM1 &&
				   z <  dcBoxZM1 )  ;
	case periodic010: return ( x <  dcBoxXM1 &&
				   z <  dcBoxZM1 )  ;
	case periodic001: return ( x <  dcBoxXM1 &&
				   y <  dcBoxYM1 )  ;
	case periodic110: return ( z <  dcBoxZM1 )  ;
	case periodic011: return ( x <  dcBoxXM1 )  ;
	case periodic101: return ( y <  dcBoxYM1 )  ;
	default         : return false              ; //maybe throw an error?! 
    };
  }

};

#endif /* LEMONADEGPU_FEATURE_BOXCHECK_H_ */ 