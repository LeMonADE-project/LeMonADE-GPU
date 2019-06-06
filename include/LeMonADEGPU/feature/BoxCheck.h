
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