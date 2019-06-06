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