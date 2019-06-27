#ifndef LEMONADEGPU_CORE_METHOD_H_
#define LEMONADEGPU_CORE_METHOD_H_
#include <LeMonADEGPU/core/SpaceFillingCurve.h>
#include <LeMonADEGPU/core/BitPacking.h>

/**
 * @brief convinience class storing methods which are used in the kernel 
 */
class Method {
 
 private:    
     SpaceFillingCurve curve; 
  
 public:
    __device__ __host__ const SpaceFillingCurve&   getCurve() const {return curve; }

   __device__ __host__ SpaceFillingCurve&   modifyCurve() {return curve; }
   
  private:
      BitPacking packing;
  public:
    __device__ __host__ const BitPacking& getPacking() const {return packing; }
    __device__ __host__ BitPacking& modifyPacking() {return packing;}
  private:
    bool useGPUForOverhead;
  public:
    bool isONGPUForOverhead() const  {return useGPUForOverhead;}
    void setOnGPUForOverhead(const bool useGPUForOverhead_){useGPUForOverhead=useGPUForOverhead_;}
  
};
#endif /*LEMONADEGPU_CORE_METHOD_H_*/