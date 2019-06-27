#ifndef LEMONADEGPU_UPDATER_GPUCONNECTION_H_
#define LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>

template< typename T_UCoordinateCuda > 
class UpdaterGPUScBFM_Connection: public UpdaterGPUScBFM_AB_Type<T_UCoordinateCuda>
{
  typedef UpdaterGPUScBFM_AB_Type<T_UCoordinateCuda> BaseClass;
  
public:
    UpdaterGPUScBFM_Connection();
    
    using BaseClass::mLog;
};


#endif 


