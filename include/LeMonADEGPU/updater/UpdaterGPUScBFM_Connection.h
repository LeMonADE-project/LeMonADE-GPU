#ifndef LEMONADEGPU_UPDATER_GPUCONNECTION_H_
#define LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>

template< typename T_UCoordinateCuda > 
class UpdaterGPUScBFM_Connection: public UpdaterGPUScBFM_AB_Type<T_UCoordinateCuda>
{
public:
  typedef UpdaterGPUScBFM_AB_Type<T_UCoordinateCuda> BaseClass;
  using T_Flags            = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Flags      ;
  using T_Lattice          = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Lattice    ;
  using T_Coordinate       = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Coordinate ;
  using T_Coordinates      = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Coordinates;
  using T_Id               = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Id         ;
  using BaseClass::mLog;

  UpdaterGPUScBFM_Connection();
  void initialize();
  
  
  void setReactiveGroup(std::vector<std::vector<T_Id> >);
//   void setMaximumConnectivity()
    
private:
  //create a lattice with the ids on the edges
  MirroredTexture< T_Id > * mLatticeIds;
  
};


#endif 


