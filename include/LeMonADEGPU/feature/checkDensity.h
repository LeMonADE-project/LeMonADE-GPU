
#ifndef LEMONADEGPU_FEATURE_CHECK_DENSITY_H
#define LEMONADEGPU_FEATURE_CHECK_DENSITY_H
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <cuda_profiler_api.h>              // cudaProfilerStop
typedef uint8_t intArray;
typedef uint32_t boxType;
//device constants 
//! average number of monomers in 4 slices in the middle and in the boundaries
__device__ __constant__ float dAvMonomerNumberInShearVolume; 
__device__ __constant__ boxType dBoxX;
__device__ __constant__ boxType dBoxY;
__device__ __constant__ boxType dBoxZ;
//d-device 
//M-minus
//P-plu
//h-half
__device__ __constant__ boxType dBoxZM1;
__device__ __constant__ boxType dBoxZM2;
__device__ __constant__ boxType dBoxZM3;
__device__ __constant__ boxType dBoxZhP1;
__device__ __constant__ boxType dBoxZhP2;
__device__ __constant__ boxType dBoxZhM3;
__device__ __constant__ boxType dBoxZhM2;
__device__ __constant__ boxType dBoxZhP3;

/**
 * @file checkDensity.h 
 * @class checkDensity
 * @brief used to get a constant density during shear
 * @details In the shear simulations a force in x-direction is applied in two 
 * layers at the top and the bottom and in opposite direction in four layers in 
 * the middle of the layer. For same cases therer is a reduced or enhanced 
 * density inside these layers, which we would identify as an artefact. Thus, there
 * is a need to held the number of monomers close to the average number to ensure 
 * a useful simulation.
 * This class uses a reduction algorithm which ist explained in 
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
template <typename T_UCoordinateCuda > class checkDensity
{
public:
    //type for unsigned coordinates on the device 
    using T_UCoordinatesCuda = typename CudaVec4< T_UCoordinateCuda >::value_type;
	/** constructor 
	 */
    checkDensity();
	checkDensity(uint32_t BoxX_, uint32_t BoxY_, uint32_t BoxZ_);
	/**
	 * 
	 */
    __device__ bool operator()( uint8_t z, const int32_t & dz) {
        if ( !checkOn) return true; 
        z=z%dBoxZ;
        if(dz == 1){
            if(*dMonomerNumber_in_ShearVolumeBoundary < dAvMonomerNumberInShearVolume ){ if(z == 1) return false ;}
            else{ if(z == dBoxZM3) return false ;}
            if(*dMonomerNumber_in_ShearVolumeMiddle < dAvMonomerNumberInShearVolume){if(z == dBoxZhP2)return false ;}
            else {if (z == dBoxZhM3) return false ;}
        }else if(dz == -1){
            if(*dMonomerNumber_in_ShearVolumeBoundary < dAvMonomerNumberInShearVolume ){ if(z == dBoxZM2) return false ;}
            else{ if(z == 2) return false ;}
            if(*dMonomerNumber_in_ShearVolumeMiddle < dAvMonomerNumberInShearVolume){if(z == dBoxZhM2 )return false ;}
            else {if (z == dBoxZhP3) return false ;}
        }else 
            return true;  
    }
    //!
    void launch_countMonomers( 
        MirroredVector<T_UCoordinatesCuda> * mpPolymerSystem,
        uint32_t offset,
        size_t nMons,
        size_t const nBlocksSpecies,
        uint32_t const nThreadsSprecies );
	//! 
    void init(uint32_t NMonomers_, uint32_t nSortedMonomers, cudaDeviceProp&  mCudaProps);
	//! 
    void calcDensity(void); 
	//! 
	void setBoxSizes(uint32_t BoxX_, uint32_t BoxY_, uint32_t BoxZ_);
    //!
    void setDensityCheckON(bool checkOn_) {checkOn=checkOn_;}
private:
	//! count the number of monomers in the middle
	MirroredVector<intArray>*  mCountMiddleMonos;
	//! count the number of monomers in the boundary
	MirroredVector<intArray> * mCountBoundaryMonos;
	//!array after the reduction used primarily for the result: middle monomers 
	MirroredVector<intArray> * mReducedCountMiddleMonos;
	//!array after the reduction used primarily for the result: boundary monomers
	MirroredVector<intArray> * mReducedCountBoundaryMonos;
	//! number of threads 
	uint32_t threads;
	//! size of the mirrored vectors 
	uint32_t arraySize;
	//! number of 
	uint32_t hMonomerNumber_in_ShearVolumeMiddle;
	//!
	uint32_t hMonomerNumber_in_ShearVolumeBoundary;
	//!
	uint32_t * dMonomerNumber_in_ShearVolumeMiddle;
	//!
	uint32_t * dMonomerNumber_in_ShearVolumeBoundary;
	//! box sizes in the three directions
	boxType BoxX,BoxY,BoxZ; 
    //! number of monomers in the simulation box 
    uint32_t NMonomers;
    //! set the check on or off
    bool checkOn;
	//! number of blocks for the kernel which performs the reduction
	uint32_t nBlocks;
	//! number of threads for the kernel which performs the reduction
	uint32_t nThreads;
};
#endif /*LEMONADEGPU_FEATURE_CHECK_DENSITY_H*/