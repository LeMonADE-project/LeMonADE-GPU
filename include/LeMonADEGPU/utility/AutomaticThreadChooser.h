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

#ifndef LEMONADEGPU_UTILITY_AUTOMATICTHREADCHOOSES_H_
#define LEMONADEGPU_UTILITY_AUTOMATICTHREADCHOOSES_H_
#include <vector>
#include <stdint.h>
#include <cuda_profiler_api.h>              // cudaProfilerStop
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>

/**
     * Logic for determining the best threadsPerBlock configuration
     *
     * This might be dependent on the species, therefore for each species
     * store the current best thread count and all timings.
     * As the cudaEventSynchronize timings are expensive, stop benchmarking
     * after having found the best configuration.
     * Only try out power two multiples of warpSize up to maxThreadsPerBlock,
     * e.g. 32, 64, 128, 256, 512, 1024, because smaller than warp size
     * should never lead to a speedup and non power of twos, e.g. 196 even,
     * won't be able to perfectly fill out the shared multi processor.
     * Also, automatically determine whether cudaMemset is faster or not (after
     * we found the best threads per block configuration
     * note: test example best configuration was 128 threads per block and use
     *       the cudaMemset version instead of the third kernel
     */
class AutomaticThreadChooser
{
public:
    AutomaticThreadChooser(const uint32_t nSpecies);
    void initialize(const cudaDeviceProp& mCudaProps);
    void addRecord(const uint32_t iSpecies, const cudaStream_t& mStream  );
    void analyze(const uint32_t iSpecies, const cudaStream_t& mStream);
    inline uint32_t getBestThread(const uint32_t iSpecies ){return vnThreadsToTry.at( benchmarkInfo[ iSpecies ].iBestThreadCount );}
    inline bool useCudaMemset(const uint32_t iSpecies) {return benchmarkInfo[ iSpecies ].useCudaMemset;}
private:
    //from 32...1024 in powers of 2
    std::vector< int > vnThreadsToTry;

    bool needToBenchmark;

    struct SpeciesBenchmarkData
    {
        /* 2 vectors of double for measurements with and without cudaMemset */
        std::vector< std::vector< float > > timings;
        std::vector< std::vector< int   > > nRepeatTimings;
        int iBestThreadCount;
        bool useCudaMemset;
        bool decidedAboutThreadCount;
        bool decidedAboutCudaMemset;
    };
    //gathered information for each species 
    std::vector< SpeciesBenchmarkData > benchmarkInfo; 
    //
    cudaEvent_t tOneGpuLoop0, tOneGpuLoop1;

    SelectedLogger mLog;

};

#endif /*LEMONADEGPU_UTILITY_AUTOMATICTHREADCHOOSES_H_ */

