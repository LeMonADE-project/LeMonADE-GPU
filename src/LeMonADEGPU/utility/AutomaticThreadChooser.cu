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

#include <LeMonADEGPU/utility/AutomaticThreadChooser.h>
#include <limits>
#include <algorithm>
#include <cuda_profiler_api.h>              // cudaProfilerStop

AutomaticThreadChooser::AutomaticThreadChooser( const uint32_t nSpecies ):
benchmarkInfo( nSpecies, SpeciesBenchmarkData{
    std::vector< std::vector< float > >( 2 /* true or false */,
        std::vector< float >( vnThreadsToTry.size(),
            std::numeric_limits< float >::infinity() ) ),
    std::vector< std::vector< int   > >( 2 /* true or false */,
        std::vector< int   >( vnThreadsToTry.size(),
        2 /* repeat 2 time, i.e. execute three times */ ) ),
    2, true, true, true
} ), needToBenchmark(true){
    mLog.file( __FILENAME__ );
    mLog.deactivate( "Check"     );
    mLog.deactivate( "Error"     );
    mLog.deactivate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
};

void AutomaticThreadChooser::initialize( const cudaDeviceProp& mCudaProps){
    for ( auto nThreads = mCudaProps.warpSize; nThreads <= mCudaProps.maxThreadsPerBlock; nThreads *= 2 )
        vnThreadsToTry.push_back( nThreads );
    cudaEventCreate( &tOneGpuLoop0 );
    cudaEventCreate( &tOneGpuLoop1 );
}

void AutomaticThreadChooser::addRecord(const uint32_t iSpecies, const cudaStream_t& mStream ){
    needToBenchmark = ! (
        benchmarkInfo[ iSpecies ].decidedAboutThreadCount &&
        benchmarkInfo[ iSpecies ].decidedAboutCudaMemset );
    if ( needToBenchmark )
        cudaEventRecord( tOneGpuLoop0, mStream );
}

void AutomaticThreadChooser::analyze(const uint32_t iSpecies, const cudaStream_t& mStream){
    if ( needToBenchmark ){
	auto const useCudaMemset = benchmarkInfo[ iSpecies ].useCudaMemset;
        auto & info = benchmarkInfo[ iSpecies ];
        cudaEventRecord( tOneGpuLoop1, mStream );
        cudaEventSynchronize( tOneGpuLoop1 );
        float milliseconds = 0;
        cudaEventElapsedTime( & milliseconds, tOneGpuLoop0, tOneGpuLoop1 );
        auto const iThreadCount = info.iBestThreadCount;
        auto & oldTiming = info.timings.at( useCudaMemset ).at( iThreadCount );
        oldTiming = std::min( oldTiming, milliseconds );

        mLog( "Info" )
        << "Using " << getBestThread(iSpecies) << " threads " 
        << " and using " << ( useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
        << " for species " << iSpecies << " took " << milliseconds << "ms\n";

        auto & nStillToRepeat = info.nRepeatTimings.at( useCudaMemset ).at( iThreadCount );
        if ( nStillToRepeat > 0 )
            --nStillToRepeat;
        else if ( ! info.decidedAboutThreadCount )
        {
            /* if not the first timing, then decide whether we got slower,
                * i.e. whether we found the minimum in the last step and
                * have to roll back */
            if ( iThreadCount > 0 )
            {
                if ( info.timings.at( useCudaMemset ).at( iThreadCount-1 ) < milliseconds )
                {
                    --info.iBestThreadCount;
                    info.decidedAboutThreadCount = true;
                }
                else
                    ++info.iBestThreadCount;
            }
            else
                ++info.iBestThreadCount;
            /* if we can't increment anymore, then we are finished */
            
            if ( (size_t) info.iBestThreadCount == vnThreadsToTry.size() )
            {
                --info.iBestThreadCount;
                info.decidedAboutThreadCount = true;
            }
            if ( info.decidedAboutThreadCount )
            {
                /* then in the next term try out changing cudaMemset
                    * version to custom kernel version (or vice-versa) */
                if ( ! info.decidedAboutCudaMemset )
                    info.useCudaMemset = ! info.useCudaMemset;
                mLog( "Info" )
                << "Using " << vnThreadsToTry.at( info.iBestThreadCount )
                << " threads per block is the fastest for species "
                << iSpecies << ".\n";
            }
        }
        else if ( ! info.decidedAboutCudaMemset )
        {
            info.decidedAboutCudaMemset = true;
            if ( info.timings.at( ! useCudaMemset ).at( iThreadCount ) < milliseconds )
                info.useCudaMemset = ! useCudaMemset;
            if ( info.decidedAboutCudaMemset )
            {
                mLog( "Info" )
                << "Using " << ( info.useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
                << " is the fastest for species " << iSpecies << ".\n";
            }
        }
    }
    needToBenchmark=true;
}