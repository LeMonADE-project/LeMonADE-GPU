

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