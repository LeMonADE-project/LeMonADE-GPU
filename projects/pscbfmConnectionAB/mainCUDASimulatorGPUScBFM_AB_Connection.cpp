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
#include <chrono>                       // std::chrono::high_resolution_clock
#include <cstring>
//#include <cstdint>                      // uint32_t (C++11)
#include <iostream>
#include <sstream>

#include <getopt.h>

#include <LeMonADE/utility/RandomNumberGenerators.h>
#include <LeMonADE/core/ConfigureSystem.h>
#include <LeMonADE/core/Ingredients.h>
#include <LeMonADE/feature/FeatureMoleculesIOUnsaveCheck.h>
#include <LeMonADE/feature/FeatureAttributes.h>
#include <LeMonADE/feature/FeatureExcludedVolumeSc.h>
#include <LeMonADE/feature/FeatureFixedMonomers.h>
#include <LeMonADE/feature/FeatureSystemInformationLinearMeltWithCrosslinker.h>
#include <LeMonADE/utility/TaskManager.h>
#include <LeMonADE/updater/UpdaterReadBfmFile.h>
#include <LeMonADE/updater/UpdaterSimpleSimulator.h>
#include <LeMonADE/feature/FeatureReactiveBonds.h>


#include <LeMonADEGPU/core/GPUScBFM_AB_Connection.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp> // __FILENAME__



void printHelp( void )
{
    std::stringstream msg;
    msg << "usage: ./SimulatorCUDAGPUScBFM_AB_Connection [options]\n"
        << "\n"
        << "Simple Simulator for the ScBFM with excluded volume and BondCheck splitted CL-PEG in z on GPU\n"
        << "\n"
        << "    -c, --colors <integer>\n"
        << "        Artificially increase the number of colors n times by splitting each color into two new ones. Therefore an argument of 3 will result in 8 times as man colors as actually needed. The default corresponds to argument 0.\n"
        << "    -e, --seeds <file path>\n"
        << "        Specify a seed file to use for reproducible result simulations. Currently this should contain 256+2 random 32-Bit values (or more).\n"
        << "    -g, --gpu <integer>\n"
        << "        Specify the GPU to use. The ID goes from 0 to the number of GPUs installed - 1\n"
        << "    -i, --initial-state <file path>\n"
        << "        (required) specify a BFM file to load the configuration to simulate from\n"
        << "    -m, --max-mcs <integer>\n"
        << "        (required) specifies the total Monte-Carlo steps to simulate.\n"
        << "    -s, --save-interval <integer>\n"
        << "        Save after every <integer> Monte-Carlo steps to the output file.\n"
        << "    -o, --output <file path>\n"
        << "        All intermediate steps at each save-interval will be appended to this file even if it already exists\n"
        << "    -v, --version\n"
        ;
    std::cout << msg.str();
}

int main( int argc, char ** argv )
{
    using hrclock = std::chrono::high_resolution_clock;
    auto const tProgram0 = hrclock::now();

    std::string infile; /* BFM file containing positions of monomers and their connections */
    std::string outfile      = "outfile.bfm"; /* at save_interval steps the current state of the simulation will be written to this file */
    uint32_t max_mcs         = 0; /* how many Monte-Carlo steps to simulate */
    uint32_t save_interval   = 0;
    int      iGpuToUse       = 0;
    std::string seedFileName = "";
    int      nSplitColors    = 0;

    try
    {

        if ( argc <= 1 )
        {
            printHelp();
            return 0;
        }

        /* Parse command line arguments */
        while ( true )
        {
            static struct option long_options[] = {
                { "colors"       , required_argument, 0, 'c' },
                { "seeds"        , required_argument, 0, 'e' },
                { "gpu"          , required_argument, 0, 'g' },
                { "help"         , no_argument      , 0, 'h' },
                { "initial-state", required_argument, 0, 'i' },
                { "max-mcs"      , required_argument, 0, 'm' },
                { "output"       , required_argument, 0, 'o' },
                { "rng"          , required_argument, 0, 'r' },
                { "save-interval", required_argument, 0, 's' },
                { 0, 0, 0, 0 }    // signal end of list
            };
            /* getopt_long stores the option index here. */
            int option_index = 0;
            int c = getopt_long( argc, argv, "c:e:g:hi:m:o:r:s:", long_options, &option_index );

            if ( c == -1 )
                break;

            switch ( c )
            {
                case 'c': nSplitColors  = std::atoi  ( optarg ); break;
                case 'e': seedFileName  = std::string( optarg ); break;
                case 'h': printHelp(); return 0;
                case 'g': iGpuToUse     = std::atoi  ( optarg ); break;
                case 'i': infile        = std::string( optarg ); break;
                case 'm': max_mcs       = std::atol  ( optarg ); break;
                case 'o': outfile       = std::string( optarg ); break;
                case 's': save_interval = std::atol  ( optarg ); break;
                    break;
                default:
                    std::cerr << "Unknown option encountered: " << optarg << "\n";
                    return 1;
            }
        }

        /* seed the globally available random number generators */
        RandomNumberGenerators rng;
//         if ( ! seedFileName.empty() )
//             rng.seedAll( seedFileName );
//         else
            rng.seedAll();

        /* Check the initial values. Note that the drawing of these random
         * values can't be omitted, or else all subsequent random numbers
         * will shift / change! */
        std::rand      ();
        rng.r250_rand32();
        rng.r250_drand ();

        /*
        FeatureExcludedVolume<> is equivalent to FeatureExcludedVolume< FeatureLattice< bool > >
        typedef LOKI_TYPELIST_3( FeatureBondset, FeatureAttributes,
            FeatureLattice< uint8_t > FeatureExcludedVolume< FeatureLatticePowerOfTwo<> > )
            Features;
        */
        typedef LOKI_TYPELIST_5( FeatureMoleculesIOUnsaveCheck, FeatureAttributes<>,
                                 FeatureExcludedVolumeSc<>, FeatureReactiveBonds,
                                 FeatureSystemInformationLinearMeltWithCrosslinker ) Features;

        typedef ConfigureSystem< VectorInt3, Features, 8 > Config;
        typedef Ingredients< Config > Ing;
        Ing myIngredients;

        /**
         * note that TaskManager stores the pointer to the updater inside
         * a UpdaterObject class and the destructor of that class
         * calls the destructor of the updater, i.e., it would unfortunately
         * be wrong to manually call the destructor or even to allocate
         * it on the heap, i.e.:
         *   GPUScBFM<Ing> gpuBfm( myIngredients, save_interval, iGpuToUse );
         */
        auto const pUpdaterGpu = new GPUScBFM_AB_Connection<Ing>( myIngredients, save_interval );
        pUpdaterGpu->setGpu( iGpuToUse );
        pUpdaterGpu->activateLogging( "Error"     );
        //pUpdaterGpu->activateLogging( "Stats"      );
        pUpdaterGpu->activateLogging( "Info"      );
        if ( nSplitColors > 0 )
            pUpdaterGpu->setSplitColors( nSplitColors );

        TaskManager taskmanager;
        taskmanager.addUpdater( new UpdaterReadBfmFile<Ing>( infile, myIngredients,UpdaterReadBfmFile<Ing>::READ_LAST_CONFIG_SAVE ), 0 );
        //here you can choose to use MoveLocalBcc instead. Careful though: no real tests made yet
        //(other than for latticeOccupation, valid bonds, frozen monomers...)
        taskmanager.addUpdater( pUpdaterGpu );

        taskmanager.addAnalyzer( new AnalyzerWriteBfmFile<Ing>( outfile, myIngredients ) );
        taskmanager.addAnalyzer( new AnalyzerWriteBfmFile<Ing>( "LastConfig.bfm", myIngredients, AnalyzerWriteBfmFile<Ing>::OVERWRITE ) );

        taskmanager.initialize();

        auto const tTasks0 = hrclock::now();
        taskmanager.run( max_mcs / save_interval );

        auto const tTasks1 = hrclock::now();
        std::stringstream sBuffered;
        sBuffered << "tTaskLoop = " << std::chrono::duration<double>( tTasks1 - tTasks0 ).count() << "s\n";
        std::cerr << sBuffered.str();

        taskmanager.cleanup();
    }
    catch( std::exception const & e )
    {
        std::cerr << "[" << __FILENAME__ << "] Caught exception: " << e.what() << std::endl;;
    }

    auto const tProgram1 = hrclock::now();
    std::stringstream sBuffered;
    sBuffered << "tProgram = " << std::chrono::duration<double>( tProgram1 - tProgram0 ).count() << "s\n";
    std::cerr << sBuffered.str();
    return 0;
}
