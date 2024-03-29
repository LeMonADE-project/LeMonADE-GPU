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
#include <LeMonADE/feature/FeatureMoleculesIO.h>
#include <LeMonADE/feature/FeatureMoleculesIOUnsaveCheck.h>
#include <LeMonADE/feature/FeatureReactiveBonds.h>
#include <LeMonADE/feature/FeatureAttributes.h>
#include <LeMonADE/feature/FeatureExcludedVolumeSc.h>
#include <LeMonADE/feature/FeatureShearForce.h>
#include <LeMonADE/feature/FeatureLabel.h>
#include <LeMonADE/utility/TaskManager.h>
#include <LeMonADE/updater/UpdaterReadBfmFile.h>
#include <LeMonADE/updater/UpdaterSimpleSimulator.h>
#include <LeMonADE/feature/FeatureConnectionSc.h>
#include <LeMonADE/updater/UpdaterSwellBox.h>
#include <LeMonADE/feature/FeatureReactiveBonds.h>
#include <LeMonADEGPU/core/GPUScBFM_Tendomers.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp> // __FILENAME__

#include "../analyzer/AnalyzerCrossLinkMSD.h"
#include "../analyzer/AnalyzerMonomerMSD.h"
#include "../analyzer/AnalyzerSystemMSD.h"

#include "../analyzer/AnalyzerShearStrain.h"
#include "../analyzer/AnalyzerWriteBfmFileEachConfig.h"

void printHelp( void )
{
    std::stringstream msg;
    msg << "usage: ./SimulatorCUDAGPUScBFM_Tendomer [options]\n"
        << "\n"
        << "Simple Simulator for simulating tendomers\n"
        << "\n"
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
        << "    -b, --swelling-on <integer>\n"
        << "        Increase the box size if monomer touch the boundary.\n"
        << "    -o, --output <file path>\n"
        << "        All intermediate steps at each save-interval will be appended to this file even if it already exists\n"
        << "    -c, --each-config <integer>\n"
        << "        0: Append/newfile to the output and LastConfig.bfm\n"
        << "        1: Append/newfile to the output, LastConfig.bfm and each mcs into a single file\n"
        << "        2: Append/newfile to the output and each mcs into a single file\n"
        << "    -t, --shear-strain-file <file path>\n"
        << "        Name of the file for the shear strain vs mcs. "
	    << "    -d, --diagonal <integer> \n"
        << "        0: use standard moves for all monomers \n"
        << "        1: use diagonal moves for all monomers \n"
        // << "        2: use standard moves for all monomers in the elastic chain and diagonal moves for the pending chain \n"
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
    double max_mcs         = 0; /* how many Monte-Carlo steps to simulate */
    double save_interval   = 0;
    int      iGpuToUse       = 0;
    int      iRngToUse       = -1;
    std::string seedFileName = "";
    int diagonalMoves        = 0; 
    uint32_t boundarySize(0);
    int write_type(0);
    std::string outfile_shear("ShearStrain.dat");
    bool analyzeShearStrainON=false;
    bool densityCheckerOn=false;
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
                { "seeds"        , required_argument, 0, 'e' },
                { "gpu"          , required_argument, 0, 'g' },
                { "help"         , no_argument      , 0, 'h' },
                { "initial-state", required_argument, 0, 'i' },
                { "max-mcs"      , required_argument, 0, 'm' },
                { "output"       , required_argument, 0, 'o' },
                { "write_type"   , required_argument, 0, 'c' },
                { "outfile_shear", required_argument, 0, 't' },
                { "boundary"     , required_argument, 0, 'b' },
                { "rng"          , required_argument, 0, 'r' },
                { "save-interval", required_argument, 0, 's' },
		        { "diagonal"     , required_argument, 0, 'd' },
                { "densityCheckerOn", no_argument   , 0, 'a' },
                { 0, 0, 0, 0 }    // signal end of list
            };
            /* getopt_long stores the option index here. */
            int option_index = 0;
            int c = getopt_long( argc, argv, "e:g:hi:m:o:c:t:b:r:s:d:a", long_options, &option_index );

            if ( c == -1 )
                break;

            switch ( c )
            {
                case 'e': seedFileName  = std::string( optarg ); break;
                case 'h': printHelp(); return 0;
                case 'g': iGpuToUse     = std::atoi  ( optarg ); break;
                case 'i': infile        = std::string( optarg ); break;
                case 'm': max_mcs       = std::atol  ( optarg ); break;
                case 'o': outfile       = std::string( optarg ); break;
                case 't': outfile_shear = std::string( optarg );analyzeShearStrainON=true; break;
                case 'c': write_type    = std::atoi  ( optarg ); break;
                case 'r': iRngToUse     = std::atoi  ( optarg ); break;
                case 'b': boundarySize  = std::atoi  ( optarg ); break;
                case 's': save_interval = std::atol  ( optarg ); break;
        		case 'd': diagonalMoves = std::atoi  ( optarg ); break;
                case 'a': densityCheckerOn=true; break;
                break;
                default:
                    std::cerr << "Unknown option encountered: " << optarg << "\n";
                    return 1;
            }
        }

        /* seed the globally available random number generators */
        RandomNumberGenerators rng;
        if ( ! seedFileName.empty() )
            rng.seedAll( seedFileName );
        else
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
//         typedef LOKI_TYPELIST_4( FeatureMoleculesIO, FeatureAttributes<>,
//                                  FeatureExcludedVolumeSc<>, FeatureConnectionSc ) Features;
//         typedef LOKI_TYPELIST_5( FeatureMoleculesIOUnsaveCheck, FeatureAttributes<>,
//                                  FeatureExcludedVolumeSc<>, FeatureConnectionSc, FeatureLabel ) Features;
        typedef LOKI_TYPELIST_6( FeatureMoleculesIOUnsaveCheck, FeatureAttributes<>,
                                 FeatureExcludedVolumeSc<>, FeatureLabel, FeatureReactiveBonds, FeatureShearForce ) Features;
				 
        typedef ConfigureSystem< VectorInt3, Features, 8 > Config;
        typedef Ingredients< Config > Ing;
        Ing myIngredients;

        /**
         * note that TaskManager stores the pointer to the updater inside
         * a UpdaterObject class and the destructor of that class
         * calls the destructor of the updater, i.e., it would unfortunately
         * be wrong to manually call the destructor or even to allocate
         * it on the heap, i.e.:
         *   GPUScBFM_AB_Type<Ing> gpuBfm( myIngredients, save_interval, iGpuToUse );
         */
        auto const pUpdaterGpu = new GPUScBFM_Tendomers<Ing>( myIngredients, save_interval, diagonalMoves );
        pUpdaterGpu->setDensityCheckOn(densityCheckerOn);
        pUpdaterGpu->setGpu( iGpuToUse );
        pUpdaterGpu->activateLogging( "Error"     );
        pUpdaterGpu->activateLogging( "Stats"      );
        pUpdaterGpu->activateLogging( "Info"      );
	    pUpdaterGpu->activateLogging( "Check"      );

        TaskManager taskmanager;
        taskmanager.addUpdater( new UpdaterReadBfmFile<Ing>( infile, myIngredients,UpdaterReadBfmFile<Ing>::READ_LAST_CONFIG_SAVE ), 0 );
        //if swelling simulations are done the box size can be adjusted here
        if ( boundarySize > 0 )
          taskmanager.addUpdater( new UpdaterSwellBox<Ing>( myIngredients, 800, 32, boundarySize ));
        //updater for the moves of the monomers and labels 
        taskmanager.addUpdater( pUpdaterGpu );
        
// 	if (analyzeON)
// 	{
// 	  taskmanager.addAnalyzer( new AnalyzerSystemMSD   <Ing>( myIngredients, 0       ) );
// 	  taskmanager.addAnalyzer( new AnalyzerMonomerMSD  <Ing>( myIngredients, 0       ) );
// 	  taskmanager.addAnalyzer( new AnalyzerCrossLinkMSD<Ing>( myIngredients, 0       ) );
// 	}
        //analyer for the shear strain (on the fly )
        if (analyzeShearStrainON)
            taskmanager.addAnalyzer(new AnalyzerShearStrain<Ing>(myIngredients, outfile_shear) );
        //append/newfile to outfile
        taskmanager.addAnalyzer( new AnalyzerWriteBfmFile<Ing>( outfile, myIngredients, AnalyzerWriteBfmFile<Ing>::APPEND ) );

        // erase the '.bfm' suffix from the outfile name to be used in the AanlyzerWriteBFM EachConfig 
        std::string toErase("_MCS");
        std::string basename(outfile);
        size_t pos = basename.find(toErase);
        std::cout << "pos=" << pos << " basename=" << basename << " length=" << basename.length()-pos <<std::endl;
        if (pos != std::string::npos) basename.erase(pos, basename.length()-pos);
        std::cout << "basename=" <<basename <<std::endl;
        //append analyzer to the program:
        switch (write_type){
            case 0 : taskmanager.addAnalyzer( new AnalyzerWriteBfmFile<Ing>( "LastConfig.bfm", myIngredients, AnalyzerWriteBfmFile<Ing>::OVERWRITE ) ); 
                     break; 
            case 1 : taskmanager.addAnalyzer( new AnalyzerWriteBfmFile<Ing>( "LastConfig.bfm", myIngredients, AnalyzerWriteBfmFile<Ing>::OVERWRITE ) ); 
                     taskmanager.addAnalyzer( new AnalyzerWriteBfmFileEachConfig<Ing>( basename , myIngredients ) );
                     break; 
            case 2 : taskmanager.addAnalyzer( new AnalyzerWriteBfmFileEachConfig<Ing>( basename , myIngredients ) );
                     break; 
        }

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