
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
#include <LeMonADE/utility/TaskManager.h>
#include <LeMonADE/updater/UpdaterReadBfmFile.h>
#include <LeMonADE/updater/UpdaterSimpleSimulator.h>
#include <LeMonADE/feature/FeatureConnectionSc.h>


#include <LeMonADEGPU/core/GPUScBFM_AA_ReversibleConnection.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp> // __FILENAME__

#include "../analyzer/AnalyzerCrossLinkMSD.h"
#include "../analyzer/AnalyzerMonomerMSD.h"
#include "../analyzer/AnalyzerSystemMSD.h"

void printHelp( void )
{
    std::stringstream msg;
    msg << "usage: ./SimulatorCUDAGPUScBFM_AA_Connection [options]\n"
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
	<< "    -b, --energy <value>\n"
        << "        bond energy between connected (reactive) monomers. \n"
	<< "    -a, --analyze-MSD-ON <bool 0/1>\n"
        << "        Analyzes the MSD of the whole system, the cross links and all monomers.\n"
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
    double energy = 10.0; 
    bool analyzeON=false;
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
		{ "energy"       , required_argument, 0, 'b' },
		{ "analyze"      , required_argument, 0, 'a' },
                { "save-interval", required_argument, 0, 's' },
                { 0, 0, 0, 0 }    // signal end of list
            };
            /* getopt_long stores the option index here. */
            int option_index = 0;
            int c = getopt_long( argc, argv, "c:e:g:hi:m:o:b:a:s:", long_options, &option_index );

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
		case 'b': energy        = std::atof  ( optarg ); break;
		case 'a': analyzeON     = std::atoi  ( optarg ); break;
                case 's': save_interval = std::atol  ( optarg ); break;
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
        typedef LOKI_TYPELIST_4( FeatureMoleculesIOUnsaveCheck, FeatureAttributes<>,
                                 FeatureExcludedVolumeSc<>, FeatureConnectionSc ) Features;

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
        auto const pUpdaterGpu = new GPUScBFM_AB_ReversibleConnection<Ing>( myIngredients, save_interval );
        pUpdaterGpu->setGpu( iGpuToUse );
        pUpdaterGpu->activateLogging( "Error"     );
        //pUpdaterGpu->activateLogging( "Stats"      );
        pUpdaterGpu->activateLogging( "Info"      );
// 	pUpdaterGpu->activateLogging( "Check"      );
	pUpdaterGpu->setBondEnergy(energy);
        if ( nSplitColors > 0 )
            pUpdaterGpu->setSplitColors( nSplitColors );

        TaskManager taskmanager;
        taskmanager.addUpdater( new UpdaterReadBfmFile<Ing>( infile, myIngredients,UpdaterReadBfmFile<Ing>::READ_LAST_CONFIG_SAVE ), 0 );
        //here you can choose to use MoveLocalBcc instead. Careful though: no real tests made yet
        //(other than for latticeOccupation, valid bonds, frozen monomers...)
        taskmanager.addUpdater( pUpdaterGpu );
	if (analyzeON)
	{
	  taskmanager.addAnalyzer( new AnalyzerSystemMSD   <Ing>( myIngredients, 0       ) );
	  taskmanager.addAnalyzer( new AnalyzerMonomerMSD  <Ing>( myIngredients, 0       ) );
	  taskmanager.addAnalyzer( new AnalyzerCrossLinkMSD<Ing>( myIngredients, 0       ) );
	}
        taskmanager.addAnalyzer( new AnalyzerWriteBfmFile<Ing>( outfile, myIngredients ) );

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