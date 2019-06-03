
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
#include <LeMonADE/analyzer/AnalyzerWriteBfmFile.h>


#include <LeMonADEGPU/core/GPUScBFMGPR_AB_Type.h>
#include <LeMonADEGPU/core/GPUScBFM_AB_Type.h>
// #include "FeatureSystemInformation.h"
// #include "FeatureShearForce.h"
// #include "AnalyzerWriteBfmFileEachFrame.h"
// #include "AnalyzerShearStrain.h"
#include <LeMonADEGPU/utility/SelectiveLogger.hpp> // __FILENAME__



void printHelp( void )
{
    std::stringstream msg;
    msg << "usage: ./SimulatorCUDAGPUScBFM_AB_Type [options]\n"
        << "\n"
        << "Simple Simulator for the ScBFM with excluded volume and BondCheck splitted CL-PEG in z on GPU\n"
        << "\n"
//         << "    -e, --seeds <file path>\n"
//         << "        specify a seed file to use for reproducible result simulations. Currently this should contain 256+2 random 32-Bit values (or more).\n"
        << "    -i, --initial-state <file path>\n"
        << "        (required) specify a BFM file to load the configuration to simulate from\n"
        << "    -m, --max-mcs <integer>\n"
        << "        (required) specifies the total Monte-Carlo steps to simulate.\n"
        << "    -s, --save-interval <integer>\n"
        << "        save after every <integer> Monte-Carlo steps to the output file.\n"
        << "    -o, --output <file path>\n"
        << "        all intermediate steps at each save-interval will be appended to this file even if it already exists\n"
        << "    -g, --gpu <integer>\n"
        << "        specify the GPU to use. The ID goes from 0 to the number of GPUs installed - 1\n"
        << "    -v, --disable_excluded_volume \n"
        << "        if given, then disable the excluded volume.\n"
        << "    -d, --enable_diagonal_moves \n"
        << "        if given, then use diagonal moves instead of standard moves.\n"
	<< "    -r, --enable_sliding \n"
        << "        if given, then single monomer (representing rings) can move along the chain backbone\n"
	<< "    -f, --shear_force \n"
        << "        the shear force applied on the two uppermost and lowermost planes and in opposite direction in four planes\n"
	<< "    -l, --strain_output \n"
        << "        filename for the output datat of the strain analysis.\n";
    std::cout << msg.str();
}

template <class taskman, class myIng>
void addtasks(taskman& mytaskman, myIng& ingredients, uint32_t interval, uint32_t max, std::string out , std::string in, bool slide, bool diagonal,bool ShearForceOn, double  ShearForce, bool analyzeStrain, std::string outfileStrain);

int main( int argc, char ** argv )
{
    using hrclock = std::chrono::high_resolution_clock;
    auto const tProgram0 = hrclock::now();

    std::string infile; /* BFM file containing positions of monomers and their connections */
    std::string outfile       = "outfile.bfm"; /* at save_interval steps the current state of the simulation will be written to this file */
    uint32_t max_mcs          = 0; /* how many Monte-Carlo steps to simulate */
    uint32_t save_interval    = 0;
    int      iGpuToUse        = 0;
    bool     EVON             = true;
    bool     DiagonalMovesON  = false; 
    bool     sliding          = false; 
    bool     ShearForceOn     = false;
    double   ShearForce       = 0.0; 
    std::string outfileStrain = "ShearStrain.dat";
    bool analyzeStrain        = false;
    std::string seedFileName  = "";

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
                { "seeds"                   , required_argument, 0, 'e' },
                { "gpu"                     , required_argument, 0, 'g' },
                { "help"                    , no_argument      , 0, 'h' },
                { "initial-state"           , required_argument, 0, 'i' },
                { "max-mcs"                 , required_argument, 0, 'm' },
                { "output"                  , required_argument, 0, 'o' },
                { "save-interval"           , required_argument, 0, 's' },
		{ "disable_excluded_volume" , no_argument      , 0, 'v' },
		{ "enable_diagonal_moves"   , no_argument      , 0, 'd' },
		{ "enable_sliding"          , no_argument      , 0, 'r' },
                { "shear_force"             , required_argument, 0, 'f' },
		{ "strain_output"           , required_argument, 0, 'l' },
		{ 0, 0, 0, 0 }    // signify end of list
            };
            /* getopt_long stores the option index here. */
            int option_index = 0;
            int c = getopt_long( argc, argv, "g:hi:m:o:s:vdrf:l:", long_options, &option_index );

            if ( c == -1 )
                break;

            switch ( c )
            {
                case 'e':
                    seedFileName = std::string( optarg );
                    break;
                case 'h':
                    printHelp();
                    return 0;
                case 'g':
                    iGpuToUse = std::atoi( optarg );
                    break;
                case 'i':
                    infile = std::string( optarg );
                    break;
                case 'm':
                    max_mcs = std::atol( optarg );
                    break;
                case 'o':
                    outfile = std::string( optarg );
                    break;
                case 's':
                    save_interval = std::atol( optarg );
                    break;
		case 'v':
                    EVON = false;
                    break;
		case 'd':
                    DiagonalMovesON = true;
                    break;
		case 'r':
                    sliding = true;
                    break;
		case 'f':
		    char* pEnd;
		    ShearForceOn=true;
                    ShearForce = std::strtod( optarg, &pEnd );
                    break;
		case 'l':
		    analyzeStrain = true;
		    outfileStrain = std::string( optarg );
                    break;
		default:
                    std::cerr << "Unknown option encountered: " << optarg << "\n";
                    return 1;
            }
        }

        std::cerr
        << "infile          = " << infile        << "\n"
        << "outfile         = " << outfile       << "\n"
        << "max_mcs         = " << max_mcs       << "\n"
        << "save_interval   = " << save_interval << "\n"
	<< "excluded volume = " << EVON          << "\n"
	<< "ring sliding    = " << sliding       << "\n"
	<< "shear force     = " << ShearForce    << "\n"
	<< "shear force on  = " << ShearForceOn  << "\n";

        //seed the globally available random number generators
        RandomNumberGenerators rng;
        if ( ! seedFileName.empty() )
        {
            std::cerr << "Use seeds from: " << seedFileName << "\n";
            rng.seedAll( seedFileName );
        }
        else
            rng.seedAll();

        /* Check the initial values. Note that the drawing of these random
         * values can't be omitted, or else all subsequent random numbers
         * will shift / change! */
        std::cerr << "std rand: " << std::setw(12) << std::rand()       << " =?= 764080779"  << "\n";
        std::cerr << "RNG rand: " << std::setw(12) << rng.r250_rand32() << " =?= 4223731124" << "\n";
        std::cerr << "RNG rand: " << std::setw(12) << rng.r250_drand()  << " =?= 0.803876"   << "\n";

// 	typedef LOKI_TYPELIST_5( FeatureMoleculesIOUnsaveCheck, FeatureAttributes<>, FeatureExcludedVolumeSc<>, FeatureSystemInformation, FeatureShearForce ) Features;
//         typedef ConfigureSystem< VectorInt3, Features, 8 > Config;
//         typedef Ingredients< Config > Ing;
//         Ing myIngredients;
// 	typedef LOKI_TYPELIST_4( FeatureMoleculesIOUnsaveCheck, FeatureAttributes<>,  FeatureSystemInformation, FeatureShearForce ) FeatureswoEV;
// 	typedef ConfigureSystem< VectorInt3, FeatureswoEV, 8 > ConfigwoEV;
//         typedef Ingredients< ConfigwoEV > IngwoEV;
//         IngwoEV myIngredientswoEV;
	
	typedef LOKI_TYPELIST_3( FeatureMoleculesIOUnsaveCheck, FeatureAttributes<>, FeatureExcludedVolumeSc<>   ) Features;
        typedef ConfigureSystem< VectorInt3, Features, 8 > Config;
        typedef Ingredients< Config > Ing;
        Ing myIngredients;
	typedef LOKI_TYPELIST_2( FeatureMoleculesIOUnsaveCheck, FeatureAttributes<>    ) FeatureswoEV;
	typedef ConfigureSystem< VectorInt3, FeatureswoEV, 8 > ConfigwoEV;
        typedef Ingredients< ConfigwoEV > IngwoEV;
        IngwoEV myIngredientswoEV;
        TaskManager taskmanager;
	if (EVON) 
	  addtasks(taskmanager, myIngredients,save_interval, max_mcs, outfile, infile, sliding, DiagonalMovesON, ShearForceOn, ShearForce,  analyzeStrain,  outfileStrain);
	else 
	  addtasks(taskmanager, myIngredientswoEV,save_interval, max_mcs, outfile, infile, sliding, DiagonalMovesON, ShearForceOn, ShearForce,  analyzeStrain, outfileStrain);

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

template <class taskman, class myIng>
void addtasks(taskman& mytaskman, myIng& ingredients, uint32_t interval, uint32_t max, std::string out , std::string in, bool slide, bool diagonal,bool ShearForceOn, double  ShearForce , bool analyzeStrain, std::string outfileStrain)
{
  auto const pUpdaterGpu = new GPUScBFMGPR_AB_Type<myIng>( ingredients, interval );
  pUpdaterGpu -> setSlideON(slide);
  pUpdaterGpu -> setDiagonalMovesON(diagonal);
  if (ShearForceOn) { pUpdaterGpu -> setForceON(ShearForce); }
  mytaskman.addUpdater( new UpdaterReadBfmFile<myIng>( in, ingredients,UpdaterReadBfmFile<myIng>::READ_LAST_CONFIG_SAVE ), 0 );
  mytaskman.addUpdater( pUpdaterGpu );
//   if ( analyzeStrain ) { mytaskman.addAnalyzer( new AnalyzerShearStrain<myIng>(ingredients, outfileStrain));}
  mytaskman.addAnalyzer( new AnalyzerWriteBfmFile<myIng>( out, ingredients ) );
//   mytaskman.addAnalyzer( new AnalyzerWriteBfmFileEachFrame<myIng>( out, ingredients ) );
  mytaskman.addAnalyzer( new AnalyzerWriteBfmFile<myIng>("LastConfig.bfm", ingredients,AnalyzerWriteBfmFile<myIng>::OVERWRITE), max / interval  );          
}