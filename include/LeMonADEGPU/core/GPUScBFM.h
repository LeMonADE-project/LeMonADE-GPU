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
#pragma once


#include <chrono>                           // std::chrono::high_resolution_clock
#include <climits>                          // CHAR_BIT
#include <limits>                           // numeric_limits
#include <iostream>

#include <LeMonADE/updater/AbstractUpdater.h>
#include <LeMonADE/utility/Vector3D.h>      // VectorInt3

#include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>
// #include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Connection.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>

#define USE_UINT8_POSITIONS


/**
 * Why is this abstraction layer being used, instead of just incorporating
 * the GPU updated into this class?
 * I think because it was tried to get a LeMonADE independent .cu file for
 * the kernels while we still need to inherit from AbstractUpdater
 */


template< class T_IngredientsType >
class GPUScBFM : public AbstractUpdater
{
public:
    typedef T_IngredientsType IngredientsType;
    typedef typename T_IngredientsType::molecules_type MoleculesType;

protected:
    IngredientsType & mIngredients;
    MoleculesType   & molecules   ;

private:
    /**
     * can't use uint8_t for boxes larger 256 on any side, so choose
     * automatically the correct type
     * ... this is some fine stuff. I almost would have wrapped all the
     * method bodies inside macros ... in order to copy paste them inside
     * an if-else-statement
     * But it makes sense, it inherits from all and then type casts it to
     * call the correct methods and members even though all classes we
     * inherit from basically shadow each other
     * @see https://stackoverflow.com/questions/3422106/how-do-i-select-a-member-variable-with-a-type-parameter
     */
    struct WrappedTemplatedUpdaters :
        UpdaterGPUScBFM< uint8_t  >,
        UpdaterGPUScBFM< uint16_t >,
        UpdaterGPUScBFM< int16_t  >,
        UpdaterGPUScBFM< int32_t  >
    {};
    WrappedTemplatedUpdaters mUpdatersGpu;

    int miGpuToUse;
    //! Number of Monte-Carlo Steps (mcs) to be executed (per GPU-call / Updater call)
    uint32_t mnSteps;
    SelectedLogger mLog;
    bool mCanUseUint8Positions;
    uint64_t mnStepsBetweenSortings;
    bool mSetStepsBetweenSortings;
    uint8_t mnSplitColors;
    bool mDiagMovesOn; 
    bool densityCheckerON;
protected:
    inline T_IngredientsType & getIngredients() { return mIngredients; }

public:
    /**
     * @brief Standard constructor: initialize the ingredients and specify the GPU.
     *
     * @param rIngredients  A reference to the IngredientsType - mainly the system
     * @param rnSteps       Number of mcs to be executed per GPU-call
     * @param riGpuToUse    ID of the GPU to use. Default: 0
     */
    inline GPUScBFM
    (
        T_IngredientsType & rIngredients,
        uint32_t            rnSteps     ,
	    bool                mDiagMovesOn_ = false,
        int                 riGpuToUse = 0
    )
    : mIngredients( rIngredients                   ),
      molecules   ( rIngredients.modifyMolecules() ),
      miGpuToUse  ( riGpuToUse                     ),
      mnSteps     ( rnSteps                        ),
      mLog        ( __FILENAME__                   ),
      mSetStepsBetweenSortings( false ),
      mDiagMovesOn(mDiagMovesOn_),
      mnSplitColors( 0 ),
      densityCheckerON(false)
    {
        mLog.deactivate( "Check"     );
        mLog.deactivate( "Error"     );
        mLog.deactivate( "Info"      );
        mLog.deactivate( "Stat"      );
        mLog.deactivate( "Warning"   );
    }

    inline void activateLogging( std::string const sLevel )
    {
        UpdaterGPUScBFM< uint8_t  > & updater1 = mUpdatersGpu;
        UpdaterGPUScBFM< uint16_t > & updater2 = mUpdatersGpu;
        UpdaterGPUScBFM< int32_t  > & updater3 = mUpdatersGpu;
        updater1.mLog.activate( sLevel );
        updater2.mLog.activate( sLevel );
        updater3.mLog.activate( sLevel );
        mLog.activate( sLevel );
    }

    void setDensityCheckerON( bool densityCheckerON_ ){densityCheckerON=densityCheckerON_;}
    inline void setGpu( int riGpuToUse ){ miGpuToUse = riGpuToUse; }
    inline void setStepsBetweenSortings( int rnStepsBetweenSortings )
    {
        mSetStepsBetweenSortings = true;
        mnStepsBetweenSortings = rnStepsBetweenSortings;
    }
    inline void setSplitColors( uint8_t rnSplitColors ){ mnSplitColors = rnSplitColors; }

    /**
     * Copies required data and parameters from mIngredients to mUpdaterGpu
     * and calls the mUpdaterGpu initializer
     * mIngredients can't just simply be given, because we want to compile
     * UpdaterGPUScBFM.cu by itself and explicit template instantitation
     * over T_IngredientsType is basically impossible
     */
    template< typename T_UCoordinateCuda >
    inline void initializeUpdater()
    {
        UpdaterGPUScBFM< T_UCoordinateCuda > & mUpdaterGpu = mUpdatersGpu;

        mUpdaterGpu.setSplitColors( mnSplitColors );
	    mUpdaterGpu.setAutoColoring(true);
        mLog( "Info" ) << "Size of mUpdater: " << sizeof( mUpdaterGpu ) << " Byte\n";
        mLog( "Info" ) << "Size of WrappedTemplatedUpdaters: " << sizeof( WrappedTemplatedUpdaters ) << " Byte\n";

        auto const tInit0 = std::chrono::high_resolution_clock::now();

        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] Forwarding relevant paramters to GPU updater\n";
        mUpdaterGpu.setGpu( miGpuToUse );
        if ( mSetStepsBetweenSortings )
            mUpdaterGpu.setStepsBetweenSortings( mnStepsBetweenSortings );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setPeriodicity\n";
        /* Forward needed parameters to the GPU updater */
        mUpdaterGpu.setAge( mIngredients.modifyMolecules().getAge() );
        mUpdaterGpu.setPeriodicity( mIngredients.isPeriodicX(),
                                    mIngredients.isPeriodicY(),
                                    mIngredients.isPeriodicZ() );

        /* copy monomer positions, attributes and connectivity of all monomers */
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setLatticeSize\n";
        mUpdaterGpu.setLatticeSize( mIngredients.getBoxX(),
                                    mIngredients.getBoxY(),
                                    mIngredients.getBoxZ() );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setNrOfAllMonomers\n";
        mUpdaterGpu.setNrOfAllMonomers( mIngredients.getMolecules().size() );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setMonomerCoordinates\n";
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
        {
            mUpdaterGpu.setMonomerCoordinates( i, molecules[i].getX(),
                                                  molecules[i].getY(),
                                                  molecules[i].getZ() );
        }
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setAttribute\n";
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
	        mUpdaterGpu.setAttributeTag( i, mIngredients.getMolecules()[i].getAttributeTag() );

        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setConnectivity\n";
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
        for ( size_t iBond = 0; iBond < mIngredients.getMolecules().getNumLinks(i); ++iBond )
            mUpdaterGpu.setConnectivity( i, mIngredients.getMolecules().getNeighborIdx( i, iBond ) );

         // false-allowed; true-forbidden
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] copy bondset from LeMonADE to GPU-class for BFM\n";
        /* maximum of (expected!!!) bond length in one dimension. Should be
         * queryable or there should be a better way to copy the bond set.
         * Note that supported range is [-4,3] */
        int const maxBondLength = 4;
        for ( int dx = -maxBondLength; dx < maxBondLength; ++dx )
        for ( int dy = -maxBondLength; dy < maxBondLength; ++dy )
        for ( int dz = -maxBondLength; dz < maxBondLength; ++dz )
        {
            /* !!! The negation is confusing, again there should be a better way to copy the bond set */
            mUpdaterGpu.copyBondSet( dx, dy, dz, ! mIngredients.getBondset().isValid( VectorInt3( dx, dy, dz ) ) );
        }
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] set move type (either standard or diagonal moves)\n";
        mUpdaterGpu.setDiagonalMovesOn(mDiagMovesOn);
        if ( mIngredients.isForceOn() )
            mUpdaterGpu.setShearForce(mIngredients.getAmplitudeShearForce());
        else 
            mUpdaterGpu.setShearForce(0);
        mUpdaterGpu.setDensityCheckerON(densityCheckerON);
        Method met;
        met.modifyCurve().setBox(mIngredients.getBoxX(),mIngredients.getBoxY(),mIngredients.getBoxZ());
	met.modifyCurve().setMode(0); // Z-Order-Curve
	//met.modifyCurve().setMode(1); // Linear Modulus
	//met.modifyCurve().setMode(2); // Linear Bit-Operation
        met.modifyPacking().setBitPackingOn(true);
        met.modifyPacking().setNBufferedTmpLatticeOn(true);
        met.setOnGPUForOverhead(true);
        mUpdaterGpu.setMethod(met);

	
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] initialize GPU updater\n";
        mUpdaterGpu.initialize();

        auto const tInit1 = std::chrono::high_resolution_clock::now();
        std::stringstream sBuffered;
        sBuffered << "tInit = " << std::chrono::duration<double>( tInit1 - tInit0 ).count() << "s\n";
    }

    /**
     * Was the 'virtual' really necessary ??? I don't think there will ever be
     * some class inheriting from this class...
     * https://en.wikipedia.org/wiki/Virtual_function
     */
    template< typename T_UCoordinateCuda >
    inline bool executeUpdater()
    {
        UpdaterGPUScBFM< T_UCoordinateCuda > & mUpdaterGpu = mUpdatersGpu;

        std::clock_t const t0 = std::clock();

        mLog( "Info" ) << "[" << __FILENAME__ << "] MCS:" << mIngredients.getMolecules().getAge() << "\n";
        mLog( "Info" ) << "[" << __FILENAME__ << "] start simulation on GPU\n";

        mUpdaterGpu.setAge( mIngredients.modifyMolecules().getAge() );
	    mUpdaterGpu.runSimulationOnGPU( mnSteps ); 

        // copy back positions of all monomers
        mLog( "Info" ) << "[" << __FILENAME__ << "] copy back monomers from GPU updater to CPU 'molecules' to be used with analyzers\n";
        for( size_t i = 0; i < mIngredients.getMolecules().size(); ++i )
        {
            molecules[i].setAllCoordinates
            (	
                mUpdaterGpu.getMonomerPositionInX(i),
                mUpdaterGpu.getMonomerPositionInY(i),
                mUpdaterGpu.getMonomerPositionInZ(i)
            );
        }
        /* update number of total simulation steps already done */
        mIngredients.modifyMolecules().setAge( mIngredients.modifyMolecules().getAge() + mnSteps );

        if ( mLog.isActive( "Stat" ) )
        {
            std::clock_t const t1 = std::clock();
            double const dt = (double) ( t1 - t0 ) / CLOCKS_PER_SEC;    // in seconds
            /* attempted moves per second */
            double const amps = ( (double) mnSteps * mIngredients.getMolecules().size() )/ dt;

            mLog( "Stat" )
            << "[" << __FILENAME__ << "] mcs " << mIngredients.getMolecules().getAge()
            << " with " << amps << " [attempted moves/s]\n"
            << "[" << __FILENAME__ << "] mcs " << mIngredients.getMolecules().getAge()
            << " passed time " << dt << " [s] with " << mnSteps << " MCS\n";
        }

        return true;
    }

    template< typename T_UCoordinateCuda >
    inline void cleanupUpdater()
    {
        UpdaterGPUScBFM< T_UCoordinateCuda > & mUpdaterGpu = mUpdatersGpu;

        mLog( "Info" ) << "[" << __FILENAME__ << "] cleanup\n";
        mUpdaterGpu.cleanup();
    }

#if defined( USE_UINT8_POSITIONS )
    inline void initialize()
    {
        auto const maxBoxSize = std::max( mIngredients.getBoxX(), std::max( mIngredients.getBoxY(), mIngredients.getBoxZ() ) );
	if ( maxBoxSize < 0 )
	  std::runtime_error("The maximum box size detected is smaller than 0! There could be something wront with the input file or the given bix sizes. ");
	
        mCanUseUint8Positions = (unsigned long long) maxBoxSize <= ( 1llu << ( CHAR_BIT * sizeof( uint8_t ) ) );
        if ( mCanUseUint8Positions )
            initializeUpdater< uint8_t >();
        else
            initializeUpdater< uint16_t >();
    }

    inline bool execute()
    {
        if ( mCanUseUint8Positions )
            return executeUpdater< uint8_t >();
        else
            return executeUpdater< uint16_t >();
    }

    inline void cleanup()
    {
        if ( mCanUseUint8Positions )
            cleanupUpdater< uint8_t >();
        else
            cleanupUpdater< uint16_t >();
    }
#else
    inline void initialize(){     initializeUpdater< int32_t >(); }
    inline bool execute   (){ return executeUpdater< int32_t >(); }
    inline void cleanup   (){        cleanupUpdater< int32_t >(); }
#endif
};
