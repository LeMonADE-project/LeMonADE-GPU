#pragma once


#include <chrono>                           // std::chrono::high_resolution_clock
#include <iostream>

#include <LeMonADE/updater/AbstractUpdater.h>
#include <LeMonADE/utility/Vector3D.h>      // VectorInt3


#include <LeMonADEGPU/updater/UpdaterGPUScBFMGPR_AB_Type.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>



/**
 * Why is this abstraction layer being used, instead of just incorporating
 * the GPU updated into this class?
 * I think because it was tried to get a LeMonADE independent .cu file for
 * the kernels while we still need to inherit from AbstractUpdater
 */


template< class T_IngredientsType >
class GPUScBFMGPR_AB_Type : public AbstractUpdater
{
public:
    typedef T_IngredientsType IngredientsType;
    typedef typename T_IngredientsType::molecules_type MoleculesType;

protected:
    IngredientsType & mIngredients;
    MoleculesType   & molecules   ;

private:
    UpdaterGPUScBFMGPR_AB_Type mUpdaterGpu;
    int miGpuToUse;
    //! Number of Monte-Carlo Steps (mcs) to be executed (per GPU-call / Updater call)
    uint32_t mnSteps;
    SelectedLogger mLog;
    bool SlideOn;
    bool DiagonalMovesOn;
    double ForceStrength;
    bool UseGivenForce;
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
    inline GPUScBFMGPR_AB_Type
    (
        T_IngredientsType & rIngredients,
        uint32_t            rnSteps     ,
        int                 riGpuToUse = 0
    )
    : mIngredients   ( rIngredients                   ),
      molecules      ( rIngredients.modifyMolecules() ),
      miGpuToUse     ( riGpuToUse                     ),
      mnSteps        ( rnSteps                        ),
      mLog           ( __FILENAME__                   ),
      SlideOn        ( false                          ),
      DiagonalMovesOn( false                          ),
      UseGivenForce  ( false                          ),
      ForceStrength  ( 0.0                            )
    {
        mLog.  activate( "Benchmark" );
        mLog.deactivate( "Check"     );
        mLog.  activate( "Error"     );
        mLog.  activate( "Info"      );
        mLog.  activate( "Stat"      );
        mLog.deactivate( "Warning"   );
    }

    inline void setSlideON( bool SlideOn_ ){ SlideOn=SlideOn_; }
    inline void setDiagonalMovesON( bool DiagonalMovesOn_){DiagonalMovesOn=DiagonalMovesOn_;}
    inline void setForceON( double ForceStrength_  ){ ForceStrength=ForceStrength_; UseGivenForce=true; }
    /**
     * Copies required data and parameters from mIngredients to mUpdaterGpu
     * and calls the mUpdaterGpu initializer
     * mIngredients can't just simply be given, because we want to compile
     * UpdaterGPUScBFMGPR_AB_Type.cu by itself and explicit template instantitation
     * over T_IngredientsType is basically impossible
     */
    inline void initialize()
    {
	auto const tInit0 = std::chrono::high_resolution_clock::now();
	//use the metadata of the features for the parsing of the used features...
	std::stringstream metadata;
	mIngredients.printMetaData(metadata);
	if ( metadata.str().find("FeatureExcludedVolume") !=std::string::npos )
	{ 
	      mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setEVON = " << true << " \n";
	      mUpdaterGpu.setEVON( true ); 
	}
	else
	{
	      mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setEVON = " << false << " \n";
	      mUpdaterGpu.setEVON( false ); 
	}
	
	// use the force given by the user or from the file if not states otherwise
// 	if( UseGivenForce )
// 	{
// 	    if (ForceStrength > 0.0){
// 	      mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setForce = " << ForceStrength << " \n";
// 	      mUpdaterGpu.setForce(ForceStrength);
// 	      
// 	      mIngredients.setForceOn(true);
// 	      mIngredients.setAmplitudeShearForce(ForceStrength);
// 	    } 
// 	    else 
// 	    {
// 	      mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setForce on  = " << false << " \n";
// 	      mUpdaterGpu.setForce(0.0);
// 	      mIngredients.setForceOn(false);
// 	    }
// 	}
// 	else 
// 	{
// 	    if( mIngredients.isForceOn())
// 	    {
// 	      mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setForce = " << mIngredients.getAmplitudeShearForce() << " \n";
// 	      mUpdaterGpu.setForce(mIngredients.getAmplitudeShearForce());
// 	      
// 	    }
// 	    else
// 	    { 
// 	      mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setForce = " << 0 << " \n";
// 	      mUpdaterGpu.setForce(0);
// 	    }
// 	}
	
	mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setDiagonalMovesON = " << DiagonalMovesOn << " \n";
	mUpdaterGpu.setDiagonalMovesON(DiagonalMovesOn);
	mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] Forwarding relevant paramters to GPU updater\n";
        mUpdaterGpu.setGpu( miGpuToUse );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setPeriodicity\n";
        /* Forward needed parameters to the GPU updater */
        mUpdaterGpu.setPeriodicity( mIngredients.isPeriodicX(),
                                    mIngredients.isPeriodicY(),
                                    mIngredients.isPeriodicZ() );

        /* copy monomer positions, attributes and connectivity of all monomers */
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
            mUpdaterGpu.setAttribute( i, mIngredients.getMolecules()[i].getAttributeTag() );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setConnectivity\n";
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
        for ( size_t iBond = 0u; iBond < mIngredients.getMolecules().getNumLinks(i); ++iBond )
            mUpdaterGpu.setConnectivity( i, mIngredients.getMolecules().getNeighborIdx( i, iBond ) );
	mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setRingSliding = " << SlideOn << " \n";
	mUpdaterGpu.setRingSliding(SlideOn);
	if(SlideOn){
	    mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setNrOfRingMonomers\n";
// 	    mUpdaterGpu.setNrOfRingMonomers( mIngredients.getNumberofCrossLinkers()*mIngredients.getNumberofChains() );
	    mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setNrOfChains\n";
// 	    mUpdaterGpu.setNrOfChains( mIngredients.getNumberofChains() );
	    mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setNrOfMonomersPerChain\n";
// 	    mUpdaterGpu.setNrOfMonomersPerChain( mIngredients.getChainLength() );
	    mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setRingConnectivity\n";
// 	    uint32_t chainlength(mIngredients.getChainLength());
// 	    uint32_t NMonomersPerChain(mIngredients.getNumberofChains());
// 	    uint32_t offset(chainlength*NMonomersPerChain);
	    
// 	    for ( size_t i = 0u; i < mIngredients.getNumberofChains(); ++i )
// 	    {
// 	      //get the sequence of the rings along a single chain
// 		std::vector<uint32_t> RingSequence;
// 		for ( size_t j = 0u; j < chainlength; ++j )
// 		{
// 		    uint32_t MonID(j+i*chainlength);
// 		    for ( size_t iBond = 0u; iBond < mIngredients.getMolecules().getNumLinks(MonID); ++iBond )
// 		    {
// 		      uint32_t Neighbor(mIngredients.getMolecules().getNeighborIdx( MonID, iBond ));
// 		      if (mIngredients.getMolecules()[Neighbor].getAttributeTag() >2)
// 			  RingSequence.push_back(Neighbor);
// 		    }
// 		}
// 		// make the virtual connection between neighboring rings
// 		for(uint32_t k = 0; k < RingSequence.size()-1; k++)
// 		{ 	
// 		    mUpdaterGpu.setRingConnectivity(RingSequence[k  ]-offset, RingSequence[k+1]-offset);
// 		    mUpdaterGpu.setRingConnectivity(RingSequence[k+1]-offset, RingSequence[k  ]-offset);
// 		}
// 	    }
// 	    for ( size_t i = offset; i < mIngredients.getMolecules().size(); ++i )
// 	      if ( mIngredients.getMolecules()[i].getAttributeTag() > 2 )
// 		for ( size_t iBond = 0u; iBond < mIngredients.getMolecules().getNumLinks(i); ++iBond )
// 		  if ( mIngredients.getMolecules()[mIngredients.getMolecules().getNeighborIdx( i, iBond )].getAttributeTag() > 2 ) 
// 		    mUpdaterGpu.setRingConnectivity( i-offset, mIngredients.getMolecules().getNeighborIdx( i, iBond )-offset );
	}
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setLatticeSize\n";
        mUpdaterGpu.setLatticeSize( mIngredients.getBoxX(),
                                    mIngredients.getBoxY(),
                                    mIngredients.getBoxZ() );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.populateLattice\n";
        mUpdaterGpu.populateLattice(); /* needs data set by setMonomerCoordinates. Is this actually needed ??? */

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

        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] initialize GPU updater\n";
        mUpdaterGpu.initialize();

        auto const tInit1 = std::chrono::high_resolution_clock::now();
        std::stringstream sBuffered;
        sBuffered << "tInit = " << std::chrono::duration<double>( tInit1 - tInit0 ).count() << "s\n";
        mLog( "Benchmark" ) << sBuffered.str();
    }

    /**
     * Was the 'virtual' really necessary ??? I don't think there will ever be
     * some class inheriting from this class...
     * https://en.wikipedia.org/wiki/Virtual_function
     */
    inline bool execute()
    {
        std::clock_t const t0 = std::clock();

        mLog( "Info" ) << "[" << __FILENAME__ << "] MCS:" << mIngredients.getMolecules().getAge() << "\n";
        mLog( "Info" ) << "[" << __FILENAME__ << "] start simulation on GPU\n";

        mUpdaterGpu.runSimulationOnGPU( mnSteps ); // sets mtCopyBack0
        auto const tCopyBack0 = mUpdaterGpu.mtCopyBack0;

        // copy back positions of all monomers
	MoleculesType NewMolecules;
        mLog( "Info" ) << "[" << __FILENAME__ << "] copy back monomers from GPU updater to CPU 'molecules' to be used with analyzers\n";
	for( size_t i = 0; i < mIngredients.getMolecules().size(); ++i )
        {
	  NewMolecules.addMonomer(
                mUpdaterGpu.getMonomerPositionInX(i),
                mUpdaterGpu.getMonomerPositionInY(i),
                mUpdaterGpu.getMonomerPositionInZ(i)
            );
        }
        // copy back the bonds of all monomers 
        mLog( "Info" ) << "[" << __FILENAME__ << "] copy back monomer connections from GPU updater to CPU 'molecules' to be used with analyzers\n";
	for( uint32_t i = 0; i < mIngredients.getMolecules().size(); ++i )
	    for (size_t j = 0; j < mUpdaterGpu.getMaximumConnectivity(i); ++j)
	    {
		 if ( mLog.isActive( "Check" ) ) 
		 {
		   std::stringstream sBuffered;
		   sBuffered<< i << " and " << mUpdaterGpu.getConnectivity(i,j) << " connection ID " << j << " " << i << " " << mIngredients.getMolecules().getNeighborIdx(i,j)<<" \n";
		    mLog( "Check" ) << sBuffered.str();
		 }
		 const uint32_t neighbor (mUpdaterGpu.getConnectivity(i,j));
		 if ( i != neighbor && ! NewMolecules.areConnected(i, neighbor) )
		   NewMolecules.connect(i, neighbor,0);
	    }
	for ( size_t i = 0; i < mIngredients.getMolecules().size(); ++i )
	  NewMolecules[i].setAttributeTag(mIngredients.getMolecules()[i].getAttributeTag());
        /* update number of total simulation steps already done */
	NewMolecules.setAge( mIngredients.modifyMolecules().getAge()+ mnSteps );
	mIngredients.modifyMolecules()=NewMolecules;
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

        if ( mLog.isActive( "Benchmark" ) )
        {
            auto const tCopyBack1 = std::chrono::high_resolution_clock::now();
            std::stringstream sBuffered;
            sBuffered << "tCopyback = " << std::chrono::duration<double>( tCopyBack1 - tCopyBack0 ).count() << "s\n";
            mLog( "Benchmark" ) << sBuffered.str();
        }
        return true;
    }

    inline void cleanup()
    {
        mLog( "Info" ) << "[" << __FILENAME__ << "] cleanup\n";
        mUpdaterGpu.cleanup();
    }
};
