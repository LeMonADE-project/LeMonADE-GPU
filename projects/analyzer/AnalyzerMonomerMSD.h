/****************************************************************************** 
 * based on LeMonADE: https://github.com/LeMonADE-project/LeMonADE/
 * author: Toni Müller
 * email: mueller-toni@ipfdd.de
 * project: topological effects 
 *****************************************************************************/

#ifndef LEMONADE_ANALYZER_MONOMER_MSD_H
#define LEMONADE_ANALYZER_MONOMER_MSD_H

#include <string>
#include "./AnalyzerAbstractMSD.h"
#include <LeMonADE/utility/MonomerGroup.h>

/*************************************************************************
 * definition of AnalyzerMonomerMSD class
 * ***********************************************************************/

/**
 * @file
 *
 * @class AnalyzerMonomerMSD
 *
 * @brief Analyzer for evaluating ...
 *
 * @tparam IngredientsType Ingredients class storing all system information( e.g. monomers, bonds, etc).
 *
 * @details 
 */
template < class IngredientsType > class AnalyzerMonomerMSD : public AnalyzerAbstractMSD<IngredientsType>
{

	//! typedef of parent class
	typedef AnalyzerAbstractMSD<IngredientsType> BaseClass;
  
private:
	//! use 
	using BaseClass::ingredients;
	//! Position is calculated for the groups in this vector
	std::vector<MonomerGroup<typename IngredientsType::molecules_type> > groups;

protected:
public:

	//! constructor
	AnalyzerMonomerMSD(const IngredientsType& ingredients_, uint32_t equilibrationTime_=0);

	//! destructor. does nothing
	virtual ~AnalyzerMonomerMSD(){}
	//! Initializes data structures. Called by TaskManager::initialize()
	virtual void initialize();
// 	//! Calculates the Rg2 for the current timestep. Called by TaskManager::execute()
// 	virtual bool execute();
// 	//! Writes the final results to file
// 	virtual void cleanup();
};

/*************************************************************************
 * implementation of memebers
 * ***********************************************************************/

/**
 * @param ingredients_ reference to the object holding all information of the system
 * @param equilibrationTime_ time after starting analyzing the system
 * */
template<class IngredientsType>
AnalyzerMonomerMSD<IngredientsType>::AnalyzerMonomerMSD(
	const IngredientsType& ingredients_, uint32_t equilibrationTime_)
:BaseClass(ingredients_, equilibrationTime_)
{
}


/**
 * @details 
 * */
template< class IngredientsType >
void AnalyzerMonomerMSD<IngredientsType>::initialize()
{
  std::cout << "AnalyzerMonomerMSD::initialize "<<std::endl;  
  //if no groups are set, use the complete system by default
  //groups can be set using the provided access function
  if(groups.size()==0){ 
    for(uint32_t i=0; i<ingredients.getMolecules().size();i++)
    {
	  groups.push_back(MonomerGroup<typename IngredientsType::molecules_type>(ingredients.getMolecules()));
	  groups[i].push_back(i);
    }
  }
  BaseClass::setOutputFilename("MonomerMSD.dat");
  BaseClass::setMonomerGroups(groups);
  BaseClass::initialize();
  std::cout << "done."<<std::endl;
}

#endif /*LEMONADE_ANALYZER_MONOMER_MSD_H*/


