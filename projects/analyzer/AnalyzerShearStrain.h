#ifndef ANALYZER_SHEAR_RATE_H
#define ANALYZER_SHEAR_RATE_H

#include <string>
#include <ostream>
#include <vector>
#include <algorithm>
#include <gsl/gsl_fit.h>
#include <LeMonADE/core/Ingredients.h>
#include "LeMonADE/utility/DepthIterator.h"
/*****************************************************************************
 *Calculates the average bond orientaion 
 *****************************************************************************/

/**
 * @file AnalyzerShearStrain.h
 *
 * @class AnalyzerShearStrain
 *
 * @brief Calculates the shear strain
 *
 * @tparam IngredientsType Ingredients class storing all system information( e.g. monomers, bonds, etc).
 * 
 * @todo 
 *
 */

template<class IngredientsType> class AnalyzerShearStrain:public AbstractAnalyzer
{
	
	
public:
	// !Constructor
	AnalyzerShearStrain(const IngredientsType& ing, std::string filename)
	:ingredients(ing)
	,outputFilename(filename)
	{}
	
	virtual ~AnalyzerShearStrain(){}
	
	//AbstractAnalyzer methods
	void initialize();
	bool execute();
	void cleanup();
	
	
private:
	typedef std::map<int32_t, std::map<int32_t,uint32_t> >::iterator It_outerMap;
	typedef std::map<int32_t,uint32_t>::iterator It_MapInt;
	typedef std::map<int32_t,double>::iterator It_MapDouble;
	
	const IngredientsType& ingredients;
	std::string outputFilename;
	VectorDouble3 UpperBondVector;
	uint32_t UpperCounter;
	VectorDouble3 LowerBondVector;
	uint32_t LowerCounter;
	double dist(VectorInt3 a){
	  return sqrt(a.getX()*a.getX()+a.getY()*a.getY()+a.getZ()*a.getZ());
	}
	uint32_t BoxX;
	uint32_t BoxY;
	uint32_t BoxZ;
	uint32_t counter;
	std::vector<std::map<int32_t,double> > ShearShiftTime;
	std::map<int32_t,double> Equilibrium_ShearShift;

};
/*****************************************************************************
 *
 *****************************************************************************/
template<class IngredientsType>
void AnalyzerShearStrain<IngredientsType>::initialize(){
	BoxX=ingredients.getBoxX();
	BoxY=ingredients.getBoxY();
	BoxZ=ingredients.getBoxZ();
	
	//! key1: z position, key2: x position, value: #monomers
	std::map<int32_t,std::map<int32_t,uint32_t> > comZ;
	for(uint32_t n=0; n<ingredients.getMolecules().size();n++){
	      VectorInt3 position(ingredients.getMolecules()[n]);
	      comZ[position.getZ()%BoxZ][position.getX()]++;
	}
	for(It_outerMap it=comZ.begin(); it!=comZ.end(); ++it){
	      Equilibrium_ShearShift[it->first]=BoxX/2.;
	}
}
/*****************************************************************************
 *gamma=tan(theta-pi/2)
 *     =bx/bz
 *****************************************************************************/
template<class IngredientsType>
bool AnalyzerShearStrain<IngredientsType>::execute(){
  
        //! key1: z position, key2: x position, value: #monomers
	std::map<int32_t,std::map<int32_t,uint32_t> > comZ;
	
	for(uint32_t n=0; n<ingredients.getMolecules().size();n++){
	      VectorInt3 position(ingredients.getMolecules()[n]);
	      comZ[position.getZ()%BoxZ][position.getX()]++;
	}
        //!key is z position and value is the displacement due to the deformation
        std::map<int32_t,double> ShearShift;
	double eqCOM((double)BoxX/2.);
	uint32_t SizeShearShift(0);
	for(It_outerMap it=comZ.begin(); it!=comZ.end(); ++it){
	      double sum(0);
	      double count(0);
	      for(It_MapInt it2=it->second.begin();it2!=it->second.end();++it2){
		   sum+=(double)(it2->second)*(double)(it2->first);
		   count+=it2->second;
	      }
	      ShearShift[it->first]=sum/count-Equilibrium_ShearShift[it->first];
	      SizeShearShift++;
	}
	
	uint32_t SizeFit((double)BoxZ/2.);
	double xUpper[SizeFit-8];
	double yUpper[SizeFit-8];
	double xLower[SizeFit-8];
	double yLower[SizeFit-8];
	ShearShiftTime.push_back(ShearShift);
	std::ofstream output2("ShearShift.dat",std::ios_base::app);
	
        uint32_t n1(0);
	uint32_t n2(0);
	for(It_MapDouble it=ShearShift.begin(); it!=ShearShift.end() ;++it){
	      output2<<it->second<<" ";
	      if(it->first<SizeFit-4 && it->first>3){
		    xLower[n1]=it->first;
		    yLower[n1]=it->second;
		    n1++;
	      }
	      if(it->first>SizeFit+3 && it->first<BoxZ-4)
	      {
		    xUpper[n2]=it->first;
		    yUpper[n2]=it->second;
		    n2++;
	      }
	      
	}
	output2<<std::endl;
	output2.close();
	
	//make a linear fit to the get shear amplitude 
	double c0Lower, c1Lower, cov00Lower, cov01Lower, cov11Lower, chisqLower;
	gsl_fit_linear (xLower, 1, yLower, 1, n1, 
			&c0Lower, &c1Lower, &cov00Lower, &cov01Lower, &cov11Lower, 
			&chisqLower);
	double c0Upper, c1Upper, cov00Upper, cov01Upper, cov11Upper, chisqUpper;
	gsl_fit_linear (xUpper, 1, yUpper, 1, n2, 
			&c0Upper, &c1Upper, &cov00Upper, &cov01Upper, &cov11Upper, 
			&chisqUpper);
	double UpperGamma(c1Upper);
	double LowerGamma(c1Lower);
	double AverageGamma(((c1Upper)-(c1Lower))/2.);
	std::ofstream output(outputFilename.c_str(),std::ios_base::app);
	output<<ingredients.getMolecules().getAge()<<" "<<UpperGamma<<" "<<LowerGamma<<" "<<AverageGamma<<std::endl;
	output.close();
	
	return true;
  
}
/*****************************************************************************
 *CleanUp(): first column is Z-position, other columns are the spacial deviation
 * from equilibrium com to the sheared com 
 *****************************************************************************/
template<class IngredientsType>
void AnalyzerShearStrain<IngredientsType>::cleanup(){
        std::ofstream output3("ShearShiftTime.dat");

	for(uint32_t n=0;n<BoxZ;n++){
		output3<<n<<" ";
		for(uint32_t m=0; m<ShearShiftTime.size();m++){
			output3<<ShearShiftTime[m][n]<<" ";
		}
		output3<<std::endl;
	  }
	  output3.close();
}

#endif
