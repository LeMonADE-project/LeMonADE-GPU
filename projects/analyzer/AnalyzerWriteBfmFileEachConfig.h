/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo---0---oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADE.

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

#ifndef LEMONADE_ANALYZER_ANALYZERWRITEBFMFILEEACHCONFIG_H
#define LEMONADE_ANALYZER_ANALYZERWRITEBFMFILEEACHCONFIG_H

#include <set>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>

#include <LeMonADE/Version.h>
#include <LeMonADE/analyzer/AbstractAnalyzer.h>
#include <LeMonADE/io/AbstractWrite.h>
#include <LeMonADE/utility/ResultFormattingTools.h>
#include <LeMonADE/analyzer/AnalyzerWriteBfmFile.h>

/***********************************************************************/
/**
 * @file
 *
 * @class AnalyzerWriteBfmFileEachConfig
 *
 * @brief Analyzer writing the configurations into a given bfm-file.
 *
 * @details The output is appended to the file, if the file already exists.
 * If it does not exist, a new file is created and the header information is written
 * at the beginning
 *
 * @tparam IngredientsType Ingredients class storing all system information( e.g. monomers, bonds, etc).
 *
 * @todo rename to WriteBFMFile or similar.
 **/
template <class IngredientsType>
class AnalyzerWriteBfmFileEachConfig: public AnalyzerWriteBfmFile<IngredientsType>
{
public:
    typedef  AnalyzerWriteBfmFile<IngredientsType> BaseClass;
	//! Standard Constructor. Default: all data will be appended to the file
    AnalyzerWriteBfmFileEachConfig(const std::string& basename_,const IngredientsType& ing);
    //! Triggers writing of the Writes that need to be written for every step (i.e. excluding header)
    virtual bool execute();

private:
    //! The filename to be used.
    std::string basename;

    //! Storage for data that are processed to file (mostly Ingredients).
    const IngredientsType& ingredients;

};

/***********************************************************************/
//constructor
/***********************************************************************/
/**
 * @details It initialize all internal values and passes all information to the corresponding classes.
 * It register all known commands from the Feature. The class uses the AnalyzerWriteBfmFile as mother class
 * and sets the write_type to overwrite. Then only the filename needs to be changed every mcs.
 *
 *
 * @param basename base name of the file to write-out
 * @param ing Class holding all information of the system (mainly Ingredients )
 * 
 */
template <class IngredientsType>
AnalyzerWriteBfmFileEachConfig<IngredientsType>::AnalyzerWriteBfmFileEachConfig(const std::string& basename_, const IngredientsType& ing )
    :BaseClass(basename_,ing,2), ingredients(ing),basename(basename_){}
/***********************************************************************/
//bool execute
/***********************************************************************/
/**
 * @details Write-out of the next configuration of all Feature.
 *
 * @return True if everthing is alrigth. False if something goes wrong.
 */
template <class IngredientsType>
bool AnalyzerWriteBfmFileEachConfig<IngredientsType>::execute(){
    std::stringstream output;
    output <<basename<<"_MCS"<<ingredients.getMolecules().getAge() << ".bfm";
    BaseClass::setFilename(output.str());
    BaseClass::execute();
}
#endif
