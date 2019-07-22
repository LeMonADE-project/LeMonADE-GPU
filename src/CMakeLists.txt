# ---------------------------------------------------------------------------------
#     ooo      L   attice-based  |
#   o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
#  o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
# oo---0---oo  A   lgorithm and  |
#  o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
#   o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
#     ooo                        |
# ---------------------------------------------------------------------------------
#
# This file is part of LeMonADE.
#
# LeMonADE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# LeMonADE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with LeMonADE.  If not, see <http://www.gnu.org/licenses/>.
#
# --------------------------------------------------------------------------------

SET ( project_SRCS
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/core/SpaceFillingCurve.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/core/mainColorBFM.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/core/BondVectorSet.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/core/kernelConnection.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.cu      
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/updater/UpdaterGPUScBFM_Connection.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/updater/UpdaterGPUScBFM_AA_Connection.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/updater/UpdaterGPUScBFM_Breaking.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/updater/UpdaterGPUScBFM_AA_Breaking.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/utility/AutomaticThreadChooser.cu
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/utility/findBestZCurveFrontOrder.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/utility/graphColoring.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/utility/SelectiveLogger.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/utility/findMaxNeighbors.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/LeMonADEGPU/utility/GPUConnectionTracker.cu
)  
     
MESSAGE( "Add src files : " ${project_SRCS})
SET (project_BIN ${PROJECT_NAME})

IF (NOT DEFINED project_build_static)
   SET(project_build_static 1)
ENDIF (NOT DEFINED project_build_static)

IF (project_build_static)
	CUDA_ADD_LIBRARY(libLeMonADEGPU-static STATIC ${project_SRCS} )
# 	set_property(TARGET libLeMonADEGPU-static PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	set_target_properties(libLeMonADEGPU-static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	#TARGET_LINK_LIBRARIES(staticlib ${project_LIBS})
	SET_TARGET_PROPERTIES(libLeMonADEGPU-static PROPERTIES VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}" OUTPUT_NAME ${project_BIN} CLEAN_DIRECT_OUTPUT 1)
	INSTALL(TARGETS libLeMonADEGPU-static DESTINATION lib)
ENDIF(project_build_static)

