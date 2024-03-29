## ----------------------------------------------------------------------------------
##     ooo      L   attice-based  |
##   o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
##  o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
## oo---0---oo  A   lgorithm and  |
##  o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
##   o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
##     ooo                        |
## ----------------------------------------------------------------------------------
##
## This file is part of LeMonADE.
##
## LeMonADE is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## LeMonADE is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with LeMonADE.  If not, see <http://www.gnu.org/licenses/>.
##
## ----------------------------------------------------------------------------------

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

# minimalistic gpuinfo tool printing all properties
# set_source_files_properties( src/gpuinfo.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
# set( CUDA_NVCC_FLAGS ${MINIMAL_CUDA_NVCC_FLAGS} )
# cuda_add_executable( gpuinfo gpuinfo.cpp )
# target_link_libraries( gpuinfo cuda )

SET(_src
  graphColoring.cpp
  findBestZCurveFrontOrder.cpp
  findMaxNeighbors.cpp
  SelectiveLogger.cpp
  AutomaticThreadChooser.cu
  GPUConnectionTracker.cu
  GPUConnectionTrackerTendomer.cu
  )
FILE(_header
    ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/utility/graphColoring.h
    ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/utility/AutomaticThreadChooser.h
    ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/utility/GPUConnectionTracker.h
    ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/utility/GPUConnectionTrackerTendomer.h
    ${CMAKE_SOURCE_DIR}/include/extern/Fundamental/Fundamental.hpp
    ${CMAKE_SOURCE_DIR}/include/extern/Fundamental/toString.hpp
    ${CMAKE_SOURCE_DIR}/include/extern/Fundamental/vectorIndex.hpp
    )
set_property(TARGET obj_utility PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
CUDA_ADD_LIBRARY(obj_utility OBJECT ${_src} ${_header} cuda )

