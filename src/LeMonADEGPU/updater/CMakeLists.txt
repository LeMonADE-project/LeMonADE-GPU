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

SET(_src
    UpdaterGPUScBFM.cu
    UpdaterGPUScBFM_AB_Connection.cu
    UpdaterGPUScBFM_AA_Connection.cu
    UpdaterGPUScBFM_AB_Breaking.cu
    UpdaterGPUScBFM_AA_Breaking.cu
    UpdaterGPUScBFM_Tendomers.cu
    UpdaterGPUScBFM_TendomersConnection.cu
   )
FILE(_header
     ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/updater/UpdaterGPUScBFM.h
     ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Connection.h
     ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/updater/UpdaterGPUScBFM_AA_Connection.h
     ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Breaking.h
     ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/updater/UpdaterGPUScBFM_AA_Breaking.h
     ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/updater/UpdaterGPUScBFM_Tendomers.h
     ${CMAKE_SOURCE_DIR}/include/LeMonADEGPU/updater/UpdaterGPUScBFM_TendomersConnection.h
    )
# set_property(TARGET obj_updater PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(obj_updater PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
CUDA_ADD_LIBRARY(obj_updater OBJECT ${_src} ${_header} cuda )
