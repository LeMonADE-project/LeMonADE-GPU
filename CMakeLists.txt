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
#
# Project Properties
#
CMAKE_MINIMUM_REQUIRED (VERSION 3.1 FATAL_ERROR ) # c11 standard required 
PROJECT (LeMonADEGPU )
SET (APPLICATION_NAME "LeMonADEGPU")
SET (APPLICATION_CODENAME "${PROJECT_NAME}")
SET (APPLICATION_COPYRIGHT_YEARS "2019")
SET (APPLICATION_VERSION_MAJOR 1)
SET (APPLICATION_VERSION_MINOR 0)
SET (APPLICATION_VERSION_PATCH 0)
SET (APPLICATION_VERSION_TYPE SNAPSHOT)
SET (APPLICATION_VERSION_STRING "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}.${APPLICATION_VERSION_PATCH}-${APPLICATION_VERSION_TYPE}")
SET (APPLICATION_ID "${APPLICATION_VENDOR_ID}.${PROJECT_NAME}")

set( CMAKE_CXX_STANDARD 11 ) # needs CMake 3.1. Need C++11 for auto-keyword and list initializers ... I sure have gotten lazy :S
if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.0" )
        message( FATAL_ERROR "Insufficient gcc version. Need at least 4.7.0 for -std=c++11 flag for 'using T=int;' and at least 4.8.0 for more generalized initializer lists! See also https://gcc.gnu.org/onlinedocs/gcc-4.6.4/gcc/C-Dialect-Options.html#C-Dialect-Options vs. https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/C-Dialect-Options.html#C-Dialect-Options" )
    endif()
endif()

#LeMonADE main repo: lib and include path 
SET (LEMONADE_INCLUDE_DIR "${LEMONADE_DIR}/include")
SET (LEMONADE_LIBRARY_DIR "${LEMONADE_DIR}/lib")

if (NOT DEFINED LEMONADE_INCLUDE_DIR)
message("LEMONADE_INCLUDE_DIR is not provided. If build fails, use -DLEMONADE_INCLUDE_DIR=/path/to/LeMonADE/headers/ or install to default location")
endif()
MESSAGE("LEMONADE_INCLUDE_DIR is ${LEMONADE_INCLUDE_DIR}")

if (NOT DEFINED LEMONADE_LIBRARY_DIR)
message("LEMONADE_LIBRARY_DIR is not provided. If build fails, use -DLEMONADE_LIBRARY_DIR=/path/to/LeMonADE/lib/ or install to default location")
endif()
MESSAGE("LEMONADE_LIBRARY_DIR is ${LEMONADE_LIBRARY_DIR}")

#define possible flags
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse2 -mssse3 -fexpensive-optimizations -std=c++11 ")
SET (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -msse2 -mssse3 -fexpensive-optimizations -std=c++11 ")

SET (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -DDEBUG -std=c++11 ")
SET (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -Wall -DDEBUG -std=c++11 ")
# SET (CMAKE_CXX_FLAGS_PROFIL "${CMAKE_CXX_FLAGS_PROFIL} -O3 -pg -msse2 -mssse3 -fexpensive-optimizations ")
# SET (CMAKE_C_FLAGS_PROFIL "${CMAKE_C_FLAGS_PROFIL} -O3 -pg -msse2 -mssse3 -fexpensive-optimizations ")

#define value of CMAKE_BUILD_TYPE depending on input
IF(NOT CMAKE_BUILD_TYPE)
SET (CMAKE_BUILD_TYPE "Release") #default build type is Release
ELSEIF(CMAKE_BUILD_TYPE STREQUAL "Release")
SET (CMAKE_BUILD_TYPE "Release") 
ELSEIF(CMAKE_BUILD_TYPE STREQUAL "Debug")
SET (CMAKE_BUILD_TYPE "Debug") 
ELSE(NOT CMAKE_BUILD_TYPE)
MESSAGE(FATAL_ERROR "Invalid build type ${CMAKE_BUILD_TYPE} specified.")
ENDIF(NOT CMAKE_BUILD_TYPE)

#output depending on build type
IF(CMAKE_BUILD_TYPE STREQUAL "Release")
SET (CMAKE_VERBOSE_MAKEFILE 0)
MESSAGE("Build type is ${CMAKE_BUILD_TYPE}")
MESSAGE("USING CXX COMPILER FLAGS ${CMAKE_CXX_FLAGS_RELEASE}")
MESSAGE("USING C COMPILER FLAGS ${CMAKE_C_FLAGS_RELEASE}")
ELSEIF(CMAKE_BUILD_TYPE STREQUAL "Debug")
SET (CMAKE_VERBOSE_MAKEFILE 1)
MESSAGE("Build type is ${CMAKE_BUILD_TYPE}")
MESSAGE("USING CXX COMPILER FLAGS ${CMAKE_CXX_FLAGS_DEBUG}")
MESSAGE("USING C COMPILER FLAGS ${CMAKE_C_FLAGS_DEBUG}")
ENDIF(CMAKE_BUILD_TYPE STREQUAL "Release")


#include cuda and set flags 
find_package( CUDA REQUIRED )
if( ${CUDA_VERSION_MAJOR} GREATER 7 )
    set( NVCC_EXPERIMENTAL_FLAG "--expt" )
else()
    set( NVCC_EXPERIMENTAL_FLAG "-" )
endif()
# set( CUDA_ARCH "60")
set( MINIMAL_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
     -Xcompiler -fPIC,-Wno-unused-parameter
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}
    ${NVCC_EXPERIMENTAL_FLAG}-relaxed-constexpr
)
set( CUDA_NVCC_FLAGS ${MINIMAL_CUDA_NVCC_FLAGS} " --compiler-options -Wall,-Wextra" )

###############################################################################
# Project Output Paths
#
SET (LEMONADEGPU_DIR ${PROJECT_SOURCE_DIR})
SET (EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
SET (LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
SET (LEMONADEGPU_INCLUDE_DIR "${LEMONADEGPU_DIR}/include")
SET (LEMONADEGPU_LIBRARY_DIR ${LIBRARY_OUTPUT_PATH})
# SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${LEMONADEGPU_DIR}/bin")


#include directory
LIST (APPEND CMAKE_PREFIX_PATH "${LEMONADEGPU_DIR}")
LIST (APPEND CMAKE_PREFIX_PATH "${LEMONADEGPU_DIR}")
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include/ ${LEMONADE_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} )
# LINK_DIRECTORIES (${PROJECT_SOURCE_DIR}/src)


## ###############  System Creators ############# ##

add_subdirectory(src)
add_subdirectory(projects)

#
# Add option for building tests
#
option(LEMONADE_TESTS "Build the test" OFF)
if(LEMONADE_TESTS)
    add_subdirectory(tests)
endif(LEMONADE_TESTS)



#
# Add Install Targets
#

# Check if INSTALLDIR_LEMONADEGPU is given
if (DEFINED INSTALLDIR_LEMONADEGPU)
    message("INSTALLDIR_LEMONADEGPU set to " ${INSTALLDIR_LEMONADEGPU})
    SET(CMAKE_INSTALL_PREFIX "${INSTALLDIR_LEMONADEGPU}")
else (DEFINED INSTALLDIR_LEMONADEGPU)
	message("INSTALLDIR_LEMONADEGPU set to default" ${CMAKE_INSTALL_PREFIX})
endif()

# 
# INSTALL(DIRECTORY "${LEMONADEGPU_DIR}/include/" DESTINATION  "include")
# 
# #
# # Add Documentation Targets
# #
# SET (DOC_INPUT_FILE_PATH "${LEMONADEGPU_DIR}/docs/")
# SET (DOC_OUTPUT_FILE_PATH "${CMAKE_BINARY_DIR}/docs/")
# 
# FIND_PACKAGE (Doxygen)
# IF (DOXYGEN_FOUND)
#     MESSAGE("Build documentation with: make docs")
#     IF (EXISTS ${DOC_INPUT_FILE_PATH})
#         MESSAGE("Existing File documentation with doxygen")
#         configure_file(${DOC_INPUT_FILE_PATH}doxygen.conf ${DOC_OUTPUT_FILE_PATH}doxygen.conf @ONLY)
#         configure_file(${DOC_INPUT_FILE_PATH}mainpage.dox ${DOC_OUTPUT_FILE_PATH}mainpage.dox @ONLY)
#         configure_file(${DOC_INPUT_FILE_PATH}figures/ProgramStructure.jpg ${DOC_OUTPUT_FILE_PATH}figures/ProgramStructure.jpg COPYONLY)
#         ADD_CUSTOM_TARGET(
#             docs
#             ${DOXYGEN_EXECUTABLE} ${DOC_OUTPUT_FILE_PATH}doxygen.conf
#             WORKING_DIRECTORY ${DOC_OUTPUT_FILE_PATH}
#             COMMENT "Generating doxygen project documentation." VERBATIM
#         )
#     ELSE (EXISTS ${DOC_INPUT_FILE_PATH})
#         ADD_CUSTOM_TARGET(docs COMMENT "Doxyfile not found. Please generate a doxygen configuration file to use this target." VERBATIM)
#     ENDIF (EXISTS ${DOC_INPUT_FILE_PATH})
# ELSE (DOXYGEN_FOUND)
#     ADD_CUSTOM_TARGET(docs COMMENT "Doxygen not found. Please install doxygen to use this target." VERBATIM)
# ENDIF (DOXYGEN_FOUND)

