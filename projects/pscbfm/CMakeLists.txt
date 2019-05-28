cmake_minimum_required(VERSION 3.1)

# cuda_add_library( pscbfmUpdater
#     src/pscbfm/UpdaterGPUScBFM_AB_Type.cu
# )
# target_link_libraries( pscbfmUpdater ${CUDA_curand_LIBRARY} cuda )
# 
# add_library( mainCUDASimulator mainCUDASimulatorGPUScBFM_AB_Type.cpp)
# # cuda_add_executable( mainCUDASimulator
# #     src/pscbfm/mainCUDASimulatorGPUScBFM_AB_Type.cpp
# # )
# cuda_add_executable(SimulatorCUDA UpdaterGPUScBFM_AB_Type.cu)
# target_link_libraries(SimulatorCUDA mainCUDASimulator LeMonADE LeMonADEGPU ${CUDA_curand_LIBRARY} cuda )
# target_link_libraries(SimulatorCUDA mainCUDASimulator pscbfmUpdater LeMonADE ${CUDA_curand_LIBRARY} )
# add_executable(mainSimulator mainCUDASimulatorGPUScBFM_AB_Type.cpp)
# target_link_libraries(mainSimulator LeMonADE LeMonADEGPU )
if (NOT DEFINED LEMONADE_INCLUDE_DIR)
message("LEMONADE_INCLUDE_DIR is not provided. If build fails, use -DLEMONADE_INCLUDE_DIR=/path/to/LeMonADE/headers/ or install to default location")
endif()

if (NOT DEFINED LEMONADE_LIBRARY_DIR)
message("LEMONADE_LIBRARY_DIR is not provided. If build fails, use -DLEMONADE_LIBRARY_DIR=/path/to/LeMonADE/lib/ or install to default location")
endif()

include_directories (${LEMONADE_INCLUDE_DIR})
link_directories (${LEMONADE_LIBRARY_DIR})

# if (NOT DEFINED LEMONADEGPU_INCLUDE_DIR)
# message("LEMONADEGPU_INCLUDE_DIR is not provided. If build fails, use -DLEMONADE_INCLUDE_DIR=/path/to/LeMonADE/headers/ or install to default location")
# endif()
# 
# if (NOT DEFINED LEMONADEGPU_LIBRARY_DIR)
# message("LEMONADEGPU_LIBRARY_DIR is not provided. If build fails, use -DLEMONADE_LIBRARY_DIR=/path/to/LeMonADE/lib/ or install to default location")
# endif()
# 
# include_directories (${LEMONADEGPU_INCLUDE_DIR})
# link_directories (${LEMONADEGPU_LIBRARY_DIR})


include_directories ("${LEMONADEGPU_DIR}/include/")
cuda_add_executable( mainSimulator mainCUDASimulatorGPUScBFM_AB_Type.cpp )
# target_link_libraries( mainSimulator cuda LeMonADEGPU LeMonADE )

# This condition was in principle copied from LeMonADEGPU/src/CMakeLists.txt,
# because only if the static library was added as a compile target to CMake,
# it is possible to directly specify the library by its CMake project name
# instead of having to add a library path. By using the CMake target directly,
# CMake also automatically infers the depedency graph correctly when using
# make -j, i.e. target Example1 will only be compiled after target
# libLeMonADE-static finished successfully
if( ( NOT DEFINED project_build_static ) OR ( project_build_static ) )
    target_link_libraries(mainSimulator libLeMonADEGPU-static LeMonADE cuda ${CUDA_curand_LIBRARY} )
else()
    link_directories ("${CMAKE_BINARY_DIR}/lib/")
    target_link_libraries(mainSimulator LeMonADEGPU LeMonADE cuda ${CUDA_curand_LIBRARY} )
endif()