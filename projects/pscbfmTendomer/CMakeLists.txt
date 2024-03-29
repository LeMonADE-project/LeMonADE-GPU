cmake_minimum_required(VERSION 3.1)

if (NOT DEFINED LEMONADE_INCLUDE_DIR)
message("LEMONADE_INCLUDE_DIR is not provided. If build fails, use -DLEMONADE_INCLUDE_DIR=/path/to/LeMonADE/headers/ or install to default location")
endif()

if (NOT DEFINED LEMONADE_LIBRARY_DIR)
message("LEMONADE_LIBRARY_DIR is not provided. If build fails, use -DLEMONADE_LIBRARY_DIR=/path/to/LeMonADE/lib/ or install to default location")
endif()

include_directories (${LEMONADE_INCLUDE_DIR})
link_directories (${LEMONADE_LIBRARY_DIR})


include_directories ("${LEMONADEGPU_DIR}/include/")
cuda_add_executable( mainSimulatorTendomer mainCUDASimulatorGPUScBFM_Tendomer.cpp )


if( ( NOT DEFINED project_build_static ) OR ( project_build_static ) )
    target_link_libraries(mainSimulatorTendomer libLeMonADEGPU-static LeMonADE cuda gsl gslcblas ${CUDA_curand_LIBRARY} )
else()
    link_directories ("${CMAKE_BINARY_DIR}/lib/")
    target_link_libraries(mainSimulatorTendomer LeMonADEGPU LeMonADE cuda gsl gslcblas ${CUDA_curand_LIBRARY} )
endif()

# cuda_add_executable( mainSimulatorTendomerConnection mainCUDASimulatorGPUScBFM_TendomerConnection.cpp )

# if( ( NOT DEFINED project_build_static ) OR ( project_build_static ) )
#     target_link_libraries(mainSimulatorTendomerConnection libLeMonADEGPU-static LeMonADE cuda gsl gslcblas ${CUDA_curand_LIBRARY} )
# else()
#     link_directories ("${CMAKE_BINARY_DIR}/lib/")
#     target_link_libraries(mainSimulatorTendomerConnection LeMonADEGPU LeMonADE cuda gsl gslcblas ${CUDA_curand_LIBRARY} )
# endif()