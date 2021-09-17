# minimalistic gpuinfo tool printing all properties
set_source_files_properties( gpuinfo.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
# set( CUDA_NVCC_FLAGS ${MINIMAL_CUDA_NVCC_FLAGS} )
cuda_add_executable( gpuinfo gpuinfo.cpp )
target_link_libraries( gpuinfo cuda )