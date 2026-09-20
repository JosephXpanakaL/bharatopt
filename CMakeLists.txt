cmake_minimum_required(VERSION 3.20)
project(bharatopt VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure consistent output paths across single and multi-config generators
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
if(CMAKE_CONFIGURATION_TYPES)
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR}/lib)
  endforeach()
endif()

option(BHARATOPT_ENABLE_CUDA "Build CUDA backend" OFF)

set(BHARATOPT_CORE_SOURCES
  src/mps.cpp
  src/solver.cpp
  src/qp.cpp
  src/preprocess.cpp
  src/interior_point.cpp
  src/bounded_iis.cpp
  src/pooling.cpp
  src/refinery_model.cpp
  src/refinery_planner.cpp
  src/simplex.cpp
  src/lp_writer.cpp
)

if(NOT BHARATOPT_ENABLE_CUDA)
  list(APPEND BHARATOPT_CORE_SOURCES src/cuda_stub.cpp)
endif()

add_library(bharatopt STATIC ${BHARATOPT_CORE_SOURCES})
target_include_directories(bharatopt PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(BHARATOPT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(bharatopt_cuda STATIC src/cuda_backend.cu)
  target_include_directories(bharatopt_cuda PUBLIC include)
  set_target_properties(bharatopt_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 86
  )
  target_compile_definitions(bharatopt PUBLIC BHARATOPT_CUDA_ENABLED=1)
  target_link_libraries(bharatopt_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)
  target_link_libraries(bharatopt PUBLIC bharatopt_cuda)
endif()

add_executable(bharatopt_cli src/main.cpp)
set_target_properties(bharatopt_cli PROPERTIES OUTPUT_NAME bharatopt)
target_link_libraries(bharatopt_cli PRIVATE bharatopt)

enable_testing()
add_executable(bharatopt_tests tests/test_solver.cpp)
target_link_libraries(bharatopt_tests PRIVATE bharatopt)
add_test(NAME bharatopt_tests COMMAND bharatopt_tests)

add_executable(bharatopt_bounded_iis_tests tests/test_bounded_iis.cpp)
target_link_libraries(bharatopt_bounded_iis_tests PRIVATE bharatopt)
add_test(NAME bharatopt_bounded_iis_tests COMMAND bharatopt_bounded_iis_tests)

add_executable(bharatopt_refinery_model_tests tests/test_refinery_model.cpp)
target_link_libraries(bharatopt_refinery_model_tests PRIVATE bharatopt)
add_test(NAME bharatopt_refinery_model_tests COMMAND bharatopt_refinery_model_tests)

add_executable(bharatopt_refinery_planner_tests tests/test_refinery_planner.cpp)
target_link_libraries(bharatopt_refinery_planner_tests PRIVATE bharatopt)
add_test(NAME bharatopt_refinery_planner_tests COMMAND bharatopt_refinery_planner_tests)

include(GNUInstallDirs)
install(TARGETS bharatopt bharatopt_cli
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set_tests_properties(bharatopt_refinery_model_tests PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
