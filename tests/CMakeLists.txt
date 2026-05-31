cmake_minimum_required(VERSION 3.16)

project(AutomaticDifferentiationTests LANGUAGES C CXX)

set(AUTODIFF_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../lib" CACHE PATH "Install prefix produced by the parent Rakefile")
set(CPPADCG_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cppadcg_benchmark_runtime")
set(CPPADCG_RUNTIME_BUILD_DIR "${CPPADCG_RUNTIME_DIR}/build")
set(CPPADCG_MODEL_NAME "autodiff_codegen_model")
set(CPPADCG_RUNTIME_LIBRARY "${CPPADCG_RUNTIME_BUILD_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${CPPADCG_MODEL_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(CPPADCG_HESSIAN_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cppadcg_hessian_runtime")
set(CPPADCG_HESSIAN_RUNTIME_BUILD_DIR "${CPPADCG_HESSIAN_RUNTIME_DIR}/build")
set(CPPADCG_HESSIAN_MODEL_NAME "autodiff_hessian_model")
set(CPPADCG_HESSIAN_RUNTIME_LIBRARY "${CPPADCG_HESSIAN_RUNTIME_BUILD_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${CPPADCG_HESSIAN_MODEL_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")

set(CPPADCG_RUNTIME_CONFIGURE_ARGS
  -S "${CPPADCG_RUNTIME_DIR}"
  -B "${CPPADCG_RUNTIME_BUILD_DIR}"
)

set(CPPADCG_RUNTIME_BUILD_ARGS
  --build "${CPPADCG_RUNTIME_BUILD_DIR}"
)

set(CPPADCG_HESSIAN_RUNTIME_CONFIGURE_ARGS
  -S "${CPPADCG_HESSIAN_RUNTIME_DIR}"
  -B "${CPPADCG_HESSIAN_RUNTIME_BUILD_DIR}"
)

set(CPPADCG_HESSIAN_RUNTIME_BUILD_ARGS
  --build "${CPPADCG_HESSIAN_RUNTIME_BUILD_DIR}"
)

if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND CPPADCG_RUNTIME_BUILD_ARGS --config "$<CONFIG>")
  list(APPEND CPPADCG_HESSIAN_RUNTIME_BUILD_ARGS --config "$<CONFIG>")
elseif(CMAKE_BUILD_TYPE)
  list(APPEND CPPADCG_RUNTIME_CONFIGURE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
  list(APPEND CPPADCG_HESSIAN_RUNTIME_CONFIGURE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 CONFIG REQUIRED
  PATHS "${AUTODIFF_PREFIX}/share/eigen3/cmake"
  NO_DEFAULT_PATH
)

find_library(CPPAD_LIB
  NAMES cppad_lib
  PATHS "${AUTODIFF_PREFIX}/lib"
  NO_DEFAULT_PATH
  REQUIRED
)

add_executable(autodiff_gradient_benchmark
  gradient_benchmark.cc
)

add_executable(autodiff_gradient_benchmark_generate
  gradient_benchmark_generate.cc
)

add_executable(autodiff_hessian_benchmark
  hessian_benchmark.cc
)

add_executable(autodiff_hessian_benchmark_generate
  hessian_benchmark_generate.cc
)

target_include_directories(autodiff_gradient_benchmark PRIVATE
  "${AUTODIFF_PREFIX}/include"
)

target_include_directories(autodiff_gradient_benchmark_generate PRIVATE
  "${AUTODIFF_PREFIX}/include"
)

target_include_directories(autodiff_hessian_benchmark PRIVATE
  "${AUTODIFF_PREFIX}/include"
)

target_include_directories(autodiff_hessian_benchmark_generate PRIVATE
  "${AUTODIFF_PREFIX}/include"
)

target_link_libraries(autodiff_gradient_benchmark PRIVATE
  Eigen3::Eigen
  "${CPPAD_LIB}"
  ${CMAKE_DL_LIBS}
)

target_link_libraries(autodiff_gradient_benchmark_generate PRIVATE
  Eigen3::Eigen
  "${CPPAD_LIB}"
  ${CMAKE_DL_LIBS}
)

target_link_libraries(autodiff_hessian_benchmark PRIVATE
  Eigen3::Eigen
  "${CPPAD_LIB}"
  ${CMAKE_DL_LIBS}
)

target_link_libraries(autodiff_hessian_benchmark_generate PRIVATE
  Eigen3::Eigen
  "${CPPAD_LIB}"
  ${CMAKE_DL_LIBS}
)

target_compile_definitions(autodiff_gradient_benchmark PRIVATE
  AUTODIFF_CODEGEN_LIBRARY_PATH="${CPPADCG_RUNTIME_LIBRARY}"
  AUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_MODEL_NAME}"
)

target_compile_definitions(autodiff_gradient_benchmark_generate PRIVATE
  AUTODIFF_CODEGEN_RUNTIME_DIR="${CPPADCG_RUNTIME_DIR}"
  AUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_MODEL_NAME}"
)

target_compile_definitions(autodiff_hessian_benchmark PRIVATE
  AUTODIFF_CODEGEN_LIBRARY_PATH="${CPPADCG_HESSIAN_RUNTIME_LIBRARY}"
  AUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_HESSIAN_MODEL_NAME}"
)

target_compile_definitions(autodiff_hessian_benchmark_generate PRIVATE
  AUTODIFF_CODEGEN_RUNTIME_DIR="${CPPADCG_HESSIAN_RUNTIME_DIR}"
  AUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_HESSIAN_MODEL_NAME}"
)

add_custom_command(
  OUTPUT "${CPPADCG_RUNTIME_LIBRARY}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CPPADCG_RUNTIME_DIR}"
  COMMAND $<TARGET_FILE:autodiff_gradient_benchmark_generate> --output-dir "${CPPADCG_RUNTIME_DIR}"
  COMMAND "${CMAKE_COMMAND}" ${CPPADCG_RUNTIME_CONFIGURE_ARGS}
  COMMAND "${CMAKE_COMMAND}" ${CPPADCG_RUNTIME_BUILD_ARGS}
  DEPENDS autodiff_gradient_benchmark_generate
  COMMENT "Generate and build the CppADCodegen benchmark library"
  VERBATIM
)

add_custom_target(cppadcg_generated_library ALL
  DEPENDS "${CPPADCG_RUNTIME_LIBRARY}"
)

add_dependencies(autodiff_gradient_benchmark cppadcg_generated_library)

add_custom_command(
  OUTPUT "${CPPADCG_HESSIAN_RUNTIME_LIBRARY}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CPPADCG_HESSIAN_RUNTIME_DIR}"
  COMMAND $<TARGET_FILE:autodiff_hessian_benchmark_generate> --output-dir "${CPPADCG_HESSIAN_RUNTIME_DIR}"
  COMMAND "${CMAKE_COMMAND}" ${CPPADCG_HESSIAN_RUNTIME_CONFIGURE_ARGS}
  COMMAND "${CMAKE_COMMAND}" ${CPPADCG_HESSIAN_RUNTIME_BUILD_ARGS}
  DEPENDS autodiff_hessian_benchmark_generate
  COMMENT "Generate and build the CppADCodegen Hessian benchmark library"
  VERBATIM
)

add_custom_target(cppadcg_generated_hessian_library ALL
  DEPENDS "${CPPADCG_HESSIAN_RUNTIME_LIBRARY}"
)

add_dependencies(autodiff_hessian_benchmark cppadcg_generated_hessian_library)

enable_testing()

add_test(
  NAME autodiff_gradient_benchmark
  COMMAND autodiff_gradient_benchmark
)

add_test(
  NAME autodiff_hessian_benchmark
  COMMAND autodiff_hessian_benchmark
)
