cmake_minimum_required(VERSION 3.16)

project(AutomaticDifferentiationTests LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Definizione globale di NDEBUG per disabilitare le assertion in tutti i target
# ---------------------------------------------------------------------------
add_compile_definitions(NDEBUG)

# ---------------------------------------------------------------------------
# Opzioni di ottimizzazione comuni (usate sia per i target normali che per Enzyme)
# Riconosce anche AppleClang come Clang
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(AUTODIFF_OPT_FLAGS "-O2")
elseif(MSVC)
  set(AUTODIFF_OPT_FLAGS "/O2")
else()
  set(AUTODIFF_OPT_FLAGS "-O3")
endif()

# Generator expression per i target normali (supporto multi‑compilatore)
# Ora include anche AppleClang
set(OPTIMIZATION_FLAGS
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:${AUTODIFF_OPT_FLAGS}>
    $<$<CXX_COMPILER_ID:MSVC>:${AUTODIFF_OPT_FLAGS}>
)
set(WARNING_SUPPRESSION_FLAGS
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-ignored-qualifiers>
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-missing-field-initializers>
)

# ---------------------------------------------------------------------------
# Percorsi e variabili per le librerie generate da CppADCodegen
# ---------------------------------------------------------------------------
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
set(CPPADCG_RUNTIME_BUILD_COMMAND "${CMAKE_COMMAND}" ${CPPADCG_RUNTIME_BUILD_ARGS})

set(CPPADCG_HESSIAN_RUNTIME_CONFIGURE_ARGS
  -S "${CPPADCG_HESSIAN_RUNTIME_DIR}"
  -B "${CPPADCG_HESSIAN_RUNTIME_BUILD_DIR}"
)

set(CPPADCG_HESSIAN_RUNTIME_BUILD_ARGS
  --build "${CPPADCG_HESSIAN_RUNTIME_BUILD_DIR}"
)
set(CPPADCG_HESSIAN_RUNTIME_BUILD_COMMAND "${CMAKE_COMMAND}" ${CPPADCG_HESSIAN_RUNTIME_BUILD_ARGS})

set(AUTODIFF_JOBSERVER_AWARE_ARGS)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28")
  list(APPEND AUTODIFF_JOBSERVER_AWARE_ARGS JOB_SERVER_AWARE TRUE)
endif()

# ---------------------------------------------------------------------------
# Forza ottimizzazioni per le librerie runtime generate da CppADCodegen (Release)
# Propaga gli stessi flag di ottimizzazione utilizzati nel progetto principale,
# forzando i flag del compilatore C e C++ (non solo le varianti _RELEASE)
# ---------------------------------------------------------------------------
if(CMAKE_CONFIGURATION_TYPES)
  # Multi-config generator (Visual Studio): usa Release esplicitamente
  list(APPEND CPPADCG_RUNTIME_BUILD_ARGS --config Release)
  list(APPEND CPPADCG_HESSIAN_RUNTIME_BUILD_ARGS --config Release)
else()
  # Single-config generator (Makefile, Ninja): imposta CMAKE_BUILD_TYPE=Release
  list(APPEND CPPADCG_RUNTIME_CONFIGURE_ARGS -DCMAKE_BUILD_TYPE=Release)
  list(APPEND CPPADCG_HESSIAN_RUNTIME_CONFIGURE_ARGS -DCMAKE_BUILD_TYPE=Release)
endif()

# Passa i flag di ottimizzazione direttamente ai compilatori (C e C++)
# per assicurarsi che vengano usati anche se il sottoprogetto ignora le variabili _RELEASE.
# Nota: usiamo -DCMAKE_C_FLAGS e -DCMAKE_CXX_FLAGS per tutte le configurazioni.
list(APPEND CPPADCG_RUNTIME_CONFIGURE_ARGS
  -DCMAKE_C_FLAGS=${AUTODIFF_OPT_FLAGS}
  -DCMAKE_CXX_FLAGS=${AUTODIFF_OPT_FLAGS}
)
list(APPEND CPPADCG_HESSIAN_RUNTIME_CONFIGURE_ARGS
  -DCMAKE_C_FLAGS=${AUTODIFF_OPT_FLAGS}
  -DCMAKE_CXX_FLAGS=${AUTODIFF_OPT_FLAGS}
)

# Se l'utente ha esplicitamente passato un generatore Makefile o Ninja, usa lo stesso comando
if(CMAKE_GENERATOR MATCHES "Makefiles")
  set(CPPADCG_RUNTIME_BUILD_COMMAND "${CMAKE_MAKE_PROGRAM}" -C "${CPPADCG_RUNTIME_BUILD_DIR}")
  set(CPPADCG_HESSIAN_RUNTIME_BUILD_COMMAND "${CMAKE_MAKE_PROGRAM}" -C "${CPPADCG_HESSIAN_RUNTIME_BUILD_DIR}")
elseif(CMAKE_GENERATOR MATCHES "Ninja")
  set(CPPADCG_RUNTIME_BUILD_COMMAND "${CMAKE_MAKE_PROGRAM}" -C "${CPPADCG_RUNTIME_BUILD_DIR}")
  set(CPPADCG_HESSIAN_RUNTIME_BUILD_COMMAND "${CMAKE_MAKE_PROGRAM}" -C "${CPPADCG_HESSIAN_RUNTIME_BUILD_DIR}")
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

# ---------------------------------------------------------------------------
# Eseguibili di generazione (compilati con ottimizzazioni e NDEBUG)
# ---------------------------------------------------------------------------
add_executable(autodiff_gradient_benchmark_generate
  gradient_benchmark_generate.cc
)
add_executable(autodiff_hessian_benchmark_generate
  hessian_benchmark_generate.cc
)

target_include_directories(autodiff_gradient_benchmark_generate PRIVATE
  "${AUTODIFF_PREFIX}/include"
)
target_include_directories(autodiff_hessian_benchmark_generate PRIVATE
  "${AUTODIFF_PREFIX}/include"
)

target_link_libraries(autodiff_gradient_benchmark_generate PRIVATE
  Eigen3::Eigen
  "${CPPAD_LIB}"
  ${CMAKE_DL_LIBS}
)
target_link_libraries(autodiff_hessian_benchmark_generate PRIVATE
  Eigen3::Eigen
  "${CPPAD_LIB}"
  ${CMAKE_DL_LIBS}
)

target_compile_definitions(autodiff_gradient_benchmark_generate PRIVATE
  AUTODIFF_CODEGEN_RUNTIME_DIR="${CPPADCG_RUNTIME_DIR}"
  AUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_MODEL_NAME}"
)
target_compile_definitions(autodiff_hessian_benchmark_generate PRIVATE
  AUTODIFF_CODEGEN_RUNTIME_DIR="${CPPADCG_HESSIAN_RUNTIME_DIR}"
  AUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_HESSIAN_MODEL_NAME}"
)

target_compile_options(autodiff_gradient_benchmark_generate PRIVATE ${OPTIMIZATION_FLAGS})
target_compile_options(autodiff_hessian_benchmark_generate PRIVATE ${OPTIMIZATION_FLAGS})

# ---------------------------------------------------------------------------
# Benchmark eseguibili: normale o con Enzyme
# ---------------------------------------------------------------------------
option(AUTODIFF_ENABLE_ENZYME "Build Enzyme autodiff benchmarks" OFF)

if(NOT AUTODIFF_ENABLE_ENZYME)
  # Compilazione normale (senza Enzyme)
  add_executable(autodiff_gradient_benchmark gradient_benchmark.cc)
  add_executable(autodiff_hessian_benchmark hessian_benchmark.cc)

  target_include_directories(autodiff_gradient_benchmark PRIVATE "${AUTODIFF_PREFIX}/include")
  target_include_directories(autodiff_hessian_benchmark PRIVATE "${AUTODIFF_PREFIX}/include")

  target_link_libraries(autodiff_gradient_benchmark PRIVATE
    Eigen3::Eigen "${CPPAD_LIB}" ${CMAKE_DL_LIBS}
  )
  target_link_libraries(autodiff_hessian_benchmark PRIVATE
    Eigen3::Eigen "${CPPAD_LIB}" ${CMAKE_DL_LIBS}
  )

  target_compile_definitions(autodiff_gradient_benchmark PRIVATE
    AUTODIFF_CODEGEN_LIBRARY_PATH="${CPPADCG_RUNTIME_LIBRARY}"
    AUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_MODEL_NAME}"
  )
  target_compile_definitions(autodiff_hessian_benchmark PRIVATE
    AUTODIFF_CODEGEN_LIBRARY_PATH="${CPPADCG_HESSIAN_RUNTIME_LIBRARY}"
    AUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_HESSIAN_MODEL_NAME}"
  )

  target_compile_options(autodiff_gradient_benchmark PRIVATE ${OPTIMIZATION_FLAGS} ${WARNING_SUPPRESSION_FLAGS})
  target_compile_options(autodiff_hessian_benchmark PRIVATE ${OPTIMIZATION_FLAGS} ${WARNING_SUPPRESSION_FLAGS})

  set(AUTODIFF_GRADIENT_BENCHMARK_COMMAND "$<TARGET_FILE:autodiff_gradient_benchmark>")
  set(AUTODIFF_HESSIAN_BENCHMARK_COMMAND "$<TARGET_FILE:autodiff_hessian_benchmark>")
else()
  # -------------------------------------------------------------------------
  # Build con Enzyme (usa clang++ con il plugin)
  # -------------------------------------------------------------------------
  message(STATUS "Enzyme support enabled – benchmarks will be built with LLVM clang++")

  set(ENZYME_PREFIX "/opt/homebrew/opt/enzyme")
  find_program(ENZYME_CLANGXX
    NAMES clang++
    PATHS "/opt/homebrew/opt/llvm/bin"
    NO_DEFAULT_PATH
    REQUIRED
  )
  find_file(ENZYME_CLANG_PLUGIN
    NAMES ClangEnzyme-22.dylib
    PATHS "${ENZYME_PREFIX}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
  set(ENZYME_INCLUDE_DIR "${ENZYME_PREFIX}/include")

  get_target_property(EIGEN_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
  list(GET EIGEN_INCLUDE_DIRS 0 EIGEN_INCLUDE_DIR)

  # Utilizzo le stesse opzioni di ottimizzazione definite globalmente
  set(ENZYME_BASE_FLAGS
    -std=c++17
    ${AUTODIFF_OPT_FLAGS}
    -DAUTODIFF_HAS_ENZYME
    -DNDEBUG
    -Wall
    -Wextra
    -Wno-ignored-qualifiers
    -Wno-missing-field-initializers
    -Wno-unused-parameter
    -Wno-unused-variable
    -Wno-deprecated-copy
    -I${AUTODIFF_PREFIX}/include
    -I${EIGEN_INCLUDE_DIR}
    -I${ENZYME_INCLUDE_DIR}
    -fplugin=${ENZYME_CLANG_PLUGIN}
  )
  set(ENZYME_LINK_FLAGS
    -L${AUTODIFF_PREFIX}/lib
    -lcppad_lib
    -ldl
  )

  macro(add_enzyme_benchmark TARGET_NAME SOURCE_FILE)
    set(OUTPUT_EXE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}")
    add_custom_command(
      OUTPUT "${OUTPUT_EXE}"
      COMMAND "${ENZYME_CLANGXX}"
        ${ENZYME_BASE_FLAGS}
        ${ARGN}
        ${ENZYME_LINK_FLAGS}
        "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}"
        -o "${OUTPUT_EXE}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/gradient_benchmark_common.hpp"
      COMMENT "Building ${TARGET_NAME} with Enzyme"
      VERBATIM
    )
    add_custom_target(build_${TARGET_NAME} ALL DEPENDS "${OUTPUT_EXE}")
  endmacro()

  add_enzyme_benchmark(autodiff_gradient_benchmark gradient_benchmark.cc
    -DAUTODIFF_CODEGEN_LIBRARY_PATH="${CPPADCG_RUNTIME_LIBRARY}"
    -DAUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_MODEL_NAME}"
  )
  add_enzyme_benchmark(autodiff_hessian_benchmark hessian_benchmark.cc
    -DAUTODIFF_CODEGEN_LIBRARY_PATH="${CPPADCG_HESSIAN_RUNTIME_LIBRARY}"
    -DAUTODIFF_CODEGEN_MODEL_NAME="${CPPADCG_HESSIAN_MODEL_NAME}"
  )

  set(AUTODIFF_GRADIENT_BENCHMARK_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/autodiff_gradient_benchmark")
  set(AUTODIFF_HESSIAN_BENCHMARK_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/autodiff_hessian_benchmark")

  add_compile_definitions(AUTODIFF_HAS_ENZYME)
  target_include_directories(autodiff_gradient_benchmark_generate PRIVATE "${ENZYME_INCLUDE_DIR}")
  target_include_directories(autodiff_hessian_benchmark_generate PRIVATE "${ENZYME_INCLUDE_DIR}")
endif()

# ---------------------------------------------------------------------------
# Generazione e compilazione delle librerie runtime CppADCodegen
# ---------------------------------------------------------------------------
add_custom_command(
  OUTPUT "${CPPADCG_RUNTIME_LIBRARY}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CPPADCG_RUNTIME_DIR}"
  COMMAND $<TARGET_FILE:autodiff_gradient_benchmark_generate> --output-dir "${CPPADCG_RUNTIME_DIR}"
  COMMAND "${CMAKE_COMMAND}" ${CPPADCG_RUNTIME_CONFIGURE_ARGS}
  COMMAND ${CPPADCG_RUNTIME_BUILD_COMMAND}
  DEPENDS autodiff_gradient_benchmark_generate
  COMMENT "Generate and build the CppADCodegen benchmark library"
  ${AUTODIFF_JOBSERVER_AWARE_ARGS}
  VERBATIM
)
add_custom_target(cppadcg_generated_library ALL
  DEPENDS "${CPPADCG_RUNTIME_LIBRARY}"
)

add_custom_command(
  OUTPUT "${CPPADCG_HESSIAN_RUNTIME_LIBRARY}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CPPADCG_HESSIAN_RUNTIME_DIR}"
  COMMAND $<TARGET_FILE:autodiff_hessian_benchmark_generate> --output-dir "${CPPADCG_HESSIAN_RUNTIME_DIR}"
  COMMAND "${CMAKE_COMMAND}" ${CPPADCG_HESSIAN_RUNTIME_CONFIGURE_ARGS}
  COMMAND ${CPPADCG_HESSIAN_RUNTIME_BUILD_COMMAND}
  DEPENDS autodiff_hessian_benchmark_generate
  COMMENT "Generate and build the CppADCodegen Hessian benchmark library"
  ${AUTODIFF_JOBSERVER_AWARE_ARGS}
  VERBATIM
)
add_custom_target(cppadcg_generated_hessian_library ALL
  DEPENDS "${CPPADCG_HESSIAN_RUNTIME_LIBRARY}"
)

if(NOT AUTODIFF_ENABLE_ENZYME)
  add_dependencies(autodiff_gradient_benchmark cppadcg_generated_library)
  add_dependencies(autodiff_hessian_benchmark cppadcg_generated_hessian_library)
endif()

# ---------------------------------------------------------------------------
enable_testing()
add_test(NAME autodiff_gradient_benchmark COMMAND "${AUTODIFF_GRADIENT_BENCHMARK_COMMAND}")
add_test(NAME autodiff_hessian_benchmark COMMAND "${AUTODIFF_HESSIAN_BENCHMARK_COMMAND}")
