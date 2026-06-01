cmake_minimum_required(VERSION 3.16)

project(HyperDualTest LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standard C++17
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Optimization flags (optional, but keep release mode)
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(OPT_FLAGS "-O2")
elseif(MSVC)
  set(OPT_FLAGS "/O2")
else()
  set(OPT_FLAGS "-O3")
endif()

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
# Executable for testing Dual
# ---------------------------------------------------------------------------
add_executable( dual_test "${SRC_DIR}/test_dual.cc" )
target_include_directories( dual_test PRIVATE "${INCLUDE_DIR}" )
target_compile_options( dual_test PRIVATE ${OPT_FLAGS} )
# ---------------------------------------------------------------------------
# Executable for testing HyperDual
# ---------------------------------------------------------------------------
add_executable(hyperdual_test "${SRC_DIR}/test_hyper_dual.cc")
target_include_directories(hyperdual_test PRIVATE "${INCLUDE_DIR}")
target_compile_options(hyperdual_test PRIVATE ${OPT_FLAGS})
# ---------------------------------------------------------------------------
# Executable test1
# ---------------------------------------------------------------------------
add_executable( test1 "${SRC_DIR}/test1.cc" )
target_include_directories( test1 PRIVATE "${INCLUDE_DIR}" )
target_compile_options( test1 PRIVATE ${OPT_FLAGS} )
# ---------------------------------------------------------------------------
# Executable test2
# ---------------------------------------------------------------------------
add_executable( test2 "${SRC_DIR}/test2.cc" )
target_include_directories( test2 PRIVATE "${INCLUDE_DIR}" )
target_compile_options( test2 PRIVATE ${OPT_FLAGS} )

# ---------------------------------------------------------------------------
# Enable testing with CTest
# ---------------------------------------------------------------------------
enable_testing()
add_test(NAME HyperDualTest COMMAND hyperdual_test)

# Optional: print a message
message(STATUS "HyperDual test configured. Run 'make hyperdual_test' or 'ctest'.")
