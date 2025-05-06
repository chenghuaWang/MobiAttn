cmake_minimum_required(VERSION 3.21)
project(
  MobiAttn
  VERSION 1.0.0
  LANGUAGES CXX C ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MOBI_ATTN_ENABLE_TEST "Enable MobiAttn test" ON)
option(MOBI_ATTN_ENABLE_BENCHMARK "Enable MobiAttn benchmark" ON)
option(MOBI_ATTN_ENABLE_EXAMPLE "Enable MobiAttn example" ON)

message(STATUS "CXX Compiler=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CXX Compiler version=${CMAKE_CXX_COMPILER_VERSION}")

set(KLEIDIAI_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/third_party/kleidiai)
set(MOBI_ATTN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add Attn Impl
add_subdirectory(mobi_attn/flash_attn_2)

# Enable Test
if (MOBI_ATTN_ENABLE_TEST)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(tests)
endif (MOBI_ATTN_ENABLE_TEST)

# Enable Benchmark
if (MOBI_ATTN_ENABLE_BENCHMARK)
  set(BENCHMARK_ENABLE_TESTING OFF)
  add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
  add_subdirectory(benchmark)
endif (MOBI_ATTN_ENABLE_BENCHMARK)

if (MOBI_ATTN_ENABLE_EXAMPLE)
  add_subdirectory(examples)
endif (MOBI_ATTN_ENABLE_EXAMPLE)
