cmake_minimum_required(VERSION 3.14)
project(Lattice VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Lattice library (shared between main and tests) ─────────────────────────
add_library(lattice_lib
    src/vector.cpp
)
target_include_directories(lattice_lib PUBLIC src)

# ── Main executable ─────────────────────────────────────────────────────────
add_executable(lattice src/main.cpp)
target_link_libraries(lattice PRIVATE lattice_lib)

# ── Tests (GoogleTest via FetchContent) ─────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(lattice_tests
    tests/test_vector.cpp
)
target_link_libraries(lattice_tests PRIVATE lattice_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(lattice_tests)
