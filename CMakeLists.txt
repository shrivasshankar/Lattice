cmake_minimum_required(VERSION 3.14)
project(Lattice VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── ThreadSanitizer (opt-in) ────────────────────────────────────────────────
# -DLATTICE_TSAN=ON builds everything with ThreadSanitizer: the compiler
# instruments every memory access and the runtime reports any data race from
# a single execution (a lucky, output-correct run still trips it). Separate
# build because instrumentation is ~5-15x slower; never used in production.
# Applied to both compile and link so the sanitizer runtime is linked in.
option(LATTICE_TSAN "Build with ThreadSanitizer" OFF)
if(LATTICE_TSAN)
    add_compile_options(-fsanitize=thread -g -O1 -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

# ── Lattice library (shared between main and tests) ─────────────────────────
add_library(lattice_lib
    src/vector.cpp
    src/distance.cpp
    src/search.cpp
    src/hnsw.cpp
    src/allocator.cpp
    src/simd_distance.cpp
    src/thread_pool.cpp
)
target_include_directories(lattice_lib PUBLIC src)

# Thread support (required for std::thread, std::mutex, etc.)
find_package(Threads REQUIRED)
target_link_libraries(lattice_lib PUBLIC Threads::Threads)

# ── Main executable ─────────────────────────────────────────────────────────
add_executable(lattice src/main.cpp)
target_link_libraries(lattice PRIVATE lattice_lib)

# ── Tuning tool ──────────────────────────────────────────────────────────────
add_executable(tune_recall tools/tune_recall.cpp)
target_link_libraries(tune_recall PRIVATE lattice_lib)

# ── Dataset exporter (for the FAISS comparison) ──────────────────────────────
add_executable(export_dataset tools/export_dataset.cpp)
target_link_libraries(export_dataset PRIVATE lattice_lib)

# ── Scaling benchmark ────────────────────────────────────────────────────────
add_executable(scaling_benchmark tools/scaling_benchmark.cpp)
target_link_libraries(scaling_benchmark PRIVATE lattice_lib)

# ── SIFT/GIST benchmark (fvecs/ivecs loader) ─────────────────────────────────
add_executable(sift_benchmark tools/sift_benchmark.cpp)
target_link_libraries(sift_benchmark PRIVATE lattice_lib)

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
    tests/test_distance.cpp
    tests/test_search.cpp
    tests/test_hnsw.cpp
    tests/test_hnsw_search.cpp
    tests/test_allocator.cpp
    tests/test_simd_distance.cpp
    tests/test_thread_pool.cpp
    tests/test_hnsw_serial.cpp
    tests/test_hnsw_parallel.cpp
)
target_link_libraries(lattice_tests PRIVATE lattice_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(lattice_tests)
