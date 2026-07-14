cmake_minimum_required(VERSION 3.20)

project(renorm_native LANGUAGES CXX)

# ============================================================
# C++ STANDARD
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# CORE SOURCES
# ============================================================

set(CORE_SOURCES

    #----------------------------------------------------------
    # CORE
    #----------------------------------------------------------

    src/tensor.cpp
    src/kernel_context.cpp
    src/compute.cpp
    src/memory.cpp

    #----------------------------------------------------------
    # WORKSPACE
    #----------------------------------------------------------

    src/workspace_allocator.cpp

    #----------------------------------------------------------
    # GRAPH
    #----------------------------------------------------------

    src/graph/graph.cpp

    #----------------------------------------------------------
    # SEARCH
    #----------------------------------------------------------

    src/search/bfs.cpp
    src/search/dijkstra.cpp
    src/search/astar.cpp

    #----------------------------------------------------------
    # BENCHMARK SUPPORT
    #----------------------------------------------------------

    src/benchmark/csv_writer.cpp
    src/benchmark/graph_generator.cpp

    #----------------------------------------------------------
    # COMPILER
    #----------------------------------------------------------

    src/compiler/renorm_compiler.cpp

    src/compiler/passes/pass_dead_end.cpp
    src/compiler/passes/pass_reachability.cpp
    src/compiler/passes/pass_chain_compression.cpp
    src/compiler/passes/pass_equivalence_folding.cpp

    #----------------------------------------------------------
    # RUNTIME
    #----------------------------------------------------------

    src/runtime/thread_pool.cpp
    src/runtime/scheduler.cpp

    src/runtime/renorm_runtime.cpp

    src/runtime/blocked_matrix.cpp
    src/runtime/blocked_matrix_builder.cpp

    # ***** THIS WAS MISSING *****
    src/runtime/compiled_block_builder.cpp

    src/runtime/frontier.cpp
    src/runtime/frontier_builder.cpp
    src/runtime/frontier_executor.cpp
    src/runtime/frontier_executor_gemm.cpp

    src/runtime/graph_compiler.cpp
    src/runtime/graph_matrix_builder.cpp
    src/runtime/renorm_optimizer.cpp

    #----------------------------------------------------------
    # REASONING ENGINE
    #----------------------------------------------------------

    src/runtime/reasoning/reasoning_engine.cpp
    src/runtime/reasoning/pattern_detector.cpp
    src/runtime/reasoning/prediction_engine.cpp
    src/runtime/reasoning/heuristic_engine.cpp
    src/runtime/reasoning/decision_engine.cpp

    src/runtime/dense_matrix.cpp
    src/runtime/dense_vector.cpp

    #----------------------------------------------------------
    # CPU
    #----------------------------------------------------------

    src/cpu/cpu_features.cpp

    #----------------------------------------------------------
    # GEMM
    #----------------------------------------------------------

    src/gemm/gemm.cpp
    src/gemm/gemm_scalar.cpp
    src/gemm/gemm_blocked.cpp
    src/gemm/gemm_avx2.cpp
    src/gemm/gemm_parallel.cpp
    src/gemm/gemm_dispatch.cpp
    src/gemm/gemm_microkernel_avx2.cpp
    src/gemm/packing.cpp
    src/gemm/packing_panel.cpp
    src/gemm/tuner.cpp
	src/runtime/compiled_region_builder.cpp
	src/runtime/compiled_block_builder.cpp
	src/runtime/adaptive_pruner.cpp
)

# ============================================================
# LIBRARY
# ============================================================

add_library(renorm_core STATIC ${CORE_SOURCES})

target_include_directories(
    renorm_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# ============================================================
# COMPILER OPTIONS
# ============================================================

if(MSVC)

    target_compile_options(
        renorm_core
        PRIVATE
            /O2
            /EHsc
            /arch:AVX2
    )

else()

    target_compile_options(
        renorm_core
        PRIVATE
            -O3
            -mavx2
            -mfma
    )

endif()

# ============================================================
# TESTING
# ============================================================

enable_testing()

# ------------------------------------------------------------
# GEMM TEST
# ------------------------------------------------------------

add_executable(
    test_gemm
    tests/test_gemm.cpp
)

target_link_libraries(
    test_gemm
    PRIVATE
        renorm_core
)

add_test(
    NAME GEMM_Test
    COMMAND test_gemm
)

# ------------------------------------------------------------
# VALIDATION TEST
# ------------------------------------------------------------

add_executable(
    test_validation
    tests/test_validation.cpp
)

target_link_libraries(
    test_validation
    PRIVATE
        renorm_core
)

add_test(
    NAME Validation_Test
    COMMAND test_validation
)

# ------------------------------------------------------------
# GEMM BENCHMARK
# ------------------------------------------------------------

add_executable(
    benchmark_gemm
    tests/benchmark_gemm.cpp
)

target_link_libraries(
    benchmark_gemm
    PRIVATE
        renorm_core
)

# ------------------------------------------------------------
# RENORM BENCHMARK
# ------------------------------------------------------------

add_executable(
    benchmark_reasoning
    src/benchmark/main.cpp
    src/benchmark/benchmark_runner.cpp
)

target_link_libraries(
    benchmark_reasoning
    PRIVATE
        renorm_core
)