cmake_minimum_required(VERSION 3.16)
project(hw4_matrix_multiply CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

find_package(benchmark QUIET)
if(NOT TARGET benchmark::benchmark)
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.5
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(google_benchmark)
endif()

set(BLOCK_SIZES 16 32 64 128 256 512 CACHE STRING "Block sizes to build")
set(FILTER "." CACHE STRING "Benchmark name regex; '.' = all methods")


set(VECLIB_THREADS 1 CACHE STRING "VECLIB_MAXIMUM_THREADS for the run targets")

if(APPLE)
    set(RUN_PREFIX
        ${CMAKE_COMMAND} -E env VECLIB_MAXIMUM_THREADS=${VECLIB_THREADS}
        caffeinate -i)
else()
    set(RUN_PREFIX "")
endif()

set(run_all_commands "")
set(verify_all_commands "")
set(run_all_targets "")

foreach(bs IN LISTS BLOCK_SIZES)
    add_executable(matmul-bs${bs}
        src/main.cpp src/verify.cpp src/matrix-multiply.cpp src/benchmark.cpp)
    target_compile_definitions(matmul-bs${bs} PRIVATE BS=${bs})
    target_compile_options(matmul-bs${bs} PRIVATE -O3 -Wall -Wextra)
    target_link_libraries(matmul-bs${bs} PRIVATE benchmark::benchmark)
    if(APPLE)
        target_compile_definitions(matmul-bs${bs} PRIVATE ACCELERATE_NEW_LAPACK)
        target_link_libraries(matmul-bs${bs} PRIVATE "-framework Accelerate")
    endif()

    add_custom_target(run-bs${bs}
        COMMAND ${RUN_PREFIX} $<TARGET_FILE:matmul-bs${bs}> "--benchmark_filter=${FILTER}"
        DEPENDS matmul-bs${bs} USES_TERMINAL VERBATIM)

    list(APPEND run_all_commands
        COMMAND ${RUN_PREFIX} $<TARGET_FILE:matmul-bs${bs}> "--benchmark_filter=${FILTER}")
    list(APPEND verify_all_commands COMMAND $<TARGET_FILE:matmul-bs${bs}> --verify)
    list(APPEND run_all_targets matmul-bs${bs})
endforeach()

add_custom_target(run-all
    ${run_all_commands}
    DEPENDS ${run_all_targets} USES_TERMINAL VERBATIM)
    
add_custom_target(verify-all
    ${verify_all_commands}
    DEPENDS ${run_all_targets} USES_TERMINAL VERBATIM)

find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} found: enabling matmul-opencv target")
    add_executable(matmul-opencv src/opencv_benchmark.cpp src/matrix-multiply.cpp)
    target_compile_definitions(matmul-opencv PRIVATE BS=64)
    target_compile_options(matmul-opencv PRIVATE -O3 -Wall -Wextra)
    target_link_libraries(matmul-opencv PRIVATE benchmark::benchmark ${OpenCV_LIBS})
    if(APPLE)
        target_compile_definitions(matmul-opencv PRIVATE ACCELERATE_NEW_LAPACK)
        target_link_libraries(matmul-opencv PRIVATE "-framework Accelerate")
    endif()

    add_custom_target(run-opencv
        COMMAND ${RUN_PREFIX} $<TARGET_FILE:matmul-opencv> "--benchmark_filter=${FILTER}"
        DEPENDS matmul-opencv USES_TERMINAL VERBATIM)
else()
    message(STATUS "OpenCV not found: skipping optional matmul-opencv target")
endif()
