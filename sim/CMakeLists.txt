cmake_minimum_required(VERSION 3.16)
project(whoop-airfoil-sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# RLtools (https://github.com/rl-tools/rl-tools) is header-only: only its include directory is used. Its own
# CMakeLists.txt is not added because it downloads optional dependencies into the checkout, which is shared with other
# projects and must stay unchanged. Tested at commit b32d9985.
set(RL_TOOLS_DIR "" CACHE PATH "RLtools checkout (tested at commit b32d9985)")
if(NOT EXISTS "${RL_TOOLS_DIR}/include/rl_tools/version.h")
    message(FATAL_ERROR "set RL_TOOLS_DIR to an RLtools checkout (commit b32d9985), e.g. -DRL_TOOLS_DIR=.../rl_tools")
endif()
add_library(rltools INTERFACE)
target_include_directories(rltools INTERFACE "${RL_TOOLS_DIR}/include")
if(MSVC)
    target_compile_options(rltools INTERFACE /bigobj /permissive-)
endif()

# Intel MKL for the matrix products of the networks (as the Fixed-Wing project); without it RLtools uses its own loops.
# Configure with -DWHOOP_MKL=ON -DMKL_DIR=<venv>/Library/lib/cmake/mkl (Windows) or <venv>/lib/cmake/mkl (Linux)
option(WHOOP_MKL "Use Intel MKL (find_package(MKL CONFIG))" OFF)
if(WHOOP_MKL)
    set(MKL_INTERFACE lp64 CACHE STRING "")
    set(MKL_THREADING sequential CACHE STRING "")
    find_package(MKL CONFIG REQUIRED)
    target_link_libraries(rltools INTERFACE MKL::MKL)
    target_compile_definitions(rltools INTERFACE RL_TOOLS_BACKEND_ENABLE_MKL)
endif()

# the whoop and plate parameters come from the JSON shared with the identification (sysid/): a header is generated
# from it at build time, so the simulator never holds its own copy of the numbers
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(WHOOP_PARAMS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/../params/whoop_nominal.json" CACHE FILEPATH "whoop and plate parameters")
set(WHOOP_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
add_custom_command(
    OUTPUT "${WHOOP_GENERATED_DIR}/whoop/params.h"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${WHOOP_GENERATED_DIR}/whoop"
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_params.py" "${WHOOP_PARAMS_JSON}" "${WHOOP_GENERATED_DIR}/whoop/params.h"
    DEPENDS "${WHOOP_PARAMS_JSON}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_params.py"
    COMMENT "Generating whoop/params.h from ${WHOOP_PARAMS_JSON}")
add_custom_target(whoop_params DEPENDS "${WHOOP_GENERATED_DIR}/whoop/params.h")

function(whoop_executable target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${WHOOP_GENERATED_DIR}")
    target_link_libraries(${target} PRIVATE rltools)
    add_dependencies(${target} whoop_params)
endfunction()

# unit tests: no fast-math and no contraction into fused multiply-adds, so that "S = 0 gives exactly the L2F
# dynamics" is tested bit for bit
whoop_executable(whoop_test tests/test_whoop.cpp)
# the tests read the parameter files at run time (whoop/params_runtime.h) and compare them with the compiled ones
target_compile_definitions(whoop_test PRIVATE WHOOP_PARAMS_JSON_PATH="${WHOOP_PARAMS_JSON}" WHOOP_PARAMS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../params")
if(NOT MSVC)
    target_compile_options(whoop_test PRIVATE -O2 -ffp-contract=off)
endif()

# training: {SAC, TD3} x {hover, velocity task} x {no plate, plate}. The plate is a compile-time switch (WHOOP_PLATE)
# that sets the nominal plate area (0 without) and the mass and inertia of the vehicle; the code is the same
foreach(algorithm sac td3)
    foreach(task hover velocity)
        foreach(plate noplate plate)
            set(target whoop_train_${algorithm}_${task}_${plate})
            whoop_executable(${target} src/train.cpp)
            target_compile_definitions(${target} PRIVATE
                WHOOP_TD3=$<STREQUAL:${algorithm},td3>
                WHOOP_TASK_VELOCITY=$<STREQUAL:${task},velocity>
                WHOOP_PLATE=$<STREQUAL:${plate},plate>)
            if(MSVC)
                target_compile_options(${target} PRIVATE /fp:fast)
            else()
                target_compile_options(${target} PRIVATE -O3 -march=native)
            endif()
        endforeach()
    endforeach()
endforeach()
