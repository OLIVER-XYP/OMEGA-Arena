cmake_minimum_required(VERSION 3.10)

# CMake 4.x drops compat with cmake_minimum_required < 3.5; override for old deps
set(CMAKE_POLICY_VERSION_MINIMUM "3.5" CACHE STRING "" FORCE)

project(environment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HALITE_ENABLE_CUDA "Build the CUDA turn backend" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)

if(MSVC)
    # define NOMINMAX so we can use std::min/std::max normally
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /EHsc /DNOMINMAX")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /EHsc /DNOMINMAX /Z7")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -pedantic -Wextra -Wno-unused-variable")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pedantic -Wextra -Wno-unused-variable -g -O0 -fno-inline")
    if(NOT MINGW)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=undefined,address")
    endif()
endif()

# versions of cmake before 3.4 always link with -rdynamic on linux, which breaks static linkage with clang
# unfortunately travis right now only has cmake 3.2, so have to do this workaround for now
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

set(HALITE_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/config
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/core/command
    ${CMAKE_SOURCE_DIR}/core/config
    ${CMAKE_SOURCE_DIR}/core/engine
    ${CMAKE_SOURCE_DIR}/core/state
    ${CMAKE_SOURCE_DIR}/events
    ${CMAKE_SOURCE_DIR}/gpu
    ${CMAKE_SOURCE_DIR}/observers
    ${CMAKE_SOURCE_DIR}/protocol
    ${CMAKE_SOURCE_DIR}/rules
    ${CMAKE_SOURCE_DIR}/rules/phases
    ${CMAKE_SOURCE_DIR}/runtime/agents
    ${CMAKE_SOURCE_DIR}/error
    ${CMAKE_SOURCE_DIR}/logging
    ${CMAKE_SOURCE_DIR}/mapgen
    ${CMAKE_SOURCE_DIR}/model
    ${CMAKE_SOURCE_DIR}/networking/common
    ${CMAKE_SOURCE_DIR}/replay
    ${CMAKE_SOURCE_DIR}/util
)

if(WIN32)
    list(APPEND HALITE_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/networking/win32)
else()
    list(APPEND HALITE_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/networking/unix)
endif()

include_directories(${HALITE_INCLUDE_DIRS})

if(NOT WIN32)
add_custom_target(VERSION_CHECK
    ./set_version.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

set(dirs
    ${CMAKE_SOURCE_DIR}/config
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/core/command
    ${CMAKE_SOURCE_DIR}/core/config
    ${CMAKE_SOURCE_DIR}/core/engine
    ${CMAKE_SOURCE_DIR}/core/state
    ${CMAKE_SOURCE_DIR}/events
    ${CMAKE_SOURCE_DIR}/gpu
    ${CMAKE_SOURCE_DIR}/observers
    ${CMAKE_SOURCE_DIR}/protocol
    ${CMAKE_SOURCE_DIR}/rules
    ${CMAKE_SOURCE_DIR}/rules/phases
    ${CMAKE_SOURCE_DIR}/runtime/agents
    ${CMAKE_SOURCE_DIR}/error
    ${CMAKE_SOURCE_DIR}/logging
    ${CMAKE_SOURCE_DIR}/mapgen
    ${CMAKE_SOURCE_DIR}/model
    ${CMAKE_SOURCE_DIR}/networking/common
    ${CMAKE_SOURCE_DIR}/replay
    ${CMAKE_SOURCE_DIR}/util
)

if(WIN32)
    list(APPEND dirs ${CMAKE_SOURCE_DIR}/networking/win32)
else()
    list(APPEND dirs ${CMAKE_SOURCE_DIR}/networking/unix)
endif()

foreach(dir ${dirs})
    file(GLOB_RECURSE SOURCE ${dir}/*.[ch]*)
    list(FILTER SOURCE EXCLUDE REGEX ".*/core/command/(CommandTransaction|Transaction|CommandLegacy)\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaTurnBackend(Cuda|Stub)\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaStateMirror(Cuda|Stub)\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaPhaseRunner(Cuda|Stub)\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaCaptureStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaCombatStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaConstructionStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaDumpStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaHaliteRebalanceStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaInspirationStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaDefendStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaMiningStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaMovementStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaPlunderStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaRegenStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaOverShipTaxStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaSpawnStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaValidationStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/CudaBatchedMapStub\\.cpp$")
    list(FILTER SOURCE EXCLUDE REGEX ".*/gpu/.*\\.cu$")
    set(SOURCE_FILES "${SOURCE_FILES}" ${SOURCE})
endforeach()

list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/rules/phases/OverShipTaxPhase.cpp)
list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/rules/phases/PlunderPhase.cpp)
list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/rules/phases/RegenPhase.cpp)
list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/core/engine/TaskExecutor.cpp)

if(HALITE_ENABLE_CUDA)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build for" FORCE)
    endif()
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(HALITE_ENABLE_CUDA=1)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaTurnBackendCuda.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaStateMirrorCuda.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaPhaseRunnerCuda.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaDevice.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaCapture.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaCombat.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaConstruction.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaDump.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaHaliteRebalance.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaInspiration.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaDefend.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaMining.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaMovement.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaPlunder.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaRegen.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaOverShipTax.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaSpawn.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaValidation.cu)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaBatchedMap.cu)
else()
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaTurnBackendStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaStateMirrorStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaPhaseRunnerStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaCaptureStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaCombatStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaConstructionStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaDumpStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaHaliteRebalanceStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaInspirationStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaDefendStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaMiningStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaMovementStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaPlunderStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaRegenStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaOverShipTaxStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaSpawnStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaValidationStub.cpp)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/gpu/CudaBatchedMapStub.cpp)
endif()

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
include(ExternalProject)

# Vendored dependency archives (catch/nlohmann/tclap/zstd). When the archive is
# present under vendor/ it is used so the build works fully offline; otherwise
# the original download URL is used. ExternalProject forbids mixing a local path
# and a URL in one list, so each source is resolved to a single value here and
# substituted into CMakeLists.txt.in by configure_file below.
set(DEPS_CACHE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor)

# resolve_dep(<out_var> <vendored_filename> <fallback_url>)
function(resolve_dep out_var filename url)
    if(EXISTS "${DEPS_CACHE_DIR}/${filename}")
        set(${out_var} "${DEPS_CACHE_DIR}/${filename}" PARENT_SCOPE)
        message(STATUS "dep ${filename}: using vendored copy")
    else()
        set(${out_var} "${url}" PARENT_SCOPE)
        message(STATUS "dep ${filename}: downloading from ${url}")
    endif()
endfunction()

resolve_dep(CATCH_URL    v2.2.3.zip            https://github.com/catchorg/Catch2/archive/v2.2.3.zip)
resolve_dep(NLOHMANN_URL include.zip           https://github.com/nlohmann/json/releases/download/v3.1.2/include.zip)
resolve_dep(TCLAP_URL    tclap-1.2.2.tar.gz    https://netix.dl.sourceforge.net/project/tclap/tclap-1.2.2.tar.gz)
resolve_dep(ZSTD_URL     v1.3.4.zip            https://github.com/facebook/zstd/archive/v1.3.4.zip)

configure_file(CMakeLists.txt.in deps-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" -Wno-dev -DCMAKE_POLICY_VERSION_MINIMUM=3.5 .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps-download)
if(result)
    message(FATAL_ERROR "CMake step for dependencies failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps-download)
if(result)
    message(FATAL_ERROR "Build step for dependencies failed: ${result}")
endif()

add_subdirectory(${EXTERNAL_INSTALL_LOCATION}/zstd/build/cmake EXCLUDE_FROM_ALL)
add_subdirectory(${EXTERNAL_INSTALL_LOCATION}/catch)

include_directories(${EXTERNAL_INSTALL_LOCATION}/zstd/lib ${EXTERNAL_INSTALL_LOCATION}/nlohmann ${EXTERNAL_INSTALL_LOCATION}/tclap/include)
link_directories(${EXTERNAL_INSTALL_LOCATION}/zstd/build/cmake/lib)

add_library(halite_engine_core INTERFACE)
target_include_directories(halite_engine_core INTERFACE ${HALITE_INCLUDE_DIRS})

add_library(halite_engine_rules INTERFACE)
target_include_directories(halite_engine_rules INTERFACE ${HALITE_INCLUDE_DIRS})
target_link_libraries(halite_engine_rules INTERFACE halite_engine_core)

add_library(halite_engine_protocol INTERFACE)
target_include_directories(halite_engine_protocol INTERFACE ${HALITE_INCLUDE_DIRS})
target_link_libraries(halite_engine_protocol INTERFACE halite_engine_core)

add_library(halite_engine_runtime INTERFACE)
target_include_directories(halite_engine_runtime INTERFACE ${HALITE_INCLUDE_DIRS})
target_link_libraries(halite_engine_runtime INTERFACE halite_engine_protocol)

add_library(halite_engine_replay INTERFACE)
target_include_directories(halite_engine_replay INTERFACE ${HALITE_INCLUDE_DIRS})
target_link_libraries(halite_engine_replay INTERFACE halite_engine_core)

add_library(halite_core OBJECT ${SOURCE_FILES})
if(HALITE_ENABLE_CUDA)
    list(APPEND HALITE_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_include_directories(halite_core PUBLIC ${HALITE_INCLUDE_DIRS})

add_executable(halite $<TARGET_OBJECTS:halite_core> main.cpp)
add_executable(halite_batched_runner $<TARGET_OBJECTS:halite_core> batched_runner/BatchedRunner.cpp)
if(NOT WIN32)
    add_dependencies(halite VERSION_CHECK libzstd_static)
    add_dependencies(halite_batched_runner VERSION_CHECK libzstd_static)
endif()

# --- pybind11 RL step environment (opt-in) -----------------------------------
# Builds halite_pyenv, a Python module exposing a vectorized CPU step env for
# self-play RL. halite_core must be PIC so its objects can link into a shared
# module. Configure with -DHALITE_BUILD_PYENV=ON -Dpybind11_DIR=... .
option(HALITE_BUILD_PYENV "Build the pybind11 RL step environment module" OFF)
if(HALITE_BUILD_PYENV)
    set_target_properties(halite_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # The vendored static zstd must be PIC to link into a shared Python module.
    set_target_properties(libzstd_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(halite_pyenv pyenv/halite_pyenv.cpp $<TARGET_OBJECTS:halite_core>)
    target_include_directories(halite_pyenv PRIVATE ${HALITE_INCLUDE_DIRS})
    target_link_libraries(halite_pyenv PRIVATE libzstd_static)
    add_dependencies(halite_pyenv libzstd_static)
    if(NOT WIN32)
        target_link_libraries(halite_pyenv PRIVATE pthread)
    endif()
endif()

file(GLOB_RECURSE SOURCE ${CMAKE_SOURCE_DIR}/test/*.[ch]*)
set(TEST_FILES "${TEST_FILES}" ${SOURCE})
list(APPEND TEST_FILES ${CMAKE_SOURCE_DIR}/test/ExecutorConsistencyTest.cpp)

if(CMAKE_BUILD_TYPE MATCHES Debug AND NOT WIN32)
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMakeModules)

    file(GLOB_RECURSE TEST_FILES ${CMAKE_SOURCE_DIR}/test/*.[ch]*)
    add_executable(halite_test $<TARGET_OBJECTS:halite_core> ${TEST_FILES} ${CMAKE_SOURCE_DIR}/test/TestMain.cpp)
else()
    add_executable(halite_test $<TARGET_OBJECTS:halite_core> ${TEST_FILES} ${CMAKE_SOURCE_DIR}/test/TestMain.cpp)
endif()

target_include_directories(halite_test PUBLIC ${HALITE_INCLUDE_DIRS})
add_dependencies(halite_test libzstd_static)
if(APPLE)
    # No static linkage here - https://stackoverflow.com/questions/5259249/creating-static-mac-os-x-build
    target_link_libraries(halite pthread libzstd_static)
    target_link_libraries(halite_test pthread libzstd_static Catch2::Catch)
elseif(MSVC)
    target_link_libraries(halite libzstd_static)
    target_link_libraries(halite_batched_runner libzstd_static)
    target_link_libraries(halite_test libzstd_static Catch2::Catch)
else()
    if(CMAKE_BUILD_TYPE MATCHES Debug AND NOT MINGW)
        # Link dynamically on Linux since AddressSanitizer doesn't work with static linkage
        # But link statically on MinGW, since it doesn't have address sanitizer and keeping MinGW dlls around is a pain
        target_link_libraries(halite pthread libzstd_static)
        target_link_libraries(halite_batched_runner pthread libzstd_static)
        target_link_libraries(halite_test pthread libzstd_static Catch2::Catch)
    else()
        if(HALITE_ENABLE_CUDA)
            # CUDA: link DYNAMICALLY (drop -static). The CUDA 13 static
            # libcudart_static.a requires a newer glibc than Ubuntu 22.04
            # (2.35), so full-static linking makes CUDA init fail and the
            # backend error path segfaults. Dynamic cudart + zstd_static works.
            target_link_libraries(halite pthread libzstd_static CUDA::cudart)
            target_link_libraries(halite_batched_runner pthread libzstd_static CUDA::cudart)
            target_link_libraries(halite_test pthread libzstd_static Catch2::Catch)
        else()
            # On Linux, make sure to link using the whole-archive option
            # See: https://stackoverflow.com/questions/35116327
            target_link_libraries(halite -Wl,--whole-archive pthread -Wl,--no-whole-archive -static libzstd_static)
            target_link_libraries(halite_batched_runner -Wl,--whole-archive pthread -Wl,--no-whole-archive -static libzstd_static)
            target_link_libraries(halite_test -Wl,--whole-archive pthread -Wl,--no-whole-archive -static libzstd_static Catch2::Catch)
        endif()
    endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${EXTERNAL_INSTALL_LOCATION}/catch/contrib)
include(Catch)
enable_testing()
catch_discover_tests(halite_test)

add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
add_dependencies(check halite_test)
