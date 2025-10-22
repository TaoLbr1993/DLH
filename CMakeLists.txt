cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    # libfaiss 安装路径
    list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    # hnsw-flex
    # add_executable(ghnsw_example_filter examples/cpp/ghnsw_example_filter.cpp)
    # target_link_libraries(ghnsw_example_filter hnswlib)
    
    # # add_executable(ghnsw_addm examples/cpp/ghnsw_addm.cpp)
    # # target_link_libraries(ghnsw_addm hnswlib)

    # add_executable(ghnsw_hnt examples/cpp/ghnsw_hopnbrtri.cpp)
    # target_link_libraries(ghnsw_hnt hnswlib)

    # add_executable(ghnsw_psl examples/cpp/ghnsw_psl.cpp)
    # target_link_libraries(ghnsw_psl hnswlib)

    find_package(faiss REQUIRED)
    add_executable(ghnsw_test examples/cpp/ghnsw_test.cpp)
    target_link_libraries(ghnsw_test hnswlib faiss)

    # # 仅对 g++ 构建的 ghnsw_psl 开启调试信息并关闭优化
    # if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    #   target_compile_options(ghnsw_psl PRIVATE -g -ggdb3 -O0 -fno-omit-frame-pointer)
    #   target_link_options(ghnsw_psl PRIVATE -Wl,--build-id)
    # endif()

    # add_executable(ghnsw_pll examples/cpp/ghnsw_pll.cpp)
    # target_link_libraries(ghnsw_pll hnswlib)

    # add_executable(ghnsw_breaker examples/cpp/ghnsw_breaker.cpp)
    # target_link_libraries(ghnsw_breaker hnswlib)

    # add_executable(phnsw_example_filter examples/cpp/phnsw_example_filter.cpp)
    # target_link_libraries(phnsw_example_filter hnswlib)

    # add_executable(ghnsw_gdist examples/cpp/ghnsw_gdist.cpp)
    # target_link_libraries(ghnsw_gdist hnswlib)
endif()