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

    # 添加 ACORN
    # ACORN 安装路径
    set(ACORN_FAISS_DIR "$ENV{HOME}/.local")
    # 使用 ACORN 的 Faiss（用 find_package，导入的目标名通常是 faiss 或 faiss::faiss）
    find_package(faiss REQUIRED PATHS "${ACORN_FAISS_DIR}" NO_DEFAULT_PATH)

    # add_executable(ghnsw_test examples/cpp/ghnsw_test.cpp)
    # target_link_libraries(ghnsw_test hnswlib faiss)
    add_executable(acorn_test examples/cpp/acorn_test.cpp)
    target_link_libraries(acorn_test hnswlib faiss)

    # 添加 Navix
    set(NAVIX_BASE_DIRS "$ENV{HOME}/faiss-navix")
    add_library(navix STATIC IMPORTED)
    set_target_properties(navix PROPERTIES
    IMPORTED_LOCATION ${NAVIX_BASE_DIRS}/build/faiss/libfaiss.a
    INTERFACE_INCLUDE_DIRECTORIES ${NAVIX_BASE_DIRS})

    find_package(BLAS REQUIRED)
    set(BLAS_LIB_TARGET BLAS::BLAS)

    add_executable(navix_test examples/cpp/navix_test.cpp)
    target_link_libraries(navix_test hnswlib navix ${BLAS_LIB_TARGET})


    # 生成 groundtruth
    add_executable(gen_groundtruth data/gen_groundtruth.cpp)
    target_link_libraries(gen_groundtruth hnswlib)

    # baseline

    # hnsw
    add_executable(baseline_hnsw data/baseline/baseline_hnsw.cpp)
    target_link_libraries(baseline_hnsw hnswlib)

    # navix
    add_executable(baseline_navix data/baseline/baseline_navix.cpp)
    target_link_libraries(baseline_navix hnswlib navix ${BLAS_LIB_TARGET})

    # acorn
    add_executable(baseline_acorn data/baseline/baseline_acorn.cpp)
    target_link_libraries(baseline_acorn hnswlib faiss)

    # ghnsw v5
    add_executable(baseline_ghnsw_v5 data/baseline/baseline_ghnsw_v5.cpp)
    target_link_libraries(baseline_ghnsw_v5 hnswlib)

    # ghnsw v6
    add_executable(baseline_ghnsw_v6 data/baseline/baseline_ghnsw_v6.cpp)
    target_link_libraries(baseline_ghnsw_v6 hnswlib)

    # bf
    add_executable(baseline_bf data/baseline/baseline_bf.cpp)
    target_link_libraries(baseline_bf hnswlib)


    # # ghnsw_psl
    # add_executable(ghnsw_psl examples/cpp/ghnsw_psl.cpp)
    # target_link_libraries(ghnsw_psl hnswlib)

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