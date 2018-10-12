cmake_minimum_required(VERSION 2.8)

project(uranus)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

set(CURRENT_PROJET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CURRENT_PROJET_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CURRENT_PROJET_DIR}/build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CURRENT_PROJET_DIR}/build/lib)

include_directories(
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(uranus
    include/uranus/Matrix.hpp
    include/uranus/Function.hpp
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)

enable_testing()
    add_executable(matrix_unittest 
        test/matrix_unittest.cc
    )
    target_link_libraries(matrix_unittest  ${GTEST_BOTH_LIBRARIES})
    
add_test(uranus matrix_unittest)