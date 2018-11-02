cmake_minimum_required(VERSION 2.8)

project(uranus)

# TODO: Use VERSION argument to project command after raising the minimum
# cmake version to 3.0
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 1)
set(PROJECT_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

include(CTest)

set(CMAKE_CXX_FLAGS "-std=c++11")

option(URANUS_BUILD_EXAMPLES "Build the Uranus examples" ON)
# option(URANUS_BUILD_TESTING "Build the Uranus Unit Test" ON)

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

find_package(glog REQUIRED) # COMPONENTS nothreads_static)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

set(CURRENT_PROJET_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CURRENT_PROJET_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CURRENT_PROJET_DIR}/build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CURRENT_PROJET_DIR}/build/lib)

include_directories(
    /usr/local/include
    "/usr/include/eigen3"
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Eigen3_INCLUDE_DIR}
)

set(HEADERS
    include/uranus/Matrix.hpp
    include/uranus/Function.hpp
    include/uranus/Tensor.hpp
    include/uranus/log.h
)

# Examples
if(URANUS_BUILD_EXAMPLES)
    add_executable(example_1 "examples/Newton.cc")
    add_executable(example_2 "examples/Fisher.cc")
    add_executable(example_3 "examples/Fisher-sonar.cc")
    add_executable(example_4 "examples/Fisher-Iris.cc")
    add_executable(example_5 "examples/knn-Iris.cc")
    target_link_libraries(example_1 glog)
    target_link_libraries(example_2 glog)
    target_link_libraries(example_3 glog)
    target_link_libraries(example_4 glog)
endif()

# Unit Tests
set(TESTS
    test/function_unittest.cc
    test/matrix_unittest.cc
)

if(URANUS_BUILD_TESTING)
    enable_testing()
        add_executable(uranus_test ${TESTS} ${HEADERS})
        target_link_libraries(uranus_test 
            glog
            ${GTEST_BOTH_LIBRARIES}
            ${CMAKE_THREAD_LIBS_INIT}
            )
    add_test(uranus uranus_test)
endif()
