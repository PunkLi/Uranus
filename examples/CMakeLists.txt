cmake_minimum_required(VERSION 2.8)

project(examples)

include_directories(
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CURRENT_PROJET_DIR}/include
)

add_executable(examples
    main.cc
)