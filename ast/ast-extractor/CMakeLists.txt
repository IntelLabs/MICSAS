cmake_minimum_required(VERSION 3.4.3)
project(ast-extractor)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tree-sitter/tree-sitter)
    execute_process(
        COMMAND git clone https://github.com/tree-sitter/tree-sitter.git ${PROJECT_SOURCE_DIR}/tree-sitter/tree-sitter
    )
endif()

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tree-sitter/tree-sitter-c)
    execute_process(
        COMMAND git clone https://github.com/tree-sitter/tree-sitter-c.git ${PROJECT_SOURCE_DIR}/tree-sitter/tree-sitter-c
    )
endif()

if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tree-sitter/tree-sitter-cpp)
    execute_process(
        COMMAND git clone https://github.com/tree-sitter/tree-sitter-cpp.git ${PROJECT_SOURCE_DIR}/tree-sitter/tree-sitter-cpp
    )
endif()

get_filename_component(TREE_SITTER_INCLUDE tree-sitter/tree-sitter/lib/include ABSOLUTE)

add_subdirectory(tree-sitter)
add_subdirectory(src)
