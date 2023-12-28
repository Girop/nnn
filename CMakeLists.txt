cmake_minimum_required(VERSION 3.12)
project(classifier)

set(CMAKE_CXX_STANDARD 17)
file(GLOB SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)
