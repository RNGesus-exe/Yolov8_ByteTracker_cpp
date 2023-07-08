cmake_minimum_required(VERSION 3.5)
project(yolo_v5_cpp VERSION 0.1)

# Version of C++ used
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

# OpenCV
find_package(OpenCV 4 REQUIRED)
# !OpenCV

# Include directories
include_directories(include)

# Get all source files in the src directory
set(SOURCES 
    src/inference.cpp
    src/main.cpp
)

set(HEADERS
    include/inference.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
