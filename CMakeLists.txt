cmake_minimum_required(VERSION 3.5)
project(yolo_v5_cpp VERSION 2.0)

# Version of C++ used
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

# OpenCV
find_package(OpenCV 4 REQUIRED)
# !OpenCV

# Eigen
find_package(Eigen3 REQUIRED)
# !Eigen

# FFMpeg
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_library(SWSCALE_LIBRARY swscale)
# !FFMpeg

# Include directories
include_directories(include)

# Get all source files in the src directory
set(SOURCES 
    src/inference.cpp
    src/main.cpp
	src/BYTETracker.cpp
	src/KalmanFilter.cpp
	src/lapjv.cpp
	src/Object.cpp
	src/Rect.cpp
	src/STrack.cpp
)

set(HEADERS
    include/inference.h
	include/ByteTrack/BYTETracker.h
    include/ByteTrack/KalmanFilter.h
	include/ByteTrack/lapjv.h
	include/ByteTrack/Object.h
	include/ByteTrack/Rect.h
	include/ByteTrack/STrack.h
)

include_directories(include ${EIGEN3_INCLUDE_DIR})
link_directories(${CMAKE_SOURCE_DIR}/libs)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE include ${SWSCALE_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${SWSCALE_LIBRARY} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${OpenCV_LIBS})
