cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME ProcessImage)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES process_image.cpp)
set(TARGET ${PROJECT_NAME})

project(${PROJECT_NAME})

find_package( OpenCV CONFIG REQUIRED )
if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
endif(OpenCV_FOUND)

add_executable(${TARGET} ${SOURCES})

target_include_directories(${TARGET} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${TARGET} PUBLIC ${OpenCV_LIBS})