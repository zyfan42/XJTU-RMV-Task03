cmake_minimum_required(VERSION 3.0.0)
project(WindMill)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找OpenCV库
find_package(OpenCV REQUIRED)

# 查找Ceres库
find_package(Ceres REQUIRED)

# 包含头文件目录
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

# 查找源文件
aux_source_directory(./src ALL_SRCS)

# 添加可执行文件
add_executable(task main.cpp ${ALL_SRCS})

# 链接库
target_link_libraries(task ${OpenCV_LIBS} ${CERES_LIBRARIES})