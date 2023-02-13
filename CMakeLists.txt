cmake_minimum_required(VERSION 3.24)
project(bvh_mixed_optimizer)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(third_party/bvh)
link_libraries(bvh)
link_libraries(mpfr)

add_executable(bvh_mixed_optimizer main.cpp)
