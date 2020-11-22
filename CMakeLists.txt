cmake_minimum_required(VERSION 3.10)

project(SIMD_Demo C)

if (UNIX)
SET(CMAKE_C_COMPILER "g++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -msse -mavx")
endif()
add_executable(sumfloat sumfloat.c)
add_executable(sumint sumint.c)
add_executable(sumdouble sumdouble.c)

if (WIN32)
target_compile_options(sumfloat PRIVATE " /arch:SSE2")
target_compile_options(sumint PRIVATE " /arch:SSE2")
target_compile_options(sumdouble PRIVATE " /arch:SSE2")
endif()


