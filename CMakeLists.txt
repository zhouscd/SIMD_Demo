cmake_minimum_required(VERSION 3.10)

project(SIMD_Demo C)

add_executable(sumfloat sumfloat.c)
add_executable(sumint sumint.c)
add_executable(sumdouble sumdouble.c)

if (WIN32)
target_compile_options(sumfloat PRIVATE " /arch:SSE2")
target_compile_options(sumint PRIVATE " /arch:SSE2")
target_compile_options(sumdouble PRIVATE " /arch:SSE2")
else()
target_compile_options(sumfloat PRIVATE  "-O3 -march=native -mavx")
target_compile_options(sumint PRIVATE  "-O3 -march=native -mavx")
target_compile_options(sumdouble PRIVATE  "-O3 -march=native -mavx")
endif()


