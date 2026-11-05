cmake_minimum_required(VERSION 3.16)

project(tinytapeout_lpu_esl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tinylpu_esl_lib STATIC
  cpp/tinylpu_esl.cpp
)
target_include_directories(tinylpu_esl_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_compile_options(tinylpu_esl_lib PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(tinylpu_esl
  cpp/tinylpu_cli.cpp
)
target_link_libraries(tinylpu_esl PRIVATE tinylpu_esl_lib)

add_executable(test_esl
  tests/test_esl.cpp
)
target_link_libraries(test_esl PRIVATE tinylpu_esl_lib)

enable_testing()
add_test(NAME test_esl COMMAND test_esl)

add_library(tinylpu_dse_lib STATIC
  cpp/tinylpu_dse.cpp
)
set_target_properties(tinylpu_dse_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tinylpu_dse_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_compile_options(tinylpu_dse_lib PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(tinylpu_dse_core
  cpp/tinylpu_dse_cli.cpp
)
target_link_libraries(tinylpu_dse_core PRIVATE tinylpu_dse_lib)

add_executable(test_dse
  tests/test_dse.cpp
)
target_link_libraries(test_dse PRIVATE tinylpu_dse_lib)

add_test(NAME test_dse COMMAND test_dse)

add_library(tinylpu_operators_lib STATIC
  cpp/tinylpu_operators.cpp
)
target_include_directories(tinylpu_operators_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_compile_options(tinylpu_operators_lib PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(tinylpu_operators
  cpp/tinylpu_operators_cli.cpp
)
target_link_libraries(tinylpu_operators PRIVATE tinylpu_operators_lib)

add_executable(test_operators
  tests/test_operators.cpp
)
target_link_libraries(test_operators PRIVATE tinylpu_operators_lib)

add_test(NAME test_operators COMMAND test_operators)

add_library(tinylpu_nonlinear_lib STATIC
  cpp/tinylpu_nonlinear.cpp
)
target_include_directories(tinylpu_nonlinear_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_compile_options(tinylpu_nonlinear_lib PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(tinylpu_nonlinear
  cpp/tinylpu_nonlinear_cli.cpp
)
target_link_libraries(tinylpu_nonlinear PRIVATE tinylpu_nonlinear_lib)

add_executable(test_nonlinear
  tests/test_nonlinear.cpp
)
target_link_libraries(test_nonlinear PRIVATE tinylpu_nonlinear_lib)

add_test(NAME test_nonlinear COMMAND test_nonlinear)

add_library(tinylpu_plan_lib STATIC
  cpp/tinylpu_plan.cpp
)
set_target_properties(tinylpu_plan_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tinylpu_plan_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_link_libraries(tinylpu_plan_lib PUBLIC tinylpu_dse_lib)
target_compile_options(tinylpu_plan_lib PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(tinylpu_plan_sim
  cpp/tinylpu_plan_cli.cpp
)
target_link_libraries(tinylpu_plan_sim PRIVATE tinylpu_plan_lib)

add_executable(test_plan
  tests/test_plan.cpp
)
target_link_libraries(test_plan PRIVATE tinylpu_plan_lib)

add_test(NAME test_plan COMMAND test_plan)

add_library(tinylpu_plan_runtime SHARED
  cpp/tinylpu_plan_c.cpp
)
target_include_directories(tinylpu_plan_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_link_libraries(tinylpu_plan_runtime PRIVATE tinylpu_plan_lib)
target_compile_options(tinylpu_plan_runtime PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(test_plan_c
  tests/test_plan_c.cpp
)
target_include_directories(test_plan_c PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_link_libraries(test_plan_c PRIVATE tinylpu_plan_runtime)

add_test(NAME test_plan_c COMMAND test_plan_c)

add_library(tinylpu_precision SHARED
  cpp/tinylpu_precision.cpp
)
target_include_directories(tinylpu_precision PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_compile_options(tinylpu_precision PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-fno-fast-math;-ffp-contract=off>
)

add_executable(test_precision
  tests/test_precision.cpp
)
target_include_directories(test_precision PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_link_libraries(test_precision PRIVATE tinylpu_precision)

add_test(NAME test_precision COMMAND test_precision)

add_library(tinylpu_isa SHARED
  cpp/tinylpu_isa.cpp
)
target_include_directories(tinylpu_isa PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_compile_options(tinylpu_isa PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(test_isa
  tests/test_isa.cpp
)
target_include_directories(test_isa PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_link_libraries(test_isa PRIVATE tinylpu_isa)

add_test(NAME test_isa COMMAND test_isa)

add_library(tinylpu_architecture SHARED
  cpp/tinylpu_architecture.cpp
)
target_include_directories(tinylpu_architecture PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_compile_options(tinylpu_architecture PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(test_architecture
  tests/test_architecture.cpp
)
target_include_directories(test_architecture PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hpp
)
target_link_libraries(test_architecture PRIVATE tinylpu_architecture)

add_test(NAME test_architecture COMMAND test_architecture)

add_library(tinylpu_device SHARED cpp/tinylpu_device.cpp)
target_include_directories(tinylpu_device PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/hpp)
target_compile_options(tinylpu_device PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
add_executable(test_device tests/test_device.cpp)
target_link_libraries(test_device PRIVATE tinylpu_device)
add_test(NAME test_device COMMAND test_device)
