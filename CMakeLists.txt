cmake_minimum_required(VERSION 3.0.0)
project(again VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 2.6.x)
FetchContent_MakeAvailable(SFML)

add_executable(again main.cpp model.cpp render.cpp)
target_link_libraries(again PRIVATE sfml-graphics)
target_compile_features(again PRIVATE cxx_std_17)

add_executable(images images.cpp model.cpp render.cpp)
target_link_libraries(images PRIVATE sfml-graphics)
target_compile_features(images PRIVATE cxx_std_17)

install(TARGETS again images)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
