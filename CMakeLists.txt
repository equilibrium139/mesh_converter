cmake_minimum_required(VERSION 3.17)

project(mesh_converter LANGUAGES CXX)

include(FetchContent)


FetchContent_Declare(
    assimp
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG v5.1.4
)
    
FetchContent_MakeAvailable(assimp)
    
add_executable(mesh_converter src/Main.cpp
                              src/stb_image.h
                              src/stb_image.cpp
                              src/stb_image_write.h
                              src/stb_image_write.cpp
)

target_link_libraries(mesh_converter assimp)

if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "$<TARGET_FILE_DIR:assimp>/$<TARGET_FILE_NAME:assimp>"
                       $<TARGET_FILE_DIR:mesh_converter>
    )
endif()