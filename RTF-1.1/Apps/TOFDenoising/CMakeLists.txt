include_directories (${PROJECT_SOURCE_DIR})

file(GLOB app_srcs *.cpp)
file(GLOB app_hdrs *.h)

add_executable (ColorDenoising ${app_srcs} ${app_hdrs})

file(TO_NATIVE_PATH "${CMAKE_SOURCE_DIR}/data/ColorDenoising" data_dir)
add_test(NAME DenoisingExample COMMAND ColorDenoising RTFTrainedForPSNR ${data_dir} . 1x1 7)  
