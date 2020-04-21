cmake_minimum_required(VERSION 3.6)
project(TensorRTClassification)

set(CMAKE_CXX_STANDARD 11)

#SET(ENV tx2)
SET(ENV x86)

IF(ENV STREQUAL "x86")
    SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../release)

    # for trt x86
    SET(CUDA_INCLUDE_DIR /usr/local/cuda/include)
    SET(CUDA_LIB_DIR /usr/local/cuda/lib64)
    SET(CUDA_LIBS cudart)

    SET(CUDNN_INCLUDE_DIR /home/surui/env/envs/TRT5/include)
    SET(CUDNN_LIB_DIR /home/surui/env/envs/TRT5/lib)
    SET(CUDNN_LIBS cudnn)

    SET(TRT_SAMPLES_SRC /home/surui/Downloads/software/TensorRT-5.1.5.0/samples)

    #set(SAMPLES_COMMON_SOURCES ${TRT_SAMPLES_SRC}/common/logger.cpp)  # for log 问题 undefined reference to `gLogger'
    SET(TRT_INCLUDE_DIR /home/surui/Downloads/software/TensorRT-5.1.5.0/targets/x86_64-linux-gnu/include)
    SET(TRT_LIB_DIR /home/surui/Downloads/software/TensorRT-5.1.5.0/lib)
    FILE(GLOB TRT_LIBS "${TRT_LIB_DIR}/*.so")

    # for tf lite x86
    set(TFLITE_LIBS /home/surui/Downloads/software/tensorflow-master/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a
                    /home/surui/Downloads/software/tensorflow-master/tensorflow/lite/tools/make/gen/linux_x86_64/lib/benchmark-lib.a)
    include_directories(/home/surui/Downloads/software/tensorflow-master/)
    include_directories(/home/surui/Downloads/software/tensorflow-master/tensorflow/lite/tools/make/downloads/flatbuffers/include)
    include_directories(/home/surui/Downloads/software/tensorflow-master/tensorflow/lite/tools/make/downloads/absl)


    # for ncnn x86
    SET(NCNN_INCLUDE_DIR /home/surui/Downloads/software/ncnn-master/build/install/include/ncnn)
    SET(NCNN_LIBRARY_DIR /home/surui/Downloads/software/ncnn-master/build/install/lib)


    # for opencv x86
    SET(OPENCV_INCLUDE_DIR /home/surui/Downloads/software/opencv-3.4.0/install/include)
    SET(OPENCV_LIBRARY_DIR /home/surui/Downloads/software/opencv-3.4.0/install/lib)
ELSE()



ENDIF()

SET(OPENCV_LIBS opencv_videostab;opencv_video;opencv_superres;
        opencv_stitching;opencv_photo;opencv_objdetect;
        opencv_ml;opencv_imgproc;opencv_highgui;
        opencv_flann;opencv_features2d;opencv_core;opencv_calib3d;
        opencv_imgcodecs)

find_package(OpenMP)
if(OPENMP_FOUND)
    message("********openmp found")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

INCLUDE_DIRECTORIES(./include ./include/common ${CUDA_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${TRT_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR})
LINK_DIRECTORIES(${CUDA_LIB_DIR} ${CUDNN_LIB_DIR} ${OPENCV_LIBRARY_DIR})

FILE(GLOB SRC "./src/*.cpp")

#ADD_EXECUTABLE(TensorRTClassification ${SRC} sampleMNIST.cpp)
ADD_EXECUTABLE(TensorRTClassification ${SRC} test/testClsEngine.cpp src/ClassifyByEngine.cpp src/MNIST.cpp include/MNIST.h)
TARGET_LINK_LIBRARIES(TensorRTClassification ${CUDA_LIBS} ${TRT_LIBS} ${CUDNN_LIBS} ${OPENCV_LIBS})