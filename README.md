# GenericClassificationTensorRTApi

> A generic classification using TensorRT C++ Api

## Environment

- opencv 3.4.0

- CUDA 10.1

- cudnn 7.5

- TensorRT 5.1.5.0 x86
 
- TensorRT 5.1.6.0 tx2


## Usage

If you have a new classify task, you just inherit the ClassifyByEngine, and override the function preProcessing follow your train step.

