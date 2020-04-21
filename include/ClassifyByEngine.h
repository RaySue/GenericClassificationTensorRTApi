//
// Created by surui on 2019/12/4.
//

#ifndef TENSORRTCLASSIFICATION_CLASSIFYBYENGINE_H
#define TENSORRTCLASSIFICATION_CLASSIFYBYENGINE_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <common.h>
#include <logger.h>
#include "NvInfer.h"

using namespace cv;
using namespace std;

// TODO : Use DLA to accelerate model
typedef struct Params {
    // required
    string enginePath{};
    int batchSize{0};
    int resizedHeight{0};
    int resizedWidth{0};
    int inputC{0};
    int classNum{0};
    //optional
    string inputNodeName{};
    string outputNodeName{};
    // optional
    int device{0};
    int workspaceSize{16};
    int iterations{10};
    int avgRuns{10};
    //  important
    int useDLACore{-1};
    bool safeMode{false};
    bool fp16{false};
    bool int8{false};
    bool verbose{false};
    bool allowGPUFallback{false};
    float pct{99};
    bool useSpinWait{false};
    bool dumpOutput{false};
    bool help{false};
} gParams;

class ClassifyByEngine {

public:

    /**
     * set gParams to ClassifyByEngine object
     * @param gParams1
     */
    void setGparams(gParams gParams1);

    /**
     * check input gparams
     *
     * @param gparams
     * @return
     */
    bool validateGparams(gParams gparams);

    /**
     * DIY data process method
     *
     * @param img
     * @param result
     */
    virtual void preProcessing(vector<Mat> imgBatch, vector<float> &result) = 0;


    /**
     * build Object -> nvinfer1::ICudaEngine *engine;
     * code 0 -> success
     * code negative num -> failed
     * @return
     */
    int createEngine();


    /**
     * do inference
     * code 0 -> success
     * code negative num -> failed
     * @return
     */
    int doInference(float *inputData, float* result);


private:
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IRuntime* infer;
    nvinfer1::IExecutionContext* context;
    gParams gparams;

};


#endif //TENSORRTCLASSIFICATION_CLASSIFYBYENGINE_H
