//
// Created by surui on 2019/12/18.
//

#include <ClassifyByEngine.h>

void ClassifyByEngine::setGparams(gParams gParams1){
    gparams.enginePath = gParams1.enginePath;
    gparams.batchSize = gParams1.batchSize;
    gparams.resizedHeight = gParams1.resizedHeight;
    gparams.resizedWidth = gParams1.resizedWidth;
    gparams.inputC = gParams1.inputC;
    gparams.classNum = gParams1.classNum;
}

bool ClassifyByEngine::validateGparams(gParams gparams) {
    bool flag;
    if (gparams.resizedHeight > 0 & gparams.resizedWidth > 0 & gparams.inputC > 0) {
        flag = true;
    } else {
        cout << "[ERROR] You have to assign the resize info and channel !" << endl;
        return false;
    }

    if (gparams.batchSize > 0) {
        flag = true;
    } else {
        cout << "[ERROR] You have to assign the batch size !" << endl;
        return false;
    }

    if (gparams.classNum > 0) {
        flag = true;
    } else {
        cout << "[ERROR] You have to assign the class num !" << endl;
        return false;
    }

    if (!gparams.enginePath.empty()){
        flag = true;
    } else {
        cout << "[ERROR] You have to assign the engine path !" << endl;
    }

    return flag;
}


void ClassifyByEngine::preProcessing(vector<Mat> imgBatch, vector<float> &result){
    result.resize(gparams.batchSize * gparams.resizedWidth * gparams.resizedHeight * gparams.inputC);
    auto data = result.data();

    auto scaleSize = cv::Size(gparams.resizedWidth, gparams.resizedHeight);
    vector<Mat> resizedBatch;
    resizedBatch.reserve(gparams.batchSize);
    for(int i = 0; i < gparams.batchSize; ++i){
        cv::Mat resized;
        cv::Mat img_float;
        cv::resize(imgBatch[i], resized, scaleSize, 0, 0);
        resized.convertTo(img_float, CV_32FC3, 1.);
        resizedBatch.push_back(img_float.clone());
    }

    for (int b = 0; b < gparams.batchSize; ++b){
        cv::Mat img_float = resizedBatch[b];

        // HWC TO CHW
        std::vector<cv::Mat> input_channels(gparams.inputC);
        cv::split(img_float, input_channels);

        // normalize
        int channelLength = gparams.resizedWidth * gparams.resizedHeight;

        for (int i = 0; i < gparams.inputC; ++i) {
            cv::Mat normed_channel;
            // custom
//        if (i == 0) {
//            normed_channel = ((input_channels[i]) / 255.0 - 0.485) / 0.229;
//        } else if (i == 1) {
//            normed_channel = ((input_channels[i]) / 255.0 - 0.456) / 0.224;
//        } else {
//            normed_channel = ((input_channels[i]) / 255.0 - 0.406) / 0.225;
//        }
            normed_channel = input_channels[i];
            memcpy(data, normed_channel.data, channelLength * sizeof(float));
            data += channelLength;
        }
    }

}

int ClassifyByEngine::createEngine() {
    std::vector<char> trtModelStream;
    size_t size{0};
    std::ifstream file(gparams.enginePath, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream.resize(size);
        file.read(trtModelStream.data(), size);
        file.close();
    }
    infer = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    if (gparams.useDLACore >= 0) {
        infer->setDLACore(gparams.useDLACore);
    }
    engine = infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
    gLogInfo << gparams.enginePath << " has been successfully loaded." << std::endl;
    infer->destroy();
}



int ClassifyByEngine::doInference(float *inputData, float *result) {
    context = engine->createExecutionContext();
    assert(engine->getNbBindings() == 2);
    void *buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex{}, outputIndex{};

    if (gparams.inputNodeName.empty() || gparams.outputNodeName.empty()) {
        for (int b = 0; b < engine->getNbBindings(); ++b) {
            if (engine->bindingIsInput(b))
                inputIndex = b;
            else
                outputIndex = b;
        }
    } else {
        inputIndex = engine->getBindingIndex(gparams.inputNodeName.c_str());
        outputIndex = engine->getBindingIndex(gparams.outputNodeName.c_str());
    }

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], gparams.batchSize * gparams.resizedHeight * gparams.resizedWidth * gparams.inputC * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], gparams.batchSize * gparams.classNum * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, gparams.batchSize * gparams.resizedHeight * gparams.resizedWidth * gparams.inputC * sizeof(float),
                          cudaMemcpyHostToDevice, stream));

    context->enqueue(gparams.batchSize, buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(result, buffers[outputIndex], gparams.batchSize * gparams.classNum * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
}