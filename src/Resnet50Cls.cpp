//
// Created by surui on 2019/12/17.
//

#include "Resnet50Cls.h"

Resnet50Cls::Resnet50Cls(gParams gparams) :
gparams(gparams)
{
    setGparams(gparams);
    if (!validateGparams(gparams))
    {
        cout << "[ERROR] Check your gparams !" << endl;
    };

}


void Resnet50Cls::preProcessing(vector<Mat> imgBatch, vector<float> &result){
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
            normed_channel = (input_channels[i] - 128.0);
            memcpy(data, normed_channel.data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
}

