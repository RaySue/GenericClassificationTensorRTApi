//
// Created by surui on 2019/12/17.
//

#ifndef TENSORRTCLASSIFICATION_RESNET50CLS_H
#define TENSORRTCLASSIFICATION_RESNET50CLS_H

#include <ClassifyByEngine.h>


class Resnet50Cls: public ClassifyByEngine{

public:


    /**
     * Empty Function
     * - inherit from ClassifyByEngine
     * - init parameters:
     * - string engineFile, int batchSize, int inputH, int inputW, int inputC, string inputNodeName="data", string outputNodeName="prob"
     */
    Resnet50Cls(gParams gparams);

    /**
     * Data process
     *  - resize & normalize
     * @param img input
     * @param result output
     */
    void preProcessing(vector<Mat> img, vector<float> &result) override ;


private:
    gParams gparams;

};


#endif //TENSORRTCLASSIFICATION_RESNET50CLS_H
