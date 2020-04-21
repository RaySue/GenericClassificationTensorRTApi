//
// Created by surui on 2019/12/4.
//

#include <iostream>
#include <Resnet50Cls.h>
#include <MNIST.h>
#include <dirent.h>

using namespace std;


void getAllFiles(string path, vector<string> &files) {
    DIR *dir;
    struct dirent *ptr;
    if ((dir = opendir(path.c_str())) == NULL) {
        perror("Open dri error...");
        exit(1);
    }
    while ((ptr = readdir(dir)) != NULL) {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        else if (ptr->d_type == 8)//file
        {
            files.push_back(path + "/" + ptr->d_name);
        } else if (ptr->d_type == 10)//link file
            continue;
        else if (ptr->d_type == 4) {
            //files.push_back(ptr->d_name);//dir
            getAllFiles(path + "/" + ptr->d_name, files);
        }
    }
    sort(files.begin(), files.end());
    closedir(dir);
}



bool testCifar10Acc(string className, int predIdx){
    map<string, int> classMap = {
            {"plane", 0}, {"car", 1}, {"bird",2},
            {"cat", 3}, {"deer",4}, {"dog", 5},
            {"frog", 6}, {"horse", 7}, {"ship", 8},
            {"truck", 9}
    };
    assert(classMap.count(className) > 0);

    int trueIdx = classMap[className];

    return trueIdx == predIdx;
}


const string &engineFile = "/home/surui/CLionProjects/TRTBuildEngine/release/resnet50.engine";
const string testImgsDir = "/home/surui/cifar-10-images/batch_test";



int resnet50() {

//    string engineFile = "/home/surui/CLionProjects/TensorRTClassification/release/ResNet50.engine";
    gParams gParams1;
    gParams1.enginePath = engineFile;
    gParams1.batchSize = 1;
    gParams1.resizedHeight = 32;
    gParams1.resizedWidth = 32;
    gParams1.inputC = 3;
    gParams1.classNum = 10;
    Resnet50Cls resnet50Cls(gParams1);
    vector<string> imgPathList;
    getAllFiles(testImgsDir, imgPathList);
    resnet50Cls.createEngine();

    int index = 0;
    float success = 0.0;
    for (int i = 0; i < imgPathList.size(); ++i){
        string imgPath = imgPathList[i];
        string imgName = imgPath.substr(imgPath.find_last_of("/") + 1, -1);
        string className = imgName.substr(imgName.find("_") + 1, imgName.find(".") - imgName.find("_") - 1);

        cv::Mat img = cv::imread(imgPath);
        vector<float> inputData;

        resnet50Cls.preProcessing(img, inputData);

        float result[gParams1.classNum];

        resnet50Cls.doInference(inputData.data(), result);


        //Calculate Softmax
//        float expSum{0.0f};
//        for (int i = 0; i < gParams1.classNum; i++) {
//            expSum += exp(result[i]);
//        }
        float val{0.0f};
        int idx{0};
        float finalProb;
        for (int i = 0; i < gParams1.classNum; i++) {
            float prob = result[i];
            val = std::max(val, prob);
            if (val == prob) {
                finalProb = prob;
                idx = i;
            }
        }

        index++;
        if (testCifar10Acc(className, idx)){
            success++;
        } else {
            gLogInfo << "Output: \n";
            for (int i = 0; i < gParams1.classNum; i++) {
                float prob = result[i];
                val = std::max(val, prob);
                if (val == prob) {
                    finalProb = prob;
                    idx = i;
                }
                gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob << " "
                         << "Class " << i << ": " << std::string(int(std::floor(prob * 10 + 0.5f)), '*') << "\n";
            }
            gLogInfo << "class index : " << idx << " prob : " << finalProb << endl;

        }
        cout << "temp acc : " << success / index << endl;
    }
    cout << "total : " << index <<  " success : " << success << "failed : " << index - success << endl;

}




int mnist() {
    string imgPath = "/home/surui/Downloads/software/TensorRT-5.1.5.0/data/mnist/7.pgm";
    string engineFile = "/home/surui/CLionProjects/TRTBuildEngine/release/caffe2EngineMNIST.engine";

    gParams gParams1;
    gParams1.enginePath = engineFile;
    gParams1.batchSize = 1;
    gParams1.resizedHeight = 28;
    gParams1.resizedWidth = 28;
    gParams1.inputC = 1;
    gParams1.classNum = 10;

    MNIST mnist1(gParams1);

    cv::Mat img = cv::imread(imgPath, 0);
    vector<float> inputData;
//    float* inputdata = (float *)malloc(sizeof(float) * gParams1.resizedWidth * gParams1.resizedHeight * gParams1.inputC);
    mnist1.preProcessing(img, inputData);
    mnist1.createEngine();
    float result[gParams1.classNum];
    mnist1.doInference(inputData.data(), result);

    float val{0.0f};
    int idx{0};

    //Calculate Softmax
    float expSum{0.0f};
    for (int i = 0; i < gParams1.classNum; i++) {
        expSum += exp(result[i]);
    }

    float finalProb;
    gLogInfo << "Output:\n";
    float prob_sum = 0.0;
    for (int i = 0; i < gParams1.classNum; i++) {
        float prob = exp(result[i]) / expSum;
        val = std::max(val, prob);
        if (val == prob) {
            finalProb = prob;
            idx = i;
        }
        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob << " "
                 << "Class " << i << ": " << std::string(int(std::floor(prob * 10 + 0.5f)), '*') << "\n";
        prob_sum += prob;
    }
    gLogInfo << std::endl;

    cout << prob_sum << endl;
    cout << "class index : " << idx << " prob : " << finalProb << endl;

}

int main() {
//    mnist();
    resnet50();
}