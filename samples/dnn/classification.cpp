#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "common.hpp"
#include "windows.h"

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes. }"
    "{ backend     | 0 | Choose one of computation backends: "
                        "0: automatically (by default), "
                        "1: Halide language (http://halide-lang.org/), "
                        "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                        "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
                        "0: CPU target (by default), "
                        "1: OpenCL, "
                        "2: OpenCL fp16 (half-float precision), "
                        "3: VPU }"
    "{ batch_size      | 1 | batch size to test}";

//using namespace cv;
using namespace cv::dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    const std::string modelName = parser.get<cv::String>("@alias");
    const std::string zooFile = parser.get<cv::String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = cv::CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run classification deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    float scale = parser.get<float>("scale");
    cv::Scalar mean = parser.get<cv::Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");
    cv::String model = findFile(parser.get<cv::String>("model"));
    cv::String config = findFile(parser.get<cv::String>("config"));
    cv::String framework = parser.get<cv::String>("framework");
    int backendId = parser.get<int>("backend");
    int targetId = parser.get<int>("target");
    int batch_size = parser.get<int>("batch_size");

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<cv::String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(cv::Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    CV_Assert(!model.empty());

    //! [Read and initialize network]
    Net net = readNet(model, config, framework);
    backendId = cv::dnn::DNN_BACKEND_CUDA;
    targetId = cv::dnn::DNN_TARGET_CUDA;
    net.setPreferableBackend(backendId);
    net.setPreferableTarget(targetId);
    //! [Read and initialize network]

    // Create a window
    static const std::string kWinName = "Deep learning image classification in OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

    //! [Open a video file or an image file or a camera stream]
    cv::VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<cv::String>("input"));
    else
        cap.open(0);
    //! [Open a video file or an image file or a camera stream]

    // Process frames.
    cv::Mat frame, blob;
    while (cv::waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        //! [Create a 4D blob from a frame]
        //blobFromImage(frame, blob, scale, cv::Size(inpWidth, inpHeight), mean, swapRB, false);
        std::vector<cv::Mat> imgs;
        //if (batch_size == 0)
        //    batch_size = 1;
        for (int i = 0; i < batch_size; i++)
            imgs.push_back(frame);
        //imgs.push_back(frame);
        //imgs.push_back(frame);
        //imgs.push_back(frame);
        blobFromImages(imgs, blob, scale, cv::Size(inpWidth, inpHeight), mean, swapRB, false);
        //! [Create a 4D blob from a frame]

        //! [Set input blob]
        net.setInput(blob);
        //! [Set input blob]
        //! [Make forward pass]
        cv::Mat prob = net.forward("fc8_fine");
        prob = net.forward("fc8_fine");

        const int nReps = 100;
        LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
        LARGE_INTEGER Frequency;

        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
        //prob = net.forward();
        for(int i = 0 ; i < nReps; i++)
            prob = net.forward("fc8_fine");
        QueryPerformanceCounter(&EndingTime);
        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
        ElapsedMicroseconds.QuadPart *= 1000000;
        ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
        //! [Make forward pass]

        //! [Get a class with a highest score]
        cv::Point classIdPoint;
        double confidence;
        minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;
        //! [Get a class with a highest score]

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time: %.2f ms", t);
        putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        label = cv::format("bs: %i, Time/image: %.2f ms", batch_size, ElapsedMicroseconds.QuadPart / (batch_size * nReps * 1000.0));
        putText(frame, label, cv::Point(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        // Print predicted class.
        label = cv::format("%s: %.4f", (classes.empty() ? cv::format("Class #%d", classId).c_str() :
                                                      classes[classId].c_str()),
                                   confidence);
        putText(frame, label, cv::Point(0, 65), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

        imshow(kWinName, frame);
    }
    return 0;
}
