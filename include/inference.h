#include "ByteTrack/BYTETracker.h"
#include <fstream>
#include <opencv4/opencv2/opencv.hpp>

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
                                        cv::Scalar(255, 0, 0)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 480.0;
const float SCORE_THRESHOLD = CONFIDENCE_THRESHOLD;
const float NMS_THRESHOLD = 0.4;

std::vector<std::string> load_class_list();

void load_net(cv::dnn::Net& net, bool is_cuda);

cv::Mat format_yolov5(const cv::Mat& source);

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<byte_track::Object>& output, const std::vector<std::string>& className,
            int& object_id);
