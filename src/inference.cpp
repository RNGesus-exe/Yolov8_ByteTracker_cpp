#include "inference.h"

std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("../yolo_v5_model/classes.txt");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net& net, bool is_cuda) {
    auto result = cv::dnn::readNet("../models/yolov8s.onnx");
    if (is_cuda) {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

float calculateIOU(byte_track::Object& rect1, byte_track::Object& rect2) {
    int x1 = std::max(rect1.rect.x(), rect2.rect.x());
    int y1 = std::max(rect1.rect.y(), rect2.rect.y());
    int x2 = std::min(rect1.rect.x() + rect1.rect.width(), rect2.rect.x() + rect2.rect.width());
    int y2 = std::min(rect1.rect.y() + rect1.rect.height(), rect2.rect.y() + rect2.rect.height());

    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }

    float intersection_area = static_cast<float>((x2 - x1) * (y2 - y1));
    float union_area =
        static_cast<float>(rect1.rect.width() * rect1.rect.height() + rect2.rect.width() * rect2.rect.height() - intersection_area);

    return intersection_area / union_area;
}

int checkOverlapping(byte_track::Object& obj_a, byte_track::Object& obj_b) {
    float iou = calculateIOU(obj_a, obj_b);

    // std::cout << obj_a.class_name << " " << obj_a.label << " " << obj_b.class_name << " " << obj_b.label << " " << iou << std::endl;

    if (obj_a.class_name == obj_b.class_name && iou >= 0.1f) {
        return 1;
    }

    // To Adjust
    return (iou >= 0.5f) ? 2 : 0;
};

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<byte_track::Object>& output, const std::vector<std::string>& className,
            int& object_id) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    if (dimensions > rows) {
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }

    float* data = (float*)outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        float* classes_scores = data + 4;
        cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
        if (max_class_score > SCORE_THRESHOLD) {

            confidences.push_back(max_class_score);

            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            // This model returns -ve values
            if (left < 0) {
                left = 0;
            }
            if (top < 0) {
                top = 0;
            }
            if (left + width >= image.cols) {
                width = image.cols - left - 1;
            }
            if (top + height >= image.rows) {
                height = image.rows - top - 1;
            }

            boxes.push_back(cv::Rect(left, top, width, height));
        }

        data += dimensions;
    }

    std::vector<byte_track::Object> new_outputs;
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        byte_track::Object result(byte_track::Rect<float>(boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height), i,
                                  confidences[idx], className[class_ids[idx]]);
        result.label = 0; // No unique id provided yet
        new_outputs.push_back(result);
    }

    // for (int i = 0; i < new_outputs.size(); i++) {
    //     printf("ID: %d, Label: %s, New: %.2f %.2f %.2f %.2f conf = %f\n", new_outputs[i].label, new_outputs[i].class_name.c_str(),
    //            new_outputs[i].rect.x(), new_outputs[i].rect.y(), new_outputs[i].rect.width(), new_outputs[i].rect.height(),
    //            new_outputs[i].prob);
    // }
    // printf("\n");
    // for (int i = 0; i < output.size(); i++) {
    //    printf("ID: %d, Label: %s, Old: %.2f %.2f %.2f %.2f conf = %f\n", output[i].label, output[i].class_name.c_str(),
    //           output[i].rect.x(), output[i].rect.y(), output[i].rect.width(), output[i].rect.height(), output[i].prob);
    // }
    // printf("\n\n");

    // Remove Overlapping Objects
    for (int i = 0; i < new_outputs.size(); i++) {
        for (int j = 0; j < output.size(); j++) {
            int ret = checkOverlapping(new_outputs[i], output[j]);
            if (ret == 1) {
                new_outputs[i].label = output[j].label;
                output.erase(output.begin() + j);
                j--;
            }
        }
    }

    for (int i = 0; i < new_outputs.size(); i++) {
        if (new_outputs[i].label == 0) {
            new_outputs[i].label = object_id++;
        }
        output.push_back(new_outputs[i]);
    }
}
