#pragma once

#include "ByteTrack/Rect.h"

namespace byte_track {
struct Object {
    Rect<float> rect;
    int label;
    float prob;
    std::string class_name;
    std::unordered_map<std::string, int> classes_score;

    Object(const Rect<float>& _rect, const int& _label, const float& _prob, const std::string& class_name);
};
} // namespace byte_track
