#include "ByteTrack/Object.h"

byte_track::Object::Object(const Rect<float>& _rect, const int& _label, const float& _prob, const std::string& _class_name)
    : rect(_rect)
    , label(_label)
    , prob(_prob)
    , class_name(_class_name) {}
