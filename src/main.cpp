#include "inference.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libswscale/swscale.h>
}

int main(int argc, char** argv) {

    const char* path = "../videos/sample.mp4";

    // Open video file
    AVFormatContext* formatContext = nullptr;
    if (avformat_open_input(&formatContext, path, nullptr, nullptr) != 0) {
        fprintf(stderr, "Failed to open video file\n");
        return -1;
    }

    // Read stream information
    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        fprintf(stderr, "Failed to retrieve stream information\n");
        avformat_close_input(&formatContext);
        return -1;
    }

    // Find the video codec and index
    AVCodecParameters* codecParameters = nullptr;
    int videoStreamIndex = -1;
    for (int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStreamIndex = i;
            codecParameters = formatContext->streams[i]->codecpar;
            break;
        }
    }

    // Throw an error if a codec/stream_index is not found
    if (videoStreamIndex == -1) {
        fprintf(stderr, "Failed to find video stream in the input file\n");
        avformat_close_input(&formatContext);
        return -1;
    }

    // Open the decoder based on the codec found above
    AVCodec* codec = avcodec_find_decoder(codecParameters->codec_id);
    if (codec == nullptr) {
        fprintf(stderr, "Failed to find codec\n");
        avformat_close_input(&formatContext);
        return -1;
    }

    AVCodecContext* codecContext = avcodec_alloc_context3(codec);
    if (codecContext == nullptr) {
        fprintf(stderr, "Failed to allocate codec context\n");
        avformat_close_input(&formatContext);
        return -1;
    }

    if (avcodec_parameters_to_context(codecContext, codecParameters) < 0) {
        fprintf(stderr, "Failed to copy codec parameters to codec context\n");
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    // Extract the motion vectors from video
    AVDictionary* opts = NULL;
    av_dict_set(&opts, "flags2", "+export_mvs", 0);
    if (avcodec_open2(codecContext, codec, &opts) < 0) {
        fprintf(stderr, "Failed to open codec\n");
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        av_dict_free(&opts);
        return -1;
    }
    av_dict_free(&opts);

    // Allocate a frame to hold decoded data
    AVFrame* frame = av_frame_alloc();
    if (frame == nullptr) {
        fprintf(stderr, "Failed to allocate frame\n");
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return -1;
    }

    AVStream* videoStream = formatContext->streams[videoStreamIndex];
    double fps = av_q2d(videoStream->r_frame_rate);

    // Reading and processing the frames
    AVPacket packet;
    packet.data = nullptr;
    packet.size = 0;

    int frameCount = 0;
    int object_id = 1;

    std::vector<std::string> class_list = load_class_list();

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    cv::dnn::Net net;
    load_net(net, is_cuda);

    byte_track::BYTETracker tracker(fps, fps);

    std::vector<byte_track::Object> output;

    while (av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == videoStreamIndex) {
            if (avcodec_send_packet(codecContext, &packet) != 0) {
                fprintf(stderr, "Failed to send packet to the codec\n");
                break;
            }
            while (avcodec_receive_frame(codecContext, frame) == 0) {

                SwsContext* swsContext =
                    sws_getContext(codecContext->width, codecContext->height, codecContext->pix_fmt, codecContext->width,
                                   codecContext->height, AV_PIX_FMT_BGR24, 0, NULL, NULL, NULL);

                AVFrame* rgbFrame = av_frame_alloc();
                av_image_alloc(rgbFrame->data, rgbFrame->linesize, codecContext->width, codecContext->height, AV_PIX_FMT_BGR24, 1);
                sws_scale(swsContext, frame->data, frame->linesize, 0, frame->height, rgbFrame->data, rgbFrame->linesize);

                cv::Mat image(codecContext->height, codecContext->width, CV_8UC3, rgbFrame->data[0], rgbFrame->linesize[0]);

                // Check if Key-Frame
                if (frame->key_frame) {
                    output.clear();
                }

                // Yolov8
                detect(image, net, output, class_list, object_id);

                // Byte Tracker
                std::vector<byte_track::BYTETracker::STrackPtr> tracked_outputs = tracker.update(output);

                // Display Byte Tracker results
                for (int i = 0; i < tracked_outputs.size(); ++i) {

                    cv::Rect box(tracked_outputs[i]->getRect().x(), tracked_outputs[i]->getRect().y(),
                                 tracked_outputs[i]->getRect().width(), tracked_outputs[i]->getRect().height());
                    auto classId = tracked_outputs[i]->getLabelId();
                    auto className = tracked_outputs[i]->getClassName();
                    const auto color = colors[classId % colors.size()];
                    cv::rectangle(image, box, color, 3);

                    cv::rectangle(image, cv::Point(box.x, box.y + 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
                    cv::putText(image, className + '(' + std::to_string(classId) + ')', cv::Point(box.x, box.y + 15),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0));
                }

                cv::putText(image, std::to_string(frameCount).c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(0, 0, 255), 2);
                cv::imshow("output", image);

                // Wait for a key press (Wait indefinitely until a key is pressed)
                cv::waitKey(1);

                frameCount++;
            }
        }
    }

    av_packet_unref(&packet);

    av_frame_free(&frame);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);

    return 0;
}
