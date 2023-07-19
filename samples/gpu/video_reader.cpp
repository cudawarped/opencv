
#include "opencv2/opencv_modules.hpp"

#if defined(HAVE_OPENCV_CUDACODEC)

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;
using namespace cv::cudacodec;

string GetCodecString(cudacodec::Codec codec) {
    switch (codec) {
    case cudacodec::MPEG1:return "MPEG-1";
    case cudacodec::MPEG2: return "MPEG-2";
    case cudacodec::MPEG4: return "MPEG-4 (ASP)";
    case cudacodec::VC1: return "VC-1/WMV";
    case cudacodec::H264: return "AVC/H.264";
    case cudacodec::JPEG: return "M-JPEG";
    case cudacodec::H264_SVC: return "H.264/SVC";
    case cudacodec::H264_MVC: return "H.264/MVC";
    case cudacodec::HEVC: return "H.265/HEVC";
    case cudacodec::VP8: return "VP8";
    case cudacodec::VP9: return "VP9";
    case cudacodec::AV1: return "AV1";
    case cudacodec::NumCodecs: return "Invalid";
    case cudacodec::Uncompressed_YUV420: return "YUV  4:2:0";
    case cudacodec::Uncompressed_YV12: return "YV12 4:2:0";
    case cudacodec::Uncompressed_NV12: return "NV12 4:2:0";
    case cudacodec::Uncompressed_YUYV: return "YUYV 4:2:2";
    case cudacodec::Uncompressed_UYVY: return "UYVY 4:2:2";
    default: return "Unknown";
    }
}

string GetChromaString(cudacodec::ChromaFormat chromaFormat) {
    switch (chromaFormat) {
    case cudacodec::Monochrome:return "YUV 400 (Monochrome)";
    case cudacodec::YUV420: return "YUV 420";
    case cudacodec::YUV422: return "YUV 422";
    case cudacodec::YUV444: return "YUV 444";
    case cudacodec::NumFormats: return "Invalid";
    default: return "Unknown";
    }
}

string GetDeinterlaceString(cudacodec::DeinterlaceMode deinterlaceMode) {
    switch (deinterlaceMode) {
    case cudacodec::Weave: return "Weave (no deinterlacing)";
    case cudacodec::Bob: return "Bob";
    case cudacodec::Adaptive: return "Adaptive deinterlacing";
    default: return "Unknown";
    }
}

// options
// output all info regarding video file
// display output
// benchmark - cpu vs gpu runs for max 1000 frames
// write raw
// rtsp - auto drop
// rtsp - udp source
// delay when rtsp
const String keys =
{
    "{help h usage ? |      | print this message   }"
    "{input i        | | path to file or uri of video source to decode }"
    "{n        | 0 | maximum number of frames to decode  }"
    "{display d        | true | display the decoded video }"
    "{fps           | 25 | approximate (using cv::waitKey(1000/fps)), fps to play the video if -d=true, if not specified the source files fps is used }"
    "{output o | | path to output file for saving raw encoded video, useful for archiving footage from live video sources }"
    "{benchmark b | 0 | benchmark gpu vs cpu decoding performance, try hardware decoder if available, check this measure decoding .. approximate, creation of Video Decoder starts the decoding process}"
    "{benchmark_hw bhw | 0 | benchmark VideoReader vs VideoCapture hardware accelerated video decoding.}"
    "{device dev | 0 | id of device to use}"
    "{color_format c | 1 | ColorFormat of decoded frame, values are 1: bgra, 2:bgr, 3:gray, 4: NV12 could be gray nv12, bgr, bgra}"
    "{allow_frame_drop afd | 0 | Allow frames to be dropped when ingesting from a live capture source to prevent delay and eventual disconnection when playing at a slower rate than the source's fps.  Useful in combination with -display=1 when pausing and resuming the video to inspect the output.}"
    "{udp_source udp | 0 | Remove limit on the number of frames which can be read before decoding commences. Can be useful when streaming with udp where packets may be lost or from a live video source which has already started streaming.}"
    "{target_width tw | 0 | Output width, should be multiples of 2, defaults to width of source. See}"
    "{target_height th | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    "{src_roi_x srx | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    "{src_roi_y sry | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    "{src_roi_width srw | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    "{src_roi_height srh | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    "{target_roi_x trx | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    "{target_roi_y try | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    "{target_roi_width trw | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    "{target_roi_height trh | 0 | Output height, should be multiples of 2, defaults to height of source.}"
    //"{target_height h | 0 | Output height, defaults to height of source.}"
    //"{target_height h | 0 | Output height, defaults to height of source.}"
};

//
// frame drop
// udp source
// cropping etc
// fps to play video may be off

// include check to see if retrieval of histogram is enabled <- could also display this?

//! [main]
int main(int argc, const char* argv[])
{
    // doesn't throw error if input is empty?

    CommandLineParser parser(argc, argv, keys);
    parser.about("cv::cudacodec::VideoReader Sample Application, requires CAP_FFMPEG and a supported Nvidia GPU");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // text about uses nvidia hardware decdoding chip and FFmpeg to demux the video
    const String input = parser.get<String>("input");
    int n = parser.get<int>("n");
    bool display = parser.get<bool>("display");
    const float fps = parser.get<float>("fps");
    const String output = parser.get<String>("output");
    bool benchmark = parser.get<bool>("benchmark");
    const bool benchmark_hw = parser.get<bool>("benchmark_hw");
    const int deviceId = parser.get<int>("device");
    const ColorFormat colorFormat = parser.get<ColorFormat>("color_format");
    const bool frameDrop = parser.get<bool>("allow_frame_drop");
    const bool udpSource = parser.get<bool>("udp_source");
    const Size targetSz = { parser.get<int>("target_width"), parser.get<int>("target_height") };
    const Rect srcRoi = { parser.get<int>("src_roi_x"), parser.get<int>("src_roi_y"), parser.get<int>("src_roi_width"), parser.get<int>("src_roi_height") };
    const Rect targetRoi = { parser.get<int>("target_roi_x"), parser.get<int>("target_roi_y"), parser.get<int>("target_roi_width"), parser.get<int>("target_roi_height") };

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    // need to print selected options - fps, output, bench, n, input
    if (benchmark_hw) benchmark = true;
    if (benchmark) {
        cout << "Display disabled due to benchmarking" << endl;
        display = false;
        // need message informing options disabled or return false
        //output = false;

    }

    string colorFormatString = colorFormat == ColorFormat::BGRA ? "BGRA" : colorFormat == ColorFormat::BGR ? "BGR" : colorFormat == ColorFormat::GRAY ? "Gray" : "NV12";


    // request single thread

    // need to get device
    printShortCudaDeviceInfo(0);

    //cout << "Decoding: " << input << endl;
    if (display) {
        cout << "Displaying output @" << fps << "fps" << endl;
    }

    // Time initialization
    //! [initialization]
    cv::TickMeter tm;
    tm.start();
    setDevice(deviceId);
    tm.stop();
    cout << "Initialization Time" << endl;
    cout << "   CUDA context : " << std::fixed << std::setprecision(2) <<  tm.getTimeMilli() << "ms" << endl;
    // context initialization
    //! [initialization]



    Ptr<VideoReader> reader;
    try {

        // we can now get all the properties from FFMpeg
        // CAP_PROP_STREAM_OPEN_TIME_USEC
        vector<int> videoCaptureParams = { CAP_PROP_N_THREADS, 1 };  // prevent launching pool of threads, FFmpeg is only used for parsing
        VideoReaderInitParams params;
        params.udpSource = udpSource;
        params.allowFrameDrop = frameDrop;
        params.targetSz = targetSz;
        params.srcRoi = srcRoi;
        params.targetRoi = targetRoi;
        if (!output.empty()) params.rawMode = true;
        if (benchmark) params.minNumDecodeSurfaces = 20;
        tm.reset();
        tm.start();
        reader = createVideoReader(input, {}, params);
        tm.stop();
        reader->set(colorFormat);
        cout << "   VideoReader:     " << tm.getTimeMilli() << "ms" << endl << endl;
    }
    catch (const cv::Exception& e) {
        // extra output when we can check if decoder works or not and maybe histogram as well
        cout << "Failed to initialize cudacodec::VideoReader with source == " << input << endl;
        cout << e.msg;
        return -1;
    }

    std::ofstream file;
    //size_t rawIdxBase = 0;
    double rawIdxBase = 0;
    if (!output.empty()) {
        if (!reader->get(VideoReaderProps::PROP_RAW_PACKAGES_BASE_INDEX, rawIdxBase)) {
            cout << "Failed to get the starting index for raw video packets." << endl;
            return -1;
        }

        file.open(output, std::ios::binary);
        if (!file.is_open()) {
            cout << "Failed to open " << output << " for writing raw video." << endl;
            return -1;
        }
    }

    //// if you didn't initialize .. with ... then the stream will have been started when you constructed VideoReader and the initial packet containing the key frame discarded.

    //// Example - example illustrating the use of `rawFrameHasKeyFrame` to search for a key frame if raw mode is enabled after the VideoReader class has been initialized
    //// if VideoReader is initialized without enabling raw mode, e.g.
    //VideoReaderInitParams params;
    //params.rawMode = false;
    //Ptr<VideoReader> reader = createVideoReader(input, {}, params);
    //// then later on raw mode is enabled, the first package retrieved may not contain a key frame.  If the footage is to be archived from a key frame the search for the first package to write could be performed as follow.
    //reader->set(VideoReaderProps::PROP_RAW_MODE, true);
    //GpuMat frame;
    //double iFirstRawPacket = -1;
    //reader->get(VideoReaderProps::PROP_RAW_PACKAGES_BASE_INDEX, iFirstRawPacket);
    //while (reader->nextFrame(frame)) {
    //    double nRawPackets = -1;
    //    reader->get(VideoReaderProps::PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB, nRawPackets);
    //    for (int iRawPacketToWrite = static_cast<int>(iFirstRawPacket); iRawPacketToWrite < static_cast<int>(iFirstRawPacket + nRawPackets); iRawPacketToWrite++) {
    //        if (reader->rawFrameHasKeyFrame(iRawPacketToWrite)) {
    //            Mat packageToWrite;
    //            reader->retrieve(packageToWrite, iRawPacketToWrite);
    //            //...
    //        }
    //    }
    //}

    FormatInfo fmt = reader->format();
    const float fpsPlay = fps == 0 ? fmt.fps : fps;
    cout << "Video Input Information :" << endl;
    cout << "    Path : " << input << endl;
    cout << "    Codec          : " << GetCodecString(fmt.codec) << endl;
    double fpsCap = 0;
    reader->get(CAP_PROP_FPS, fpsCap);
    cout << "    Frame rate     : " << (fpsCap ? fpsCap : fmt.fps) << endl;
    cout << "    Coded size     : [" << fmt.ulWidth << ", " << fmt.ulHeight << "]" << endl;
    cout << "    Display area   : [" << fmt.displayArea.x << ", " << fmt.displayArea.y << ", " << fmt.displayArea.width << ", " << fmt.displayArea.height << "]" << endl;
    cout << "    Chroma         : " << GetChromaString(fmt.chromaFormat) << endl;
    cout << "    Bit depth      : " << fmt.nBitDepthMinus8 + 8 << endl;
    cout << "    Full color range: " << (fmt.videoFullRangeFlag ? "YES" : "NO") << endl;
    cout << "    Deinterlace Mode: " << GetDeinterlaceString(fmt.deinterlaceMode) << endl;
    if (display)
        cout << "    Frame rate play : " << fpsPlay << endl;
    cout << endl << "Video Decoding Params :" << endl;
    cout << "    Num Surfaces   : " << fmt.ulNumDecodeSurfaces <<endl;
    cout << endl << "Output :" << endl;
    if (file.is_open())
        cout << "    Destingation : " << output << endl;
    cout << "    Color format : " << colorFormatString << endl;

    // only output if there are any
    cout << endl << "Streaming Options :" << endl;
    if (file.is_open())
        cout << "    Writing raw encoded video to " << output << endl;
    if(udpSource)
        cout << "    Allow frame drops   : " << udpSource  << endl;
    if(frameDrop)
        cout << "    UDP source   : " << frameDrop << endl;
    // output size

    //    Codec        : AVC / H.264
    //    Frame rate : 24 / 1 = 24 fps
    //    Sequence : Progressive
    //    Coded size : [672, 384]
    //    Display area : [0, 0, 672, 384]
    //    Chroma : YUV 420
    //    Bit depth : 8
    //    Video Decoding Params :
    //Num Surfaces : 5
    //    Crop : [0, 0, 0, 0]
    //    Resize : 672x384
    //    Deinterlace : Weave
    //if (argc != 2)
        //return -1;

    //const std::string fname(argv[1]);

    //cv::namedWindow("CPU", cv::WINDOW_NORMAL);
    string winName = "GPU (" + colorFormatString + ")";
    const int windowFlags = targetSz.empty() ? cv::WINDOW_NORMAL : cv::WINDOW_AUTOSIZE;
    if (display) {
#if defined(HAVE_OPENGL)
        winName += " displayed from device";
        cv::namedWindow(winName, cv::WINDOW_OPENGL | windowFlags);
        cv::cuda::setGlDevice();
#else
        winName += " displayed from host";
        cv::namedWindow(winName, windowFlags);
#endif
        cout << "Diplaying decoded frames, press q to quit, p to pause and r to resume." << endl;
    }

    Stream stream;
    GpuMat frameDevice;
    Mat frameHost;
    const int displayTimeFps = 1000.0f / fpsPlay;
    int nFrames = 0, displayTime = displayTimeFps;
    //cv::TickMeter tm;
    if (benchmark) {
        tm.reset();
        tm.start();
    }
    double gpuDecodingTimeMs = 0;
    while (reader->nextFrame(frameDevice, stream)) {

        if (display) {
#if defined(HAVE_OPENGL)
            cv::imshow(winName, cv::ogl::Texture2D(frameDevice));
#else
            frameDevice.download(frameHost);
            cv::imshow(winName, frameHost);
#endif
            char c = waitKey(displayTime);
            if (c == 'p')
                displayTime = 0;
            else if (c == 'r')
                displayTime = displayTimeFps;
            else if (c == 'q')
                break;
        }

        if (file.is_open()) {
            //size_t N = 0;
            double N = 0;
            if (!reader->get(VideoReaderProps::PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB, N)) {
                cout << "Failed to get the number of raw packets." << endl;
                return -1;
            }
            for (size_t i = rawIdxBase; i < static_cast<size_t>(N + rawIdxBase); i++) {
                cv::Mat packageToWrite;
                if (!reader->retrieve(packageToWrite, i)) {
                    cout << "Failed to retrieve raw packet: " << i << endl;
                    return -1;
                }
                file.write((char*)packageToWrite.data, packageToWrite.total());
            }
        }

        nFrames++;
        if (n != 0 &&  nFrames >= n)
            break;
    }
    if (benchmark) {
        tm.stop();
        gpuDecodingTimeMs = tm.getTimeMilli();
        cout << endl << "Decoding Performance" << endl;
        cout << "   cudacodec::VideReader" << endl;
        cout << "       Total Frames Decoded: " << nFrames << "@" << nFrames / tm.getTimeSec() << "fps" << endl;
        cout << "       Decoding time: " << endl;
        cout << "           Total: " << tm.getTimeMilli() << "ms" << endl;
        cout << "           Average: " << tm.getTimeMilli()/nFrames << "ms" << endl;
        cout << "           Decoding FPS: " << nFrames/tm.getTimeSec() << endl;
    }

    if (file.is_open())
        file.close();

    if (benchmark) {
        Mat frame;
        VideoCapture cap;
        int hwDevice = -1;
        if (benchmark_hw) {
            cap.open(input, CAP_FFMPEG, { CAP_PROP_HW_ACCELERATION, VIDEO_ACCELERATION_ANY });
            const bool hwAccel = static_cast<bool>(cap.get(CAP_PROP_HW_ACCELERATION));
            if (!hwAccel) {
                cout << "Benchmark Failed - unable to open cv::VideoCapture with hardware-accelerated video decoding!";
                return -1;
            }
            hwDevice = static_cast<int>(cap.get(CAP_PROP_HW_DEVICE));
        }
        else
            cap.open(input);
        if (!cap.isOpened()) {
            cout << "Benchmark Failed - unable to open cv::VideoCapture!" << endl;
            return -1;
        }
        // get backend
        //cap.getBackendName();

        int nHostFrames = 0;
        tm.reset();
        tm.start();
        while (cap.read(frame))
            nHostFrames++;
        tm.stop();
        cout << "   cv::VideoCapture with CAP_FFMPEG";
        if (benchmark_hw) cout << ", hardware acceleration device: " << hwDevice << endl;
        else cout << endl;
        cout << "       Total Frames Decoded: " << nHostFrames << "@" << nHostFrames / tm.getTimeSec() << "fps" << endl;
        cout << "       Decoding time: " << endl;
        cout << "           Total: " << tm.getTimeMilli() << "ms" << endl;
        cout << "           Average: " << tm.getTimeMilli() / nHostFrames << "ms" << endl;
        cout << "           Decoding FPS: " << nHostFrames / tm.getTimeSec() << endl;
        if (tm.getTimeMilli() < gpuDecodingTimeMs)
            cout << "VideoReader decoding was slower than VideoCapture, likely that video resolution and/or the number of frames to decode are too small to saturate the Nvidia decoder and realize performance gains from hardware decoding." << endl;
    }
    // try to use pollKey


    // benchmark would also include CPU, but no display
    //cv::TickMeter tm;
    //cv::Mat frame;
    //cv::VideoCapture reader(fname);
    //for (;;)
    //{
    //    if (!reader.read(frame))
    //        break;
    //    cv::imshow("CPU", frame);
    //    if (cv::waitKey(3) > 0)
    //        break;
    //}

//    cv::cuda::GpuMat d_frame;
//    cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
//    for (;;)
//    {
//        if (!d_reader->nextFrame(d_frame))
//            break;
//#if defined(HAVE_OPENGL)
//        cv::imshow("GPU", cv::ogl::Texture2D(d_frame));
//#else
//        d_frame.download(frame);
//        cv::imshow("GPU", frame);
//#endif
//        if (cv::waitKey(3) > 0)
//            break;
//    }

    return 0;
}
//! [main]

#else

int main()
{
    std::cout << "OpenCV was built without CUDA Video decoding support\n" << std::endl;
    return 0;
}

#endif
