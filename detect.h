#pragma once

#define WINVER 0x0601          // Windows 7
#define _WIN32_WINNT 0x0601    // Windows 7
#define NTDDI_VERSION 0x06010000  // Windows 7

#include <iostream>
#include <vector>
#include <string>
#include "opencv2\opencv.hpp"
#include "nlohmann\json.hpp"
#include <torch\torch.h>
#include <torch\script.h>
#include <tuple>



#define IMG_SIZE_H 640  
#define IMG_SIZE_W 640  
#define IMG_CHN 3
#define NOMINMAX

// 检测框结构
struct Detection {
    cv::RotatedRect box;  // 中心坐标+尺寸+角度
    float score;
    int class_id;
};

//机场评估函数
struct damage_use
{
    double angle;                // 区域对应的旋转角度
    double score;                // 区域得分（越小越好）
    int area;                    // 区域大小（像素）
    cv::Point2f rectPoints[4];   // 区域四个角的坐标
};

//目标类别
static std::vector<std::string> classes = {
    "car",
    "plane",
    "build",
    "runway",
    "bridge"
    };


//全局变量
//std::string get_executable_path();
void preprocess(cv::Mat& img, at::Tensor& input_tensor, float& ratio, int& padding_top, int& padding_left);

#ifdef DLL_EXPORTS
#define MYFUNC_API __declspec(dllexport)  // 导出函数
#else
#define MYFUNC_API __declspec(dllimport)  // 导入函数
#endif

#ifdef __cplusplus
extern "C" {
#endif

    MYFUNC_API int load_models(const char* rotate_path);
    MYFUNC_API int target_detect(const char* image_path, const char* json_path,
        bool car, bool plane, bool build,
        bool runway );

#ifdef __cplusplus
}
#endif
