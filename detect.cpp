#define WINVER 0x0601
#define _WIN32_WINNT 0x0601

#include "detect.h"

#ifdef _WIN32
#include <windows.h>

std::string get_executable_path() {
    char path[MAX_PATH] = { 0 };

    // 使用 GetModuleFileNameA 而不是可能依赖新API的函数
    DWORD length = GetModuleFileNameA(NULL, path, MAX_PATH);
    if (length == 0 || length == MAX_PATH) {
        return "";
    }

    std::string fullPath(path);
    size_t lastSlash = fullPath.find_last_of("\\/");
    return (lastSlash != std::string::npos) ? fullPath.substr(0, lastSlash) : "";
}

#elif __linux__
#include <dlfcn.h>
#include <unistd.h>
#include <limits.h>

std::string get_executable_path() {
    Dl_info info;
    // 获取当前执行路径
    if (dladdr((void*)get_executable_path, &info)) {
        std::string fullPath(info.dli_fname);
        size_t lastSlash = fullPath.find_last_of("\\/");
        return (lastSlash != std::string::npos) ? fullPath.substr(0, lastSlash) : "";
    }
    return "";
}

#endif


//功能函数
//角度规范化
void normalize_rboxes(torch::Tensor& box_tensor) {
    // box_tensor: [N,5] -> x, y, w, h, t
    auto w = box_tensor.index({ torch::indexing::Slice(), 2 });
    auto h = box_tensor.index({ torch::indexing::Slice(), 3 });
    auto t = box_tensor.index({ torch::indexing::Slice(), 4 });

    // swap = t % pi >= pi/2
    auto t_mod_pi = t.fmod(M_PI);
    auto swap = t_mod_pi.ge(M_PI / 2);

    // 交换 w 和 h（就地修改）
    auto w_new = torch::where(swap, h, w);
    auto h_new = torch::where(swap, w, h);
    w.copy_(w_new);
    h.copy_(h_new);

    // 角度规范到 [0, pi/2)
    t.copy_(t.fmod(M_PI / 2));
}

//协方差矩阵计算
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_covariance_matrix(const torch::Tensor& obb) {
    auto w = obb.select(1, 2);   // width
    auto h = obb.select(1, 3);   // height
    auto r = obb.select(1, 4);   // rotation

    auto cos_r = r.cos();
    auto sin_r = r.sin();

    torch::Tensor a = (w.pow(2) * cos_r.pow(2) + h.pow(2) * sin_r.pow(2)) / 12.0;
    torch::Tensor b = (w.pow(2) * sin_r.pow(2) + h.pow(2) * cos_r.pow(2)) / 12.0;
    torch::Tensor c = ((h.pow(2) - w.pow(2)) * sin_r * cos_r) / 12.0;

    return std::make_tuple(a, b, c);
}

//计算检测框重合情况
torch::Tensor iou_rotate(const torch::Tensor& obb1, const torch::Tensor& obb2, double eps = 1e-7) {
    torch::Tensor obb1_tensor = obb1;
    torch::Tensor obb2_tensor = obb2;

    // 1. 中心坐标
    auto x1 = obb1_tensor.select(1, 0).unsqueeze(-1); // (N,1)
    auto y1 = obb1_tensor.select(1, 1).unsqueeze(-1);
    auto x2 = obb2_tensor.select(1, 0).unsqueeze(0);  // (1,M)
    auto y2 = obb2_tensor.select(1, 1).unsqueeze(0);

    // 2. 协方差矩阵
    torch::Tensor a1, b1, c1;
    std::tie(a1, b1, c1) = get_covariance_matrix(obb1_tensor);
    torch::Tensor a2, b2, c2;
    std::tie(a2, b2, c2) = get_covariance_matrix(obb2_tensor);

    a2 = a2.unsqueeze(0); b2 = b2.unsqueeze(0); c2 = c2.unsqueeze(0);

    // 3. t1, t2, t3
    torch::Tensor t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) /
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25;

    torch::Tensor t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) /
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5;

    torch::Tensor t3 = ((((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) /
        (4.0 * ((a1 * b1 - c1.pow(2)).clamp_min(0.0) *
            (a2 * b2 - c2.pow(2)).clamp_min(0.0)).sqrt() + eps) + eps).log()) * 0.5;

    // 4. 概率距离
    torch::Tensor bd = (t1 + t2 + t3).clamp(eps, 100.0);
    torch::Tensor hd = (1.0 - (-bd).exp() + eps).sqrt();

    return 1.0 - hd; // 相似度矩阵 (N,M)
}

std::vector<torch::Tensor> nms(at::Tensor preds, float score, float iou,std::vector<int> select_target)
{
    std::vector<torch::Tensor> output_all;

    //yolov8-obb输出格式：[batch,4+num+1,8400]
    for (size_t i = 0; i < preds.sizes()[0]; ++i)
    {
        torch::Tensor pred = preds.select(0, i);

        //std::cout << pred.sizes() << std::endl;

        //转换为按照行储存数据
        pred = pred.permute({ 1,0 }).contiguous();
        std::cout << pred.sizes() << std::endl;

        //分离置信分数与检测框
        torch::Tensor index_tensor = pred.slice(1, 0, 4);//坐标
        torch::Tensor conf_tensor = pred.slice(1, 4, pred.size(1) - 1);//置信度
        torch::Tensor theta_tensor = pred.slice(1, pred.size(1) - 1, pred.size(1));//角度分离
        torch::Tensor box_tensor = torch::cat({ index_tensor ,theta_tensor }, 1);
        //std::cout << box_tensor.sizes() << std::endl;
        //std::cout << box_tensor.sizes() << std::endl;
        //std::cout << conf_tensor.sizes() << std::endl;

        //获取每组预测结果的最大分类置信度
        auto result = torch::max(conf_tensor, 1);
        torch::Tensor scores = std::get<0>(result); //分数
        torch::Tensor id = std::get<1>(result);//分数对应的索引

        //根据检测框置信度阈值过滤数据
        std::cout << box_tensor.sizes() << std::endl;
        auto mask = scores > score;
        box_tensor = box_tensor.index({ mask });
        std::cout << box_tensor.sizes() << std::endl;
        scores = scores.index({ mask });
        id = id.index({ mask });
        //std::cout << id.sizes() << std::endl;
        if (box_tensor.sizes()[0] == 0) continue;
        std::vector<torch::Tensor> nms_results;
        //角度规范化
        normalize_rboxes(box_tensor);

        
        //iou计算
        std::vector<torch::Tensor> output_vec;
        for (int j = 0; j < conf_tensor.size(1); j++) {

            ////筛选不需要的类别
            //bool found = false;
            //for (int num : select_target)
            //{
            //    if (num == j)
            //    {
            //        found = true;
            //        break;
            //    }
            //}
            //if (not found) continue;

            auto mask_id = (id == j);
            torch::Tensor box_tensor_id = box_tensor.index({ mask_id });
            torch::Tensor scores_id = scores.index({ mask_id });

            // 按置信度降序排序
            auto sorted_result = std::get<1>(scores_id.sort(-1, /*descending=*/true));
            box_tensor_id = box_tensor_id.index_select(0, sorted_result);
            //std::cout << box_tensor_id.sizes() << std::endl;
            scores_id = scores_id.index_select(0, sorted_result);

            // 如果没有框，直接跳过
            if (box_tensor_id.size(0) == 0) continue;

            //计算 Probiou 矩阵
            torch::Tensor ious = iou_rotate(box_tensor_id, box_tensor_id);

            //使用上三角矩阵，只保留上三角部分
            ious = ious.triu(1);

            //NMS 筛选
            std::vector<int64_t> keep_idx;
            torch::Tensor remaining_idx = torch::arange(box_tensor_id.size(0));       
            while (remaining_idx.numel() > 0) {
                if (remaining_idx.numel() == 0) break;
                //std::cout << remaining_idx << std::endl;
                int64_t current = remaining_idx[0].item<int64_t>();
                keep_idx.push_back(current);

                if (remaining_idx.numel() == 1) break;

                // 计算哪些框需要保留
                auto iou_row = ious.index({ current, remaining_idx.index({torch::arange(1, remaining_idx.size(0))}) });
                auto keep_mask = (iou_row < iou).nonzero().squeeze(1);

                // 更新 remaining_idx
                remaining_idx = remaining_idx.index({ keep_mask + 1 }); // +1 因为第0个是current
            }

            //取最终保留框
            auto keep_tensor = torch::tensor(keep_idx, torch::kLong);
            torch::Tensor final_boxes = box_tensor_id.index_select(0, keep_tensor);
            torch::Tensor final_scores = scores_id.index_select(0, keep_tensor);
            torch::Tensor final_cls = torch::full({ (int64_t)keep_idx.size() }, j, torch::kLong);

            torch::Tensor final_result = torch::cat({ final_boxes,
                                                     final_scores.unsqueeze(1),
                                                     final_cls.unsqueeze(1) }, 1);
            output_vec.push_back(final_result);
        }    
        torch::Tensor output;
        if (!output_vec.empty()) {
            output = torch::cat(output_vec, 0); // 按行拼接
        }
        else {
            // 创建一个空的Tensor，维度为[0,6] (x,y,w,h,theta,score,class)
            output = torch::empty({ 0, 6 }, torch::kFloat);
        }
        //std::cout << output << std::endl;
        output_all.push_back(output);
    }

    if (output_all.size() == 0)
    {
        torch::Tensor output = torch::empty({ 0, 6 }, torch::kFloat);
        output_all.push_back(output);
    }

    return output_all;
}

//预处理
void preprocess(cv::Mat& img, at::Tensor& input_tensor, float& ratio, int& padding_top, int& padding_left)
{
    int img_h = img.rows;
    int img_w = img.cols;

    float ratio_h = float(IMG_SIZE_H) / float(img_h);
    float ratio_w = float(IMG_SIZE_W) / float(img_w);
    ratio = std::min(ratio_h, ratio_w);

    int new_img_h = int(std::round(img_h * ratio));
    int new_img_w = int(std::round(img_w * ratio));

    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(new_img_w, new_img_h));
    //cv::imwrite("./img_resize.jpg", img_resized);

    int dw = IMG_SIZE_W - new_img_w;
    int dh = IMG_SIZE_H - new_img_h;

    int top = int(std::floor(dh / 2.0));
    int bottom = int(std::ceil(dh / 2.0));
    int left = int(std::floor(dw / 2.0));
    int right = int(std::ceil(dw / 2.0));

    padding_top = top;
    padding_left = left;

    cv::copyMakeBorder(img_resized, img_resized, top, bottom, left, right,
    cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    //cv::imwrite("./img_padding.jpg", img_resized);

    cv::cvtColor(img_resized, img_resized, cv::COLOR_BGR2RGB);
    //cv::imwrite("./img_color.jpg", img_resized);
    img_resized.convertTo(img_resized, CV_32FC3, 1.0f / 255.0f);


    input_tensor = torch::from_blob(
        img_resized.clone().data,
        { 1, img_resized.rows, img_resized.cols, img_resized.channels() },
        torch::kFloat32
    ).clone();

    input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).contiguous();
}

//绘制图像
cv::Mat visualize_rotated_boxes(
    const cv::Mat& image,
    const torch::Tensor& preds,         // [N,7] -> x,y,w,h,radian,score,class
    const std::vector<std::vector<cv::Point2f>> contours,
    const std::vector<std::string>& class_names = {}
) {
    

    torch::Tensor cpu_preds = preds.cpu();
    auto preds_acc = cpu_preds.accessor<float, 2>();

    //创建画布
    cv::Mat img = image.clone();
    for (int64_t i = 0; i < cpu_preds.size(0); ++i) {


        //绘制封闭多边形
        for (size_t j = 0; j < contours[i].size(); j++) cv::line(img, contours[i][j], contours[i][(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
        float score = preds_acc[i][5];
        int cls = static_cast<int>(preds_acc[i][6]);


        std::string label = std::to_string(cls);
        if (!class_names.empty() && cls >= 0 && cls < class_names.size())
            label = class_names[cls];
        label += ":" + std::to_string(score).substr(0, 4);

        cv::Point text_pos(contours[i][0].x, contours[i][0].y - 5);
        cv::putText(img, label, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    return img;
}
//旋转矩形框坐标转换
std::vector<cv::Point2f> order_points_clockwise(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> ordered(4);
    
    // 1. 找左上角 (y 最小，若相等则 x 最小)
    int idx_top_left = 0;
    for (int i = 1; i < 4; i++) {
        if (pts[i].y < pts[idx_top_left].y ||
           (std::abs(pts[i].y - pts[idx_top_left].y) < 1e-6 && pts[i].x < pts[idx_top_left].x)) {
            idx_top_left = i;
        }
    }
    ordered[0] = pts[idx_top_left]; // 左上

    // 2. 计算其他点相对左上角的角度，顺时针排序
    std::vector<std::pair<float,int>> angles;
    for (int i = 0; i < 4; i++) {
        if (i == idx_top_left) continue;
        float dx = pts[i].x - ordered[0].x;
        float dy = pts[i].y - ordered[0].y;
        float angle = std::atan2(dy, dx);
        angles.emplace_back(angle, i);
    }

    // 顺时针排列：atan2 从 -pi 到 pi，逆时针增大 → 我们反向排序
    std::sort(angles.begin(), angles.end(),
              [](auto& a, auto& b){ return a.first < b.first; });

    // 3. 按顺时针填充
    for (int k = 0; k < 3; k++) {
        ordered[k+1] = pts[angles[k].second];
    }
    return ordered;
}


// 四点排序函数：左上 → 右上 → 右下 → 左下
std::vector<cv::Point2f> index_sort(const std::vector<cv::Point2f>& pts) 
{
    std::vector<cv::Point2f> ordered(4);

    // 1. 找左上角 (y 最小，若相等则 x 最小)
    int idx_top_left = 0;
    for (int i = 1; i < 4; i++) {
        if (pts[i].y < pts[idx_top_left].y ||
            (std::abs(pts[i].y - pts[idx_top_left].y) < 1e-6 && pts[i].x < pts[idx_top_left].x)) {
            idx_top_left = i;
        }
    }
    ordered[0] = pts[idx_top_left];

    // 2. 将剩下三个点按相对于左上角的角度排序（顺时针）
    std::vector<std::pair<float, int>> angles;
    for (int i = 0; i < 4; i++) {
        if (i == idx_top_left) continue;
        float dx = pts[i].x - ordered[0].x;
        float dy = pts[i].y - ordered[0].y;
        float angle = std::atan2(dy, dx);
        angles.emplace_back(angle, i);
    }

    std::sort(angles.begin(), angles.end(),
        [](auto& a, auto& b) { return a.first < b.first; });

    // 顺时针填充
    for (int k = 0; k < 3; k++) {
        ordered[k + 1] = pts[angles[k].second];
    }

    return ordered;
}

// xywhr -> 角点坐标 (原图系, 左上开始顺时针)
std::vector<std::vector<cv::Point2f>> obb_to_xy(
    const torch::Tensor& boxes, float ratio, int pad_top, int pad_left)
{
    std::vector<std::vector<cv::Point2f>> polys;
    polys.reserve(boxes.size(0));

    auto acc = boxes.accessor<float, 2>();
    for (int i = 0; i < boxes.size(0); i++) {
        float cx = acc[i][0];
        float cy = acc[i][1];
        float w = acc[i][2];
        float h = acc[i][3];
        float r = acc[i][4];

        std::vector<cv::Point2f> relPts = {
            {  w / 2,  h / 2},
            { -w / 2,  h / 2},
            { -w / 2, -h / 2},
            {  w / 2, -h / 2}
        };

        std::vector<cv::Point2f> pts;
        pts.reserve(4);

        float cos_r = std::cos(r);
        float sin_r = std::sin(r);

        for (auto& p : relPts) {
            float x_pred = cx + p.x * cos_r - p.y * sin_r;
            float y_pred = cy + p.x * sin_r + p.y * cos_r;

            float x_orig = (x_pred - pad_left) / ratio;
            float y_orig = (y_pred - pad_top) / ratio;

            pts.emplace_back(cv::Point2f(x_orig, y_orig));
        }
        std::vector<cv::Point2f> final_pts = index_sort(pts);
        polys.push_back(final_pts);
    }

    return polys;
}


void result_to_json(torch::Tensor dets, std::vector<std::vector<cv::Point2f>> contours,nlohmann::json& Final_json)
{

    for (size_t i = 0; i < contours.size(); i++)
    {

        //查找哪些类别需要保留
        int cls = dets[i][6].item<int>();
        float conf = dets[i][5].item<float>();

        //josn中写入类别和置信度
        nlohmann::json single_result;
        single_result["classification"] = classes[cls];
        single_result["confidence"] = conf;

        //json中写入角点坐标
        nlohmann::json pos;
        for (size_t j = 0; j < contours[i].size(); j++)
        {
            pos.push_back(contours[i][j].x);
            pos.push_back(contours[i][j].y);
        }
        single_result["pos"] = pos;
        Final_json.push_back(single_result);
    }
}

//机场跑道检测框绘制
std::vector<cv::Point> runway_index(cv::Mat result_img)
{
    cv::Mat mask;
    cv::inRange(result_img, cv::Scalar(50), cv::Scalar(50), mask);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 假设只关心最大的区域
    int max_idx = -1;
    double max_area = 0.0;
    for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_idx = i;
        }
    }

    if (max_idx >= 0) {
        // 多边形近似
        std::vector<cv::Point> poly;
        cv::approxPolyDP(contours[max_idx], poly, 2.0, true);

        // 找到最靠近左上角的点（x+y最小）
        int start_idx = 0;
        int min_sum = poly[0].x + poly[0].y;
        for (int i = 1; i < poly.size(); i++) {
            int s = poly[i].x + poly[i].y;
            if (s < min_sum) {
                min_sum = s;
                start_idx = i;
            }
        }
        //判断顺时针还是逆时针
        double area_signed = cv::contourArea(poly, true); // true 表示有符号面积
        bool is_clockwise = (area_signed < 0); // 面积为负表示顺时针

        //从 start_idx 开始，顺时针排序
        std::vector<cv::Point> ordered_poly;
        for (int i = 0; i < poly.size(); i++) {
            int idx;
            if (is_clockwise) {
                idx = (start_idx + i) % poly.size();
            }
            else {
                idx = (start_idx - i + poly.size()) % poly.size();
            }
            ordered_poly.push_back(poly[idx]);
        }

        return ordered_poly;
    }
}


#ifdef __cplusplus
extern "C" 
{
#endif
torch::jit::script::Module rotate_model;
torch::jit::script::Module runway_model;
torch::Device device(torch::kCPU);

//主要执行函数
MYFUNC_API int load_models(const char* rotate_path)
{
    //std::cout << "LibTorch 版本: " << TORCH_VERSION << std::endl;
    std::string path = get_executable_path();
    std::string rotate_model_path(rotate_path);

    torch::NoGradGuard no_grad;
    

    std::cout << "rotate_model_path: " << rotate_model_path << std::endl;

    // 尝试加载第一个模型
    try {
        rotate_model = torch::jit::load(rotate_model_path, device);
    }
    catch (const c10::Error& e) {
        std::cerr << "[Error] Failed to load model 1: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "[Error] Model 1 file error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

MYFUNC_API int target_detect(const char* image_path, const char* json_path, bool car, bool plane, bool build, bool runway)
{
    //std::cout << "开始进行目标检测" << std::endl;
    std::string Image_path(image_path);
    std::string Json_path(json_path);

    //获取保存文件名称
    size_t last_name = Image_path.find_last_of("/\\");
    std::string filename = Image_path.substr(last_name + 1);

    //去掉后缀
    size_t last_dot = filename.find_last_of(".");
    if (last_dot != std::string::npos) {
        filename = filename.substr(0, last_dot);
    }

    Json_path += "/" + filename + ".json";
    nlohmann::json Final_josn = nlohmann::json::array();

    cv::Mat img = cv::imread(Image_path);
    if (img.empty())
    {
        std::cerr << "Failed to load image: " << Image_path << std::endl;
        return 1;
    }

    //判断是否需要进行检测任务
    std::vector<int> select_target;
    select_target.clear();
    if (car) select_target.push_back(0);
    if (build) select_target.push_back(1);
    if (plane) select_target.push_back(2);
    if (runway) select_target.push_back(3);

    // 图像预处理，包括等比例缩放，padding到所需的尺寸，归一化，转成tensor
    float ratio;
    int padding_top, padding_left;
    at::Tensor img_tensor;
    preprocess(img, img_tensor, ratio, padding_top, padding_left);

    //模型推理
    img_tensor = img_tensor.to(device);
    //std::cout << img_tensor.sizes() << std::endl;
    auto value = rotate_model.forward({ img_tensor });

    //auto elements = output->elements();w
    //at::Tensor tensor = elements.at(0).toTensor();
    at::Tensor tensor = value.toTensor();

    // 调整类型到 CPU
    tensor = tensor.to(at::kCPU);
    //std::cout << tensor.sizes() << std::endl;

    // 设置 NMS 阈值
    float score_threshold = 0.5f;  // 置信度阈值
    float iou_threshold = 0.5f;   // NMS IOU 阈值
    std::vector<torch::Tensor> nms_result = nms(tensor, score_threshold, iou_threshold, select_target);
    //std::cout << nms_result[0].sizes() << std::endl;
    //std::cout << nms_result[2] << std::endl;
    std::vector<std::vector<cv::Point2f>> contours = obb_to_xy(nms_result[0], ratio, padding_top, padding_left);
    //std::cout << contours << std::endl;
    result_to_json(nms_result[0], contours, Final_josn);

    std::vector<std::string> class_names = { "car","plane","build","runway"};

    cv::Mat vis_img = visualize_rotated_boxes(img, nms_result[0],contours,class_names);

    cv::imshow("Rotated Boxes", vis_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    //检测结果清空
    select_target.clear();
    select_target.shrink_to_fit();

    nms_result.clear();
    nms_result.shrink_to_fit();

    contours.clear();
    for (auto& c : contours) c.shrink_to_fit();
    contours.shrink_to_fit();

    if (tensor.defined()) {
        tensor.reset();
    }

/*
    //如果选择了机场跑道检测
    if (runway)
    {
        cv::Mat src_img = img.clone();
        cv::resize(src_img, src_img, cv::Size(320, 320));

        //转换为浮点数的Tensor格式
        src_img.convertTo(src_img, CV_32FC3, 1.0f / 255.0f);

        torch::Tensor input_tensor = torch::from_blob(
            src_img.clone().data,
            { 1, src_img.rows, src_img.cols, src_img.channels() },
            torch::kFloat32
        ).clone();

        input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).contiguous();

        auto result = runway_model({ input_tensor });

        torch::Tensor logits = result.toTensor();//转换为Tensor
        torch::Tensor probs = torch::softmax(logits, 1);//计算softmax
        torch::Tensor max_indices = torch::argmax(probs, 1);
        int predicted_class = max_indices[0].item<int>();

        //检测结果储存为json数据
        nlohmann::json runway_json;
        int name = 3;
        runway_json["classification"] = name;
        runway_json["existence"] = predicted_class;
        runway_json["confidence"] = probs[0][predicted_class].item<float>();

        runway_json["pos"] = " ";
        Final_josn.push_back(runway_json);

        //重置json对象
        runway_json = nlohmann::json();

        //图像数据清空
        src_img.release();
     }
*/
     
     
    //if (runway)
    //{
    //    int IMG_WIDTH = 1024;
    //    int IMG_HEIGHT = 1024;

    //    cv::Mat src_image = img.clone();
    //    int width = src_image.rows;
    //    int height = src_image.cols;

    //    // 将图像缩放为 1024x1024
    //    cv::resize(src_image, src_image, cv::Size(IMG_WIDTH, IMG_HEIGHT));
    //    // 转为 RGB 格式
    //    cv::cvtColor(src_image, src_image, cv::COLOR_BGR2RGB);

    //    // 转为 float 类型
    //    src_image.convertTo(src_image, CV_32FC3);

    //    // 构造 Torch 输入张量
    //    at::Tensor input_tensor = torch::from_blob(src_image.data, { 1, IMG_WIDTH, IMG_HEIGHT, IMG_CHN }).toType(torch::kFloat32);
    //    input_tensor = input_tensor.permute({ 0, 3, 1, 2 }); // 调整通道顺序为 NCHW
    //    // 归一化（除以 255）
    //    input_tensor[0][0] = input_tensor[0][0].div(255);
    //    input_tensor[0][1] = input_tensor[0][1].div(255);
    //    input_tensor[0][2] = input_tensor[0][2].div(255);

    //    torch::Device device(torch::kCPU); // 使用 CPU 推理
    //    input_tensor = input_tensor.to(device);

    //    torch::NoGradGuard no_grad;

    //    // 推理
    //    at::Tensor output = runway_model.forward({ input_tensor }).toTensor();
    //    output = torch::squeeze(output);
    //    output = torch::argmax(output, 0); // 获取每个像素的分类结果

    //    // 将推理结果转换为 OpenCV 的 Mat 格式
    //    cv::Mat seg_result(1024, 1024, CV_8UC1);
    //    for (int i = 0; i < 1024; i++)
    //    {
    //        for (int j = 0; j < 1024; j++)
    //        {
    //            int label = output[i][j].item<int>();
    //            seg_result.at<uchar>(i, j) = label * 50; // 颜色映射
    //        }
    //    }
    //    //分割结果尺寸还原
    //    cv::resize(seg_result, seg_result, cv::Size(img.cols, img.rows), 0, 0, cv::INTER_NEAREST);
    //    std::vector<cv::Point> result_index = runway_index(seg_result);

    //    //检测结果储存为json数据
    //    nlohmann::json runway_json;
    //    int name = 4;
    //    runway_json["classification"] = name;
    //    runway_json["confidence"] = 1;
    //    nlohmann::json runway_pos;
    //    for (size_t i = 0; i < result_index.size(); i++)
    //    {
    //        runway_pos.push_back(result_index[i].x);
    //        runway_pos.push_back(result_index[i].y);
    //    }
    //    runway_json["pos"] = runway_pos;
    //    Final_josn.push_back(runway_json);

    //    //重置json对象
    //    runway_json = nlohmann::json();

    //    ////绘制检测结果
    //    //std::cout << result_index << std::endl;
    //    //cv::Mat draw_img = img.clone();
    //    //std::vector<std::vector<cv::Point>> contours;
    //    //contours.push_back(result_index);
    //    //cv::polylines(draw_img, contours, true, cv::Scalar(0, 0, 255), 2);
    //    //cv::imshow("polygon_draw.png", draw_img);
    //    //cv::waitKey(0);
    //    // 
    //    //图像数据清空
    //    src_image.release();
    //    seg_result.release();

    //    //检测结果清空
    //    result_index.clear();
    //    result_index.shrink_to_fit();

    //    if (output.defined()) {
    //        output.reset();
    //    }

    //}
    
    //保存最终的json文件
    std::ofstream file(Json_path);

    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << Json_path << std::endl;
        return -1;
    }
    // pretty 格式化写入（带缩进）
    file << Final_josn.dump(4);
    file.close();

    // 重置JSON对象
    Final_josn = nlohmann::json();

    //清空OpenCV的内存
    img.release();


    return 0;
}
#ifdef __cplusplus
}
#endif

