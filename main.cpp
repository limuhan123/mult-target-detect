#include "detect.h"
#include "filesystem"



int main(int argc, char* argv[])
{
	if (argc == 8)
	{
		std::string model_path = argv[1];
		std::string img_path = argv[2];
		std::string json_path = argv[3];
		bool car = argv[4];
		bool plane = argv[5];
		bool build = argv[6];
		bool runway = argv[7];

		const char* model = model_path.c_str();
		const char* img = img_path.c_str();
		const char* json = json_path.c_str();

		load_models(model);
		target_detect(img, json, car, plane, build, runway);
	}

	else
	{
		std::cout << "请正确输入参数再运行程序" << std::endl;
		std::cout << "正确的参数顺序：\n" << "权重路径，图像路径，json保存路径，车辆，飞机，建筑物，跑道 " << std::endl;
	}

	system("pause");
	return 0;
}




/*
int main()
{
	
	load_models("D:/cpp/target-detect/x64/Release/model_1.torchscript");
	std::string img_path = "D:/cpp/target-detect/x64/Release/train/4.png";
	//std::string img_path = "D:/QT/exersice/CS-detect/build/Desktop_Qt_6_6_3_MSVC2019_64bit-Release/release/data/0/images/0001.png";
	std::string json_path = "D:/cpp/target-detect/x64/Release/train/";
	const char* img = img_path.c_str();
	const char* json = json_path.c_str();
	target_detect(img, json,true, true, true,true);

	//测试代码
	std::string class_model_path = "C:/Users/li_muhan/Desktop/实验/runway_model.torchscript";
	torch::jit::Module class_model = torch::jit::load(class_model_path);

	std::string test_img_path = "C:/Users/li_muhan/Desktop/实验/runway_data/train/01/image-1.jpg";

	std::string folder_path = "C:/Users/li_muhan/Desktop/实验/runway_data/train/01";

	std::vector<std::string> img_file;
	for (const auto& entry : std::filesystem::directory_iterator(folder_path))
	{
		if (entry.is_regular_file()) {
			img_file.push_back(entry.path().string());
		}
	}

	for (int i = 0; i < img_file.size(); i++)
	{
		cv::Mat input_img = cv::imread(img_file[i]);

		cv::resize(input_img, input_img, cv::Size(320, 320));
		input_img.convertTo(input_img, CV_32FC3, 1.0f / 255.0f);

		torch::Tensor input_tensor = torch::from_blob(
			input_img.clone().data,
			{ 1, input_img.rows, input_img.cols, input_img.channels() },
			torch::kFloat32
		).clone();

		input_tensor = input_tensor.permute({ 0, 3, 1, 2 }).contiguous();

		auto result = class_model({ input_tensor });
		//std::cout << result << std::endl;
		torch::Tensor logits = result.toTensor();//转换为Tensor
		torch::Tensor probs = torch::softmax(logits, 1);//计算softmax
		torch::Tensor max_indices = torch::argmax(probs, 1);
		int predicted_class = max_indices[0].item<int>();

		std::cout << "最大类别索引:" << predicted_class << std::endl;
	}
	system("pause");
}
*/
