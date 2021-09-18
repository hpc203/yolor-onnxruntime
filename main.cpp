#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class YOLO
{
public:
	YOLO(float nms_threshold, float objThreshold);
	void detect(Mat& cv_image);
private:
	float nms_threshold;
	float objThreshold;
	const string model_path = "yolor-p6-640-640.onnx";
	const string classesFile = "label.txt";
	vector<string> class_names;
	int num_class;

	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
	vector<float> normalize_(Mat img);
	void nms(vector<BoxInfo>& input_boxes);
	const bool keep_ratio = false;
	const bool swaprgb = true;
	const int inpWidth = 640;
	const int inpHeight = 640;
	int num_anchors;
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolor");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

YOLO::YOLO(float nms_threshold, float objThreshold)
{
	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();

	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	auto pred_dims = output_node_dims.at(0); // (1,n,85)
	num_anchors = pred_dims.at(1);
}

Mat YOLO::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1)
		{
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
		}
		else
		{
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else
	{
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

vector<float> YOLO::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	vector<float> output(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				output[c * row * col + i * col + j] = pix / 255.0;
				//output[c * row * col + i * col + j] = pix;
			}
		}
	}
	return output;
}

void YOLO::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nms_threshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void YOLO::detect(Mat& srcimg)
{
	int newh = 0, neww = 0, top = 0, left = 0;
	Mat cv_image = srcimg.clone();
	if (this->swaprgb)
	{
		cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
	}
	Mat dst = this->resize_image(cv_image, &newh, &neww, &top, &left);
	vector<float> input_image_ = this->normalize_(dst);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
	vector<BoxInfo> generate_boxes;
	const float* outs = ort_outputs[0].GetTensorMutableData<float>();

	/////generate proposals
	float ratioh = (float)srcimg.rows / newh, ratiow = (float)srcimg.cols / neww;
	int i = 0, j = 0, nout = this->class_names.size() + 5;
	for (i = 0; i < num_anchors; i++)
	{
		const float* pdata = outs + i * nout;
		float obj_conf = pdata[4];
		if (obj_conf > this->objThreshold)
		{
			int max_ind = 0;
			float max_class_socre = 0;
			for (j = 0; j < this->class_names.size(); j++)
			{
				if (pdata[5 + j] > max_class_socre)
				{
					max_class_socre = pdata[5 + j];
					max_ind = j;
				}
			}
			float x0 = max<float>((pdata[0] - 0.5 * pdata[2] - left)*ratiow, 0.f);
			float y0 = max<float>((pdata[1] - 0.5 * pdata[3] - top)*ratioh, 0.f);
			float x1 = min<float>((pdata[0] + 0.5 * pdata[2] - left)*ratiow, (float)cv_image.cols);
			float y1 = min<float>((pdata[1] + 0.5 * pdata[3]- top)*ratioh, (float)cv_image.rows);
			generate_boxes.push_back(BoxInfo{ x0, y0, x1, y1, max_class_socre * obj_conf, max_ind });
		}
	}
	
	//// Perform non maximum suppression to eliminate redundant overlapping boxes with
	//// lower confidences
	nms(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = (int)generate_boxes[i].x1;
		int ymin = (int)generate_boxes[i].y1;
		int xmax = (int)generate_boxes[i].x2;
		int ymax = (int)generate_boxes[i].y2;
		rectangle(srcimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(srcimg, label, Point(xmin, ymin-5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}

int main()
{
	YOLO mynet(0.5, 0.5);
	string imgpath = "bus.jpg";
	Mat srcimg = imread(imgpath);
	mynet.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}