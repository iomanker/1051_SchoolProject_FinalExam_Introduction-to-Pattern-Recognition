#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;

int picture_decode(const char*);
string code[4] = { "" };
int read_data_from_code(Mat data,int n_samples);

int read_data_from_csv(const char* filename, Mat data, Mat classes, int n_samples);

string neural_network(const char*,int);
#define NUMBER_OF_TRAINING_SAMPLES 40
#define ATTRIBUTES_PER_SAMPLE 520 //(長:26x寬:20)
#define NUMBER_OF_TESTING_SAMPLES 4

#define NUMBER_OF_CLASSES 10

int main(int argc,char **argv){
	int decode_result = picture_decode(argv[2]);
	string result = neural_network(argv[1],decode_result);
	cout << result << endl;
	system("pause");
	return 1;
}

int picture_decode(const char* path){
	if (path == "")
		path = "CaptchaCreator.jpg";

	int CharsCount = 4;  // 字元數量
	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);  // 以灰階化模式載入圖片
	int GrayValue = 15; // 灰階值門檻

	// 定位
	int posX2 = 87, posY2 = 35, posX1 = 6, posY1 = 9;
	Rect rect(posX1, posY1, posX2 - posX1 + 1, posY2 - posY1);

	// 新增Mat類別暫存裁切圖片
	Mat roiImage;
	img(rect).copyTo(roiImage);

	// 起點：取得切割後的圖
	int pHorizontalColNumber = 4, pVerticalRowNumber = 1;
	int avgWidth = roiImage.size().width / pHorizontalColNumber;
	int avgHeight = roiImage.size().height / pVerticalRowNumber;
	Mat Split[4];

	for (int i = 0; i < pVerticalRowNumber; ++i)
	{
		for (int j = 0; j < pHorizontalColNumber; ++j)
		{
			Rect Split_rect(j*avgWidth, i*avgHeight, avgWidth, avgHeight);
			roiImage(Split_rect).copyTo(Split[i * pHorizontalColNumber + j]);
		}
	}
	// 終點：取得切割後的圖


	// 起點：取得解碼後字元

	for (int i = 0; i < CharsCount; ++i){
		for (int height = 0; height < Split[i].size().height; ++height)
		{
			for (int width = 0; width < Split[i].size().width; ++width){
				Scalar PickPixel = Split[i].at<uchar>(height, width);
				int Pixel = (int)PickPixel.val[0];
				if (Pixel < GrayValue){
					code[i] += "1";
				}
				else{
					code[i] += "0";
				}
			}
		}
	}
	// 終點：取得解碼後字元

	return 1;
}

int read_data_from_csv(const char* filename, Mat data, Mat classes, int n_samples)
{
	int classlabel;
	float tempfloat;
	FILE* f = fopen(filename, "r");
	if (!f){
		cout << "錯誤：找不到檔案( " << filename << " )讀取\n";
		return 0;
	}
	for (int line = 0; line < n_samples; ++line){
		for (int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); ++attribute){
			if (attribute < ATTRIBUTES_PER_SAMPLE)
			{
				fscanf(f, "%f", &tempfloat);
				data.at<float>(line, attribute) = tempfloat;
			}
			else if (attribute == ATTRIBUTES_PER_SAMPLE)
			{
				fscanf(f, "%i,", &classlabel);
				classes.at<float>(line, classlabel) = 1.0;
			}
		}
	}

	fclose(f);
	return 1;
}

int read_data_from_code(Mat data, int n_samples){
	float tempfloat;
	for (int line = 0; line < n_samples; ++line){
		for (int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); ++attribute){
			if (attribute < ATTRIBUTES_PER_SAMPLE)
			{
				if (code[line][attribute] == '1')
					tempfloat = 1;
				else
					tempfloat = 0;
				data.at<float>(line, attribute) = tempfloat;
			}
		}
	}
	return 1;
}

string neural_network(const char* trainfile,int decode_result){
	Mat training_data = Mat::zeros(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
	Mat training_classifications = Mat::zeros(NUMBER_OF_TRAINING_SAMPLES, NUMBER_OF_CLASSES, CV_32FC1);
	Mat testing_data = Mat::zeros(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);

	Mat classificationResult = Mat(1, NUMBER_OF_CLASSES, CV_32FC1);
	Point max_loc = Point(0, 0);

	if (read_data_from_csv(trainfile, training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
		read_data_from_code(testing_data, NUMBER_OF_TESTING_SAMPLES) && decode_result)
	{
		int layers_d[] = { ATTRIBUTES_PER_SAMPLE, 10, NUMBER_OF_CLASSES };
		Mat layers = Mat(1, 3, CV_32SC1);
		layers.at<int>(0, 0) = layers_d[0];
		layers.at<int>(0, 1) = layers_d[1];
		layers.at<int>(0, 2) = layers_d[2];


		CvANN_MLP nnetwork;
		nnetwork.create(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);


		CvANN_MLP_TrainParams params = CvANN_MLP_TrainParams(
			cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001),
			CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1);
		cout << endl << "使用的訓練資料庫: " << trainfile << endl;
		int iterations = nnetwork.train(training_data, training_classifications, Mat(), Mat(), params);

		Mat test_sample;
		stringstream Output_Stream;
		cout << "使用的測試圖片結果: ";
		for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
		{
			test_sample = testing_data.row(tsample);
			nnetwork.predict(test_sample, classificationResult);

			minMaxLoc(classificationResult, 0, 0, 0, &max_loc);
			//printf("Testing Sample %i -> class result (digit %d)\n", tsample, max_loc.x);
			Output_Stream << max_loc.x;
		}
		return Output_Stream.str();
	}
	else
	{
		return"發生載入錯誤";
	}
}