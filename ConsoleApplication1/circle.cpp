#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<sstream>
using namespace cv;
using namespace std;
#define PI 3.1415926

//拟合曲线
bool polynomial_curve_fit(std::vector<cv::Point2f>& key_point, int n, cv::Mat&A)
{
	//Number of key points
	int N = key_point.size();
	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_32FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<float>(i, j) = X.at<float>(i, j) + std::pow(key_point[k].x, i + j);
			}
		}
	}
	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_32FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<float>(i, 0) = Y.at<float>(i, 0) + std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}
	A = cv::Mat::zeros(n + 1, 1, CV_32FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}

//拟合圆
Point3f LeastSquareFittingCircle(vector<Point2f> temp_coordinates)//高斯消元法直接求解方程组
{
	float x1 = 0;
	float x2 = 0;
	float x3 = 0;
	float y1 = 0;
	float y2 = 0;
	float y3 = 0;
	float x1y1 = 0;
	float x1y2 = 0; float x2y1 = 0;
	int num;
	vector<Point2f>::iterator k;
	Point3f tempcircle;
	for (k = temp_coordinates.begin(); k != temp_coordinates.end(); k++)
	{
		x1 = x1 + (*k).x;
		x2 = x2 + (*k).x * (*k).x;
		x3 = x3 + (*k).x * (*k).x * (*k).x;
		y1 = y1 + (*k).y;
		y2 = y2 + (*k).y * (*k).y;
		y3 = y3 + (*k).y * (*k).y * (*k).y;
		x1y1 = x1y1 + (*k).x * (*k).y;
		x1y2 = x1y2 + (*k).x * (*k).y * (*k).y;
		x2y1 = x2y1 + (*k).x * (*k).x * (*k).y;
	} float C, D, E, G, H, a, b, c;
	num = temp_coordinates.size();
	C = num * x2 - x1 * x1;
	D = num * x1y1 - x1 * y1;
	E = num * x3 + num * x1y2 - x1 * (x2 + y2);
	G = num * y2 - y1 * y1;
	H = num * x2y1 + num * y3 - y1 * (x2 + y2);
	a = (H * D - E * G) / (C * G - D * D);
	b = (H * C - E * D) / (D * D - G * C);
	c = -(x2 + y2 + a * x1 + b * y1) / num;
	tempcircle.x = -a / 2; //圆心x坐标
	tempcircle.y = -b / 2;//圆心y坐标
	tempcircle.z = sqrt(a * a + b * b - 4 * c) / 2;//圆心半径
	return tempcircle;
}

void circle_self(Mat src, Mat &result, Point3f &position)
{
	Mat dst;
	threshold(src, dst, 220, 255, CV_THRESH_BINARY);
	//imshow("二值化结果", dst);
	Mat element = getStructuringElement(MORPH_RECT, Size(11, 11));
	erode(dst, dst, element);
	dilate(dst, dst, element);
	cv::imshow("分割结果", dst);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(dst, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Rect> boundRect(contours.size());  //定义外接矩形集合
	vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
	Point2f rect[4];
	for (int i = 0; i < contours.size(); i++)
	{
		box[i] = minAreaRect(Mat(contours[i]));  //计算每个轮廓最小外接矩形
		boundRect[i] = boundingRect(Mat(contours[i]));
		circle(src, Point(box[i].center.x, box[i].center.y), 5, Scalar(0, 255, 0), -1, 8);  //绘制最小外接矩形的中心点
		//box[i].points(rect);  //把最小外接矩形四个端点复制给rect数组
		//rectangle(src,Point(boundRect[i].x, boundRect[i].y), Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), Scalar(0, 255, 0), 2, 8);
	}
	//imshow("找圆心", src);

	int m = 10;//平滑程度

	src.copyTo(result);
	int circle_number, r, x, y;
	for (int i = 0; i < contours.size(); i++)
	{
		cv::circle(result, Point(box[i].center.x, box[i].center.y), boundRect[i].width*0.45, Scalar(0, 255, 0), 1.5, 8);
		cv::circle(result, Point(box[i].center.x, box[i].center.y), boundRect[i].width*0.55, Scalar(0, 255, 0), 1.5, 8);
		if (boundRect[i].width > 200) //估计最大圆的圆心坐标与半径
		{
			r = boundRect[i].width*0.5;
			x = box[i].center.x;
			y = box[i].center.y;
			circle_number = i;
		}
	}
	//检测最大的圆
	int ring_r = boundRect[circle_number].width*0.05; //圆环宽度
	float Ang_Resolution = 2 * PI / 12; //角度分辨率
	vector<Point2f> fitting_points;
	for (int i = 0; i < 12; i++)
	{
		vector<Point3f> edge;
		for (int j = 0; j < ring_r * 2; j++)
		{
			float temp_r = r - ring_r + j;
			float temp_x = x + temp_r * cos(Ang_Resolution * i);
			float temp_y = y + temp_r * sin(Ang_Resolution * i);
			Point3f edge_temp;
			//采样点坐标
			edge_temp.x = temp_x - x;
			edge_temp.y = temp_y - y;
			//采样点灰度值
			edge_temp.z = src.at<uchar>(temp_y, temp_x);
			circle(result, Point(temp_x, temp_y), 1, Scalar(0), -1);
			edge.push_back(edge_temp);
		}

		//求导
		vector<Point3f> edge_d = edge;
		float derivate_max = 0;
		int derivate_maxIndex = 0;
		edge_d[0].z = abs(edge[1].z - 0) / 2;
		for (int k = 1; k < edge.size() - 1; ++k)
		{
			float derivate = abs(edge[k + 1].z - edge[k - 1].z) / 2;
			if (derivate > derivate_max)
			{
				derivate_max = derivate;
				derivate_maxIndex = k;
			}
			edge_d[k].z = derivate;

		}
		edge_d[edge.size() - 1].z = abs(0 - edge[edge.size() - 2].z) / 2;
		//抛物线拟合
		vector<Point2f> fitting_data;
		float init_x = edge_d[derivate_maxIndex - 2].x;
		float init_y = edge_d[derivate_maxIndex - 2].y;

		for (int k = 0; k < 5; ++k)
		{
			int temp_index = derivate_maxIndex - 2 + k;
			Point2f temp_data;
			temp_data.x = abs(edge_d[temp_index].x - init_x) + abs(edge_d[temp_index].y - init_y); //加速
			temp_data.y = edge_d[temp_index].z;
			fitting_data.push_back(temp_data);
		}
		Mat A;
		polynomial_curve_fit(fitting_data, 2, A);
		float a0 = A.at<float>(0, 0);
		float a1 = A.at<float>(1, 0);
		float a2 = A.at<float>(2, 0);
		float fitting_r = -a1 / (2 * a2);
		Point2f fitting_p(x + init_x + fitting_r * cos(Ang_Resolution * i), y + init_y + fitting_r * sin(Ang_Resolution * i));
		circle(result, fitting_p, 5, Scalar(0), -1);
		cout << "亚像素点" << i << ":" << fitting_p << endl;
		fitting_points.push_back(fitting_p);
	}
	position = LeastSquareFittingCircle(fitting_points);
}

int main()
{
	Mat src, dst;
	src = imread("C:/Users/98611/Desktop/circle.png", 0);

	if ((!src.data))
	{
		printf("could not load image...\n");
		return -1;
	}
	//cout << src.type() << endl;
	namedWindow("模板", CV_WINDOW_AUTOSIZE);
	imshow("原图", src);//原图显示
	Point3f position; //x,y,r
	Mat result;
	cout << "检测最大的圆" << endl;
	double timeStart1 = (double)getTickCount();  //计算程序运行时间
	circle_self(src, result, position);
	cout << "X：" << position.x << " Y:" << position.y << endl;
	cout << "D:" << 2 * position.z << endl;
	double nTime = ((double)getTickCount() - timeStart1) / getTickFrequency();
	cout << "耗时：" << nTime << "秒\n" << endl;
	circle(result, Point2f(position.x, position.y), position.z, Scalar(0, 0, 0), 3);
	circle(result, Point2f(position.x, position.y), 3, Scalar(0, 0, 0), -1);
	string position_x = to_string(int(position.x));
	string position_y = to_string(int(position.y));
	string d = to_string(2 * int(position.z));
	std::string text = "  X:" + position_x + "  Y:" + position_y + " D:" + d;
	putText(result, text, Point(position.x, position.y), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0), 1, 8, 0);
	imshow("result", result);
	waitKey(0);
	return 0;
}

