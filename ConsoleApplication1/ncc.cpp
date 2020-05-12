#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<sstream>
using namespace cv;
using namespace std;

void ncc(Mat &model, Mat &src,Point &max)
{
	Mat score;
	src.copyTo(score);
	score.setTo(0);
	int row_max_model = model.rows;
	int col_max_model = model.cols;
	float m_t = 0; float sum_model = 0;
	for (int y = 0; y < row_max_model; y++)
	{
		uchar* data_model = model.ptr<uchar>(y);
		for (int x = 0; x < col_max_model ; x++)
		{
			sum_model += data_model[x];
		}
	}
	m_t = sum_model / (row_max_model+1) / (col_max_model+1) ; //model均值
	//cout << m_t << endl;
	float s_t_2=0,st_temp=0;
	for (int y = 0; y < row_max_model; y++)
	{
		uchar* data_model = model.ptr<uchar>(y);
		for (int x = 0; x < col_max_model; x++)
		{
			st_temp += (data_model[x]- m_t)*(data_model[x] - m_t);
		}
	}
	s_t_2 = st_temp / (row_max_model + 1) / (col_max_model + 1); 
	//cout << s_t_2 << endl;

	//动态二维数组
	float ** T = new float*[model.rows];
	for (int i = 0; i < model.rows; i++)
	{
		T[i] = new float[model.cols];
	}

	for (int y = 0; y < row_max_model; y++)
	{
		uchar* data_model = model.ptr<uchar>(y);
		for (int x = 0; x < col_max_model; x++)
		{
			
			T[y][x] = (float)((data_model[x] - m_t) / sqrt(s_t_2));
			//cout << "T" << y<< ',' << x << ':' << T[y][x] << endl;
		}
	}

	int row_max_src = src.rows;
	int col_max_src = src.cols;

	for (int y_score = row_max_model/2 ; y_score < row_max_src - row_max_model/2 ; y_score++)
	{
		uchar* data_score = score.ptr<uchar>(y_score);
		for (int x_score = col_max_model/2 ; x_score < col_max_src - col_max_model/2 ; x_score++)
		{

			float ncc_sum=0;
			float mf = 0; float sf_2 = 0;

			//选定区域求mf，sf_2
			int y = y_score;
			int x = x_score;
			for (int i = y - row_max_model / 2; i < y + row_max_model / 2; i++)
			{
				uchar* data_model = src.ptr<uchar>(i);
				for (int j = x - col_max_model / 2; j < x + col_max_model / 2; j++)
				{
					mf += data_model[j];
				}
			}
			mf = mf / (row_max_model + 1) / (col_max_model + 1);

			for (int i = y - row_max_model / 2; i < y + row_max_model / 2; i++)
			{
				uchar* data_model = src.ptr<uchar>(i);
				for (int j = x - col_max_model / 2; j < x + col_max_model / 2; j++)
				{
					sf_2 += (data_model[j]-mf)*(data_model[j] - mf);
				}
			}
			sf_2 = sf_2 / (row_max_model + 1) / (col_max_model + 1);
			//cout << sf_2 << endl;
			//求score
			float score=0;
			int a = 0;
			int b = 0;
			for (int i = y - row_max_model / 2; i < y + row_max_model / 2; i++)
			{
				if (i <= y)
				{
					a = i - (y - row_max_model / 2);
				}
				else
				{
					a = row_max_model/2 + i - y;
				}
				uchar* data_model = src.ptr<uchar>(i);
				for (int j = x - col_max_model / 2; j < x + col_max_model / 2; j++)
				{
					if (j <= x)
					{
						b = j - (x - col_max_model / 2);
					}
					else
					{
						b = col_max_model / 2 + j - x;
					}
					score += T[a][b] * (data_model[j] - mf) / sqrt(sf_2);
				}
			}
			score = score / (row_max_model + 1) / (col_max_model + 1);
			//cout << score << endl;
			data_score[x_score] = abs(score)*255;
		}
	}

	minMaxLoc(score, 0, 0, 0, &max);
	//cout << max.x << "," << max.y << endl;
	circle(score, max, 5, Scalar(255, 255, 0), 2, 8, 0);
	imshow("Score", score);
	//释放空间
	for (int i = 0; i < model.rows; i++)
		delete[]T[i];
	delete[]T;
}


int main()
{
	Mat model, src;
	model = imread("C:/Users/98611/Desktop/model.jpg", 0);
	src = imread("C:/Users/98611/Desktop/yuantu.jpg", 0);
	if ((!src.data)||(!model.data))
	{
		printf("could not load image...\n");
		return -1;
	}
	//cout << src.type() << endl;
	namedWindow("模板", CV_WINDOW_AUTOSIZE);
	imshow("模板", model);//模板显示
	imshow("原图", src);//原图显示
	//匹配位置坐标
	Point position;
	double timeStart1 = (double)getTickCount();  //计算程序运行时间
	ncc(model,src,position);
	double nTime = ((double)getTickCount() - timeStart1) / getTickFrequency();
	cout << "ncc耗时：" << nTime << "秒\n" << endl;
	cout << "匹配位置坐标 "<<" X:" << position.x << "  Y:" << position.y << endl;
	circle(src, position, 5, Scalar(255, 255, 0), 2, 8, 0);
	string position_x = to_string(position.x);
	string position_y = to_string(position.y);
	string time = to_string(nTime);
	std::string text = "  X:"+position_x+"  Y:"+position_y;
	putText(src, text, position, FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 0), 1, 8, 0);

	imshow("结果", src);
	waitKey(0);
	return 0;
}
