// ConsoleApplication1.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
//src 原图像  dst滤波后图像  ksize滤波器大小  flag=0时用定义滤波  flag=1时用分离滤波器
void MeanFilter(Mat src,Mat &dst, Size ksize,int flag)
{
	int row_max = src.rows;
	int col_max = src.cols;
	src.copyTo(dst);
	//3*3滤波，定义
	if ((ksize.height == 3 )&& (flag == 0))
	{
		double timeStart1 = (double)getTickCount();  //计算程序运行时间
		for (int y = 1; y < row_max - 1; y++)
		{
			uchar* data_src = src.ptr<uchar>(y);
			uchar* data_dst = dst.ptr<uchar>(y);
			uchar* data_src_up = src.ptr<uchar>(y - 1);
			uchar* data_src_down = src.ptr<uchar>(y + 1);
			for (int x = 1; x < col_max-1; x++)
			{
				data_dst[x] = ((data_src[x] + data_src[x - 1] + data_src[x + 1] + 
					data_src_up[x] + data_src_up[x - 1] + data_src_up[x + 1] + 
					data_src_down[x] + data_src_down[x - 1] + data_src_down[x + 1]) / 9);
			}
		}
		double nTime = ((double)getTickCount() - timeStart1) / getTickFrequency();
		cout << "定义均值滤波3*3耗时：" << nTime << "秒\n" << endl;
	}
	//3*3滤波,分离滤波器，先行后列
	if ((ksize.height == 3) && (flag == 1))
	{
		double timeStart2 = (double)getTickCount();  //计算程序运行时间
		for (int y = 1; y < row_max - 1; y++)
		{
			uchar* data_src = src.ptr<uchar>(y);
			uchar* data_dst = dst.ptr<uchar>(y);
			uchar* data_src_up1 = src.ptr<uchar>(y - 1);
			uchar* data_src_down1 = src.ptr<uchar>(y + 1);
			for (int x = 1; x < col_max - 1; x++)
			{
				int data_src_sum =  data_src[x] + data_src[x - 1] + data_src[x + 1];
				int data_src_up1_sum = data_src_up1[x] + data_src_up1[x - 1] + data_src_up1[x + 1];
				int data_src_down1_sum = data_src_down1[x] + data_src_down1[x - 1] + data_src_down1[x + 1];
				data_dst[x] = ((data_src_up1_sum + data_src_down1_sum  + data_src_sum) /9);
			}
		}
		double nTime = ((double)getTickCount() - timeStart2) / getTickFrequency();
		cout << "分离滤波器均值滤波3*3耗时：" << nTime << "秒\n" << endl;
	}
	//5*5滤波，定义
	if ((ksize.height == 5) && (flag == 0))
	{
		double timeStart3 = (double)getTickCount();
		for (int y = 2; y < row_max - 2; y++)
		{
			uchar* data_src = src.ptr<uchar>(y);
			uchar* data_src_up2 = src.ptr<uchar>(y - 2);
			uchar* data_src_up1 = src.ptr<uchar>(y - 1);
			uchar* data_src_down1 = src.ptr<uchar>(y + 1);
			uchar* data_src_down2 = src.ptr<uchar>(y + 2);
			uchar* data_dst = dst.ptr<uchar>(y);
			for (int x = 2; x < col_max - 2; x++)
			{
				data_dst[x] = ((data_src[x] + data_src[x - 1] + data_src[x + 1] + data_src[x-2] + data_src[x +2] + 
					data_src_up1[x] + data_src_up1[x-1] + data_src_up1[x+1] + data_src_up1[x-2] + data_src_up1[x+2] + 
					data_src_up2[x] + data_src_up2[x - 1] + data_src_up2[x + 1] + data_src_up2[x - 2] + data_src_up2[x + 2]+ 
					data_src_down1[x] + data_src_down1[x - 1] + data_src_down1[x + 1] + data_src_down1[x - 2] + data_src_down1[x + 2]+ 
					data_src_down2[x] + data_src_down2[x - 1] + data_src_down2[x + 1] + data_src_down2[x - 2] + data_src_down2[x + 2]) / 25);
			}
		}
		double nTime = ((double)getTickCount() - timeStart3) / getTickFrequency();
		cout << "定义均值滤波5*5耗时：" << nTime << "秒\n" << endl;
	}
	//5*5滤波,分离滤波器，先行后列
	if ((ksize.height == 5 )&& (flag == 1))
	{
		double timeStart4 = (double)getTickCount();
		for (int y = 2; y < row_max - 2; y++)
		{
			uchar* data_src = src.ptr<uchar>(y);
			uchar* data_src_up2 = src.ptr<uchar>(y - 2);
			uchar* data_src_up1 = src.ptr<uchar>(y - 1);
			uchar* data_src_down1 = src.ptr<uchar>(y + 1);
			uchar* data_src_down2 = src.ptr<uchar>(y + 2);
			uchar* data_dst = dst.ptr<uchar>(y);
			for (int x = 2; x < col_max-2; x++)
			{

				int data_src_sum = data_src[x] + data_src[x + 1] + data_src[x + 2] + data_src[x - 1] + data_src[x - 2];
				int data_src_down2_sum = data_src_down2[x] + data_src_down2[x + 1] + data_src_down2[x + 2] + data_src_down2[x - 1] + data_src_down2[x - 2];
				int data_src_down1_sum = data_src_down1[x] + data_src_down1[x + 1] + data_src_down1[x + 2] + data_src_down1[x - 1] + data_src_down1[x - 2];
				int data_src_up2_sum = data_src_up2[x] + data_src_up2[x + 1] + data_src_up2[x + 2] + data_src_up2[x - 1] + data_src_up2[x - 2];
				int data_src_up1_sum = data_src_up1[x] + data_src_up1[x + 1]+ data_src_up1[x + 2]+ data_src_up1[x - 1]+ data_src_up1[x -2];
				data_dst[x] = ((data_src_up1_sum + data_src_up2_sum + data_src_down1_sum + data_src_down2_sum + data_src_sum) / 25);
			}
		}
		double nTime = ((double)getTickCount() - timeStart4) / getTickFrequency();
		cout << "分离滤波器均值滤波5*5耗时：" << nTime << "秒\n" << endl;
	}
}

int main()
{
	Mat src,dst;
	src = imread("C:/Users/98611/Desktop/1.png",0);
	if (!src.data)
	{
		printf("could not load image3...\n");
		return -1;
	}
	//cout << src.type() << endl;
	namedWindow("输入图片", CV_WINDOW_AUTOSIZE);
	imshow("输入图片", src);//原图显示

	double timeStart1 = (double)getTickCount();
	blur(src, dst, Size(3, 3), Point(-1, -1));
	double nTime1 = ((double)getTickCount() - timeStart1) / getTickFrequency();
	cout << "opencv均值滤波3*3耗时：" << nTime1 << "秒\n" << endl;
	imshow("opencv均值滤波3*3", dst);
	double timeStart2 = (double)getTickCount();
	blur(src, dst, Size(5, 5), Point(-1, -1));
	double nTime2 = ((double)getTickCount() - timeStart2) / getTickFrequency();
	cout << "opencv均值滤波5*5耗时：" << nTime2 << "秒\n" << endl;
	imshow("opencv均值滤波5*5", dst);


	Mat dst33;
	MeanFilter(src, dst33, Size(3, 3),0);
    //namedWindow("定义均值滤波3*3", CV_WINDOW_AUTOSIZE);
	imshow("定义均值滤波3*3", dst33);
	MeanFilter(src, dst33, Size(3, 3), 1);
	//namedWindow("分离滤波器均值滤波3*3", CV_WINDOW_AUTOSIZE);
	imshow("分离滤波器均值滤波3*3", dst33);

	Mat dst55;
	MeanFilter(src, dst55, Size(5, 5), 0);
	//namedWindow("定义均值滤波5*5", CV_WINDOW_AUTOSIZE);
	imshow("定义均值滤波5*5", dst55);
	MeanFilter(src, dst55, Size(5, 5),1);
	//namedWindow("分离滤波器均值滤波5*5", CV_WINDOW_AUTOSIZE);
	imshow("分离滤波器均值滤波5*5", dst55);
	waitKey(0);
	return 0;
}

