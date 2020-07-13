#include "stdafx.h"
#include <Windows.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>


#ifdef _DEBUG
#pragma comment(lib, "opencv_world420d.lib") 
#else
#pragma comment(lib, "opencv_world420.lib") 
#endif

//===【鼠标事件回调函数】===
int detectTHD = 900000;		//亮度阈值：有车辆经过的；
std::vector<int>		myLanneLightSum_Last;			//车道亮度和：上一帧的
std::vector<int>		myLanneVihicleCnt;				//车道车辆计数器

std::vector<cv::Rect>		myLanneRect;			//车道矩形框；显示为红色；
std::vector<cv::Point>		myMousePoints;		//鼠标点向量；显示为蓝色；
int	myMouseEventBusy = 0;							//鼠标回调事件忙:简单的资源锁
static void onMouse(int event, int x, int y, int flags, void*)
{
	myMouseEventBusy = 1;
	cv::Point  mPoint;
	cv::Rect mRect;
	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:		//左键按下:增加myMousePoints中的点数
		mPoint = cv::Point(x, y);
		myMousePoints.push_back(mPoint);	//将当前鼠标点推送到向量中；
		if (myMousePoints.size() > 4)
			myMousePoints.erase(myMousePoints.begin());	//保证myMousePoints向量中节点数不大于4；
		break;
	case cv::EVENT_RBUTTONDOWN:		//右键键按下：将myMousePoints中的4个点推送到矩形框向量myLanneRect
		if (myMousePoints.size() == 4)
		{
			int Xmin = 100000; int Ymin = 100000;
			int Xmax = 0; int Ymax = 0;
			for (int k = 0; k < 4; k++)
			{
				Xmin = std::min(Xmin, myMousePoints.at(k).x);
				Ymin = std::min(Ymin, myMousePoints.at(k).y);
				Xmax = std::max(Xmax, myMousePoints.at(k).x);
				Ymax = std::max(Ymax, myMousePoints.at(k).y);
			}//for k   <<< === 用四个点构成矩形框的参数
			mRect = cv::Rect(Xmin, Ymin, Xmax - Xmin, Ymax - Ymin);		//构成矩形框
			myLanneRect.push_back(mRect);
			myLanneLightSum_Last.push_back(0);
			myLanneVihicleCnt.push_back(0);
			myMousePoints.clear();  //清除鼠标点向量

		}///if
		break;
	case cv::EVENT_MBUTTONDOWN:		//中间键键按下：删除myMousePoints中的一个点；myMousePoints为空时，删除myLanneRect中的节点；
		printf("EVENT_MBUTTONDOWN\n");
		if (myMousePoints.size() > 0)
			myMousePoints.pop_back();
		else
		{
			myLanneRect.pop_back();
			myLanneLightSum_Last.pop_back();
			myLanneVihicleCnt.pop_back();
		}
		break;

	}/////switch
myMouseEventBusy = 0;
return;
}

int main(int argc, char* argv[])
{
	char errorMSG[256];
	char curPathName[384] = ""; char curModulerPath[384] = "";
	GetCurrentDirectory(383, curModulerPath); 
	printf("Line39: curModulerPath = %s\n", curModulerPath);

	//=======读取标记的矩形框文件内容到myLanneRect：=======
	#ifndef READ_RECT_FILE
		FILE *pFILE = fopen("MarkRect.txt", "r");
		if (pFILE != NULL)
		{
			cv::Rect  mRect;
			while (fgets(errorMSG, 255, pFILE) != NULL)
			{
				int rtn = sscanf(errorMSG, "%d %d %d %d", &mRect.x, &mRect.y, &mRect.width, &mRect.height);
				if (rtn == 4) {
					myLanneRect.push_back(mRect);
					myLanneLightSum_Last.push_back(0);
					myLanneVihicleCnt.push_back(0);
					}
			}
			fclose(pFILE);
		}///if
	#endif // !READ_RECT_FILE

	std::string  imgName = "video-02.mp4";
	char FilePath[384];
	if (strlen(curPathName) > 0)
		sprintf(FilePath, "%s\\%s", curPathName, imgName.c_str());	//图片文件路径
	else
		sprintf(FilePath, "%s", imgName.c_str());	//图片文件路径

	//==【01】== 打开视频文件或摄像头
		cv::VideoCapture cap; //VideoCapture类实例化，使用缺省摄像头

		if (0 && "UsingCam")
			cap.open(0);
		else
			cap.open(FilePath);
		if (!cap.isOpened()) // check if we succeeded
		{
			printf("error#73: 打开设备或文件失败，检查是否存在！回车退出！\n路径=%s\n", FilePath);
			fgets(FilePath, 127, stdin);
			return -1;
		}
		cv::Mat frame, newframe, greyFrame, floatFrame, lastFrame, frame2, mog2RES, KNN, out_frame, avgFrame;
		std::vector<cv::Mat> diffIMGvec;

	//==【02】== 创建运动视频背景提取对象：用于分离背景和运动对象
		cv::Ptr<cv::BackgroundSubtractorMOG2> bgMOG2 = cv::createBackgroundSubtractorMOG2();
		cv::Ptr<cv::BackgroundSubtractorKNN> bgKNN = cv::createBackgroundSubtractorKNN();
		bgMOG2->setVarThreshold(30);
		bool update_bg_model = true;
	//==【03】== 命名几个显示窗口
		cv::namedWindow("RawWnd", cv::WINDOW_NORMAL);
		cv::setMouseCallback("RawWnd", onMouse, &newframe);		//设置鼠标事件回调函数（"RawWnd"窗口的）：同时传递彩色图像指针；
		cv::namedWindow("Out_KNN", cv::WINDOW_NORMAL);
		cv::namedWindow("Out_MOG2", cv::WINDOW_NORMAL);

	int frameNums = 0;
	for (;;)
	{
		frame.rows = 0;
		double t1 = (double)cv::getCPUTickCount();  //开始统计时间
		cap.read(frame);
		if (frame.rows == 0)
			break;
		cv::Size newSize(frame.cols / 2, frame.rows / 2);  //压缩图像，将其尺寸缩小
		cv::resize(frame, newframe, newSize);
		cv::cvtColor(newframe, greyFrame, cv::COLOR_RGB2GRAY);  //转换为灰度图
		cv::blur(greyFrame, greyFrame, cv::Size(3, 3));		//使用平滑运算
	
		double t2 = (double)cv::getCPUTickCount();
		bgMOG2->apply(greyFrame, mog2RES, update_bg_model ? -1 : 0);   //使用MOG2算法提取前景
		double t3 = (double)cv::getCPUTickCount();  //获取处理时间

		double t4 = (double)cv::getCPUTickCount();
		bgKNN->apply(greyFrame, KNN, update_bg_model ? -1 : 0);			//使用KNN算法提取前景
		double t5 = (double)cv::getCPUTickCount();  //获取处理时间
		printf("MOG2 Time is %.3fms\n", 1e0 * (t3 - t2) / (double)cv::getTickFrequency());
		printf("KNN Time is %.3fms\n", 1e0 * (t5 - t4) / (double)cv::getTickFrequency());
		printf("Total Time is %.3fms\n", 1e0 * (t5 - t1) / (double)cv::getTickFrequency());
		//printf("--------------------\n");

		if (!mog2RES.empty())  //计算MOG2算法下矩形框的积分亮度值
		{
			cv::Mat showMat;
			mog2RES.copyTo(showMat);
			if (myMouseEventBusy == 0)
			{
				for (int k = 0; k < myLanneRect.size(); k++)
				{

					cv::rectangle(showMat, myLanneRect.at(k), cv::Scalar(255, 255, 255), 3);
					cv::Mat subMat = mog2RES(myLanneRect.at(k));	//再MOG2的前景提取结果中，取车道标记矩形框区域为subMat矩阵
					cv::Mat sumMat;		//积分图 == subMat的积分矩阵
					cv::integral(subMat, sumMat, CV_32S);		//设置积分矩阵的数据类型为uint；
					int sumValue = (int)sumMat.at<int>((int)sumMat.rows - 1, (int)sumMat.cols - 1);  //获取积分图右下角的值，就是矩形框内亮度和；
					sprintf(errorMSG, "sum = %d;", sumValue);
					cv::putText(showMat, errorMSG, cv::Point(myLanneRect.at(k).x, myLanneRect.at(k).y + 4), 0.2, 1, cv::Scalar(255, 0, 0), 2);//显示矩形框内的亮度和；

				}//for k
			}////if
			cv::imshow("Out_MOG2", showMat);

		}

		if (!KNN.empty())   //计算KNN算法下矩形框的积分亮度值
		{
			cv::Mat showMat;
			KNN.copyTo(showMat);
			if (myMouseEventBusy == 0)
			{
				for (int k = 0; k < myLanneRect.size(); k++)
				{

					cv::rectangle(showMat, myLanneRect.at(k), cv::Scalar(255, 255, 255), 3);
					cv::Mat subMat = KNN(myLanneRect.at(k));	//再KNN的前景提取结果中，取车道标记矩形框区域为subMat矩阵
					cv::Mat sumMat;		//积分图 == subMat的积分矩阵
					cv::integral(subMat, sumMat, CV_32S);		//设置积分矩阵的数据类型为uint；
					int sumValue = (int)sumMat.at<int>((int)sumMat.rows - 1, (int)sumMat.cols - 1);  //获取积分图右下角的值，就是矩形框内亮度和；
					sprintf(errorMSG, "sum = %d;", sumValue);
					cv::putText(showMat, errorMSG, cv::Point(myLanneRect.at(k).x, myLanneRect.at(k).y + 4), 0.2, 1, cv::Scalar(255, 0, 0), 2);//显示矩形框内的亮度和；

				}//for k
			}////if
			imshow("Out_KNN", showMat);
		}
			

		//===>>> 显示原始图像：显示车道标记信息 + 矩形框内亮度和 + 车流量统计
		#ifndef SHOW_RAW_MAT
			cv::Mat showMat;
			newframe.copyTo(showMat); //矩阵复制
			sprintf(errorMSG, "mL=add Point; mR=add Rect; mM=delete Point;");
			cv::putText(showMat, errorMSG, cv::Point(8, 32), 0.2, 1, cv::Scalar(255, 0, 0), 2);//显示提示信息；
			//==>> 显示车道矩形框为红色 + 车流量统计 + 车流量显示
			if (myMouseEventBusy == 0)
			{
				for (int k = 0; k < myLanneRect.size(); k++)
				{
					cv::rectangle(showMat, myLanneRect.at(k), cv::Scalar(0, 0, 255), 3);
					cv::Mat subMat = mog2RES(myLanneRect.at(k));	//再MOG2的前景提取结果中，取车道标记矩形框区域为subMat矩阵
					cv::Mat sumMat;		//积分图 == subMat的积分矩阵
					cv::integral(subMat, sumMat, CV_32S);		//设置积分矩阵的数据类型为int，计算车道矩形框内亮度积分图；
					int sumValue = (int)sumMat.at<int>((int)sumMat.rows - 1, (int)sumMat.cols - 1);  //获取积分图右下角的值，就是矩形框内亮度和；
					sprintf(errorMSG, "sum = %d;", sumValue);
					cv::putText(showMat, errorMSG, cv::Point(myLanneRect.at(k).x, myLanneRect.at(k).y + 4), 0.2, 1, cv::Scalar(255, 255, 0), 2);//显示矩形框内的亮度和；
					//===>>> 车流量统计：
					if (myLanneLightSum_Last.at(k) > detectTHD && sumValue <= detectTHD)
					{
						//:: 车辆通过了矩形框：上一帧亮度和大于阈值，本帧亮度和小于阈值；车辆计数器自加；
						myLanneVihicleCnt.at(k)++;
						myLanneLightSum_Last.at(k) = sumValue;
					}
					else 
						myLanneLightSum_Last.at(k) = sumValue;  //存储当前亮度和到myLanneLightSum_Last				
				}//for k

				//===>> 车流量统计结果显示
				cv::Mat topareaMat = showMat(cv::Rect(0, 0, showMat.cols, 75));		//最顶部48行置0；
				topareaMat *= 255;

				std::string strVihicleCnt = "VihicleCnt: ";
				for (int k = 0; k < myLanneRect.size(); k++)
				{
					sprintf(errorMSG, "L%d = %d;", k, myLanneVihicleCnt.at(k));
					strVihicleCnt += errorMSG;
				}
				cv::putText(showMat, strVihicleCnt.c_str(), cv::Point(8, 64), 0.2, 1, cv::Scalar(0, 0, 255), 2); //流量统计显示到彩色图片上

			}////if
			 //==>> 显示正在标记的坐标点为蓝色：
			if (myMouseEventBusy == 0)
			{
				for (int k = 1; k < myMousePoints.size(); k++)
				{
					cv::line(showMat, myMousePoints.at(k - 1), myMousePoints.at(k), cv::Scalar(255, 0, 0), 15);
				}//for k
				if(myMousePoints.size() == 4)
					cv::line(showMat, myMousePoints.at(0), myMousePoints.at(3), cv::Scalar(255, 0, 0), 2);
			}////if


			imshow("RawWnd", showMat);
		#endif // SHOW_RAW_MAT
		int keycode = cv::waitKey(100);		//等待100ms
		if (keycode == 'q')
			break;
		else if (keycode == ' ')
		{
			update_bg_model = !update_bg_model;
			printf("Learn background is in state = %d\n", update_bg_model);
		}
		else if (keycode == 'w')
		{
			//写文件：记录标记的矩形框到文件中：
			#ifndef WRITE_RECT_FILE
			FILE *pFILE = fopen("MarkRect.txt", "w");
			if (pFILE != NULL)
			{
				for (int k = 0; k < myLanneRect.size(); k++) {
					fprintf(pFILE, "%d %d %d %d\n", myLanneRect.at(k).x, myLanneRect.at(k).y, myLanneRect.at(k).width, myLanneRect.at(k).height);
				}
				fclose(pFILE);
			}///if
			#endif // !WRITE_RECT_FILE

		}
		frameNums++;
		Sleep(50);
	}//for 
	cap.release();
	return 0;
}

