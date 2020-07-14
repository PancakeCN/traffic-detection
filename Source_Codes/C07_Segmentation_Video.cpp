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

//===������¼��ص�������===
int detectTHD = 900000;		//������ֵ���г��������ģ�
std::vector<int>		myLanneLightSum_Last;			//�������Ⱥͣ���һ֡��
std::vector<int>		myLanneVihicleCnt;				//��������������

std::vector<cv::Rect>		myLanneRect;			//�������ο���ʾΪ��ɫ��
std::vector<cv::Point>		myMousePoints;		//������������ʾΪ��ɫ��
int	myMouseEventBusy = 0;							//���ص��¼�æ:�򵥵���Դ��
static void onMouse(int event, int x, int y, int flags, void*)
{
	myMouseEventBusy = 1;
	cv::Point  mPoint;
	cv::Rect mRect;
	switch (event)
	{
	case cv::EVENT_LBUTTONDOWN:		//�������:����myMousePoints�еĵ���
		mPoint = cv::Point(x, y);
		myMousePoints.push_back(mPoint);	//����ǰ�������͵������У�
		if (myMousePoints.size() > 4)
			myMousePoints.erase(myMousePoints.begin());	//��֤myMousePoints�����нڵ���������4��
		break;
	case cv::EVENT_RBUTTONDOWN:		//�Ҽ������£���myMousePoints�е�4�������͵����ο�����myLanneRect
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
			}//for k   <<< === ���ĸ��㹹�ɾ��ο�Ĳ���
			mRect = cv::Rect(Xmin, Ymin, Xmax - Xmin, Ymax - Ymin);		//���ɾ��ο�
			myLanneRect.push_back(mRect);
			myLanneLightSum_Last.push_back(0);
			myLanneVihicleCnt.push_back(0);
			myMousePoints.clear();  //�����������

		}///if
		break;
	case cv::EVENT_MBUTTONDOWN:		//�м�������£�ɾ��myMousePoints�е�һ���㣻myMousePointsΪ��ʱ��ɾ��myLanneRect�еĽڵ㣻
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

	//=======��ȡ��ǵľ��ο��ļ����ݵ�myLanneRect��=======
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
		sprintf(FilePath, "%s\\%s", curPathName, imgName.c_str());	//ͼƬ�ļ�·��
	else
		sprintf(FilePath, "%s", imgName.c_str());	//ͼƬ�ļ�·��

	//==��01��== ����Ƶ�ļ�������ͷ
		cv::VideoCapture cap; //VideoCapture��ʵ������ʹ��ȱʡ����ͷ

		if (0 && "UsingCam")
			cap.open(0);
		else
			cap.open(FilePath);
		if (!cap.isOpened()) // check if we succeeded
		{
			printf("error#73: ���豸���ļ�ʧ�ܣ�����Ƿ���ڣ��س��˳���\n·��=%s\n", FilePath);
			fgets(FilePath, 127, stdin);
			return -1;
		}
		cv::Mat frame, newframe, greyFrame, floatFrame, lastFrame, frame2, mog2RES, KNN, out_frame, avgFrame;
		std::vector<cv::Mat> diffIMGvec;

	//==��02��== �����˶���Ƶ������ȡ�������ڷ��뱳�����˶�����
		cv::Ptr<cv::BackgroundSubtractorMOG2> bgMOG2 = cv::createBackgroundSubtractorMOG2();
		cv::Ptr<cv::BackgroundSubtractorKNN> bgKNN = cv::createBackgroundSubtractorKNN();
		bgMOG2->setVarThreshold(30);
		bool update_bg_model = true;
	//==��03��== ����������ʾ����
		cv::namedWindow("RawWnd", cv::WINDOW_NORMAL);
		cv::setMouseCallback("RawWnd", onMouse, &newframe);		//��������¼��ص�������"RawWnd"���ڵģ���ͬʱ���ݲ�ɫͼ��ָ�룻
		cv::namedWindow("Out_KNN", cv::WINDOW_NORMAL);
		cv::namedWindow("Out_MOG2", cv::WINDOW_NORMAL);

	int frameNums = 0;
	for (;;)
	{
		frame.rows = 0;
		double t1 = (double)cv::getCPUTickCount();  //��ʼͳ��ʱ��
		cap.read(frame);
		if (frame.rows == 0)
			break;
		cv::Size newSize(frame.cols / 2, frame.rows / 2);  //ѹ��ͼ�񣬽���ߴ���С
		cv::resize(frame, newframe, newSize);
		cv::cvtColor(newframe, greyFrame, cv::COLOR_RGB2GRAY);  //ת��Ϊ�Ҷ�ͼ
		cv::blur(greyFrame, greyFrame, cv::Size(3, 3));		//ʹ��ƽ������
	
		double t2 = (double)cv::getCPUTickCount();
		bgMOG2->apply(greyFrame, mog2RES, update_bg_model ? -1 : 0);   //ʹ��MOG2�㷨��ȡǰ��
		double t3 = (double)cv::getCPUTickCount();  //��ȡ����ʱ��

		double t4 = (double)cv::getCPUTickCount();
		bgKNN->apply(greyFrame, KNN, update_bg_model ? -1 : 0);			//ʹ��KNN�㷨��ȡǰ��
		double t5 = (double)cv::getCPUTickCount();  //��ȡ����ʱ��
		printf("MOG2 Time is %.3fms\n", 1e0 * (t3 - t2) / (double)cv::getTickFrequency());
		printf("KNN Time is %.3fms\n", 1e0 * (t5 - t4) / (double)cv::getTickFrequency());
		printf("Total Time is %.3fms\n", 1e0 * (t5 - t1) / (double)cv::getTickFrequency());
		//printf("--------------------\n");

		if (!mog2RES.empty())  //����MOG2�㷨�¾��ο�Ļ�������ֵ
		{
			cv::Mat showMat;
			mog2RES.copyTo(showMat);
			if (myMouseEventBusy == 0)
			{
				for (int k = 0; k < myLanneRect.size(); k++)
				{

					cv::rectangle(showMat, myLanneRect.at(k), cv::Scalar(255, 255, 255), 3);
					cv::Mat subMat = mog2RES(myLanneRect.at(k));	//��MOG2��ǰ����ȡ����У�ȡ������Ǿ��ο�����ΪsubMat����
					cv::Mat sumMat;		//����ͼ == subMat�Ļ��־���
					cv::integral(subMat, sumMat, CV_32S);		//���û��־������������Ϊuint��
					int sumValue = (int)sumMat.at<int>((int)sumMat.rows - 1, (int)sumMat.cols - 1);  //��ȡ����ͼ���½ǵ�ֵ�����Ǿ��ο������Ⱥͣ�
					sprintf(errorMSG, "sum = %d;", sumValue);
					cv::putText(showMat, errorMSG, cv::Point(myLanneRect.at(k).x, myLanneRect.at(k).y + 4), 0.2, 1, cv::Scalar(255, 0, 0), 2);//��ʾ���ο��ڵ����Ⱥͣ�

				}//for k
			}////if
			cv::imshow("Out_MOG2", showMat);

		}

		if (!KNN.empty())   //����KNN�㷨�¾��ο�Ļ�������ֵ
		{
			cv::Mat showMat;
			KNN.copyTo(showMat);
			if (myMouseEventBusy == 0)
			{
				for (int k = 0; k < myLanneRect.size(); k++)
				{

					cv::rectangle(showMat, myLanneRect.at(k), cv::Scalar(255, 255, 255), 3);
					cv::Mat subMat = KNN(myLanneRect.at(k));	//��KNN��ǰ����ȡ����У�ȡ������Ǿ��ο�����ΪsubMat����
					cv::Mat sumMat;		//����ͼ == subMat�Ļ��־���
					cv::integral(subMat, sumMat, CV_32S);		//���û��־������������Ϊuint��
					int sumValue = (int)sumMat.at<int>((int)sumMat.rows - 1, (int)sumMat.cols - 1);  //��ȡ����ͼ���½ǵ�ֵ�����Ǿ��ο������Ⱥͣ�
					sprintf(errorMSG, "sum = %d;", sumValue);
					cv::putText(showMat, errorMSG, cv::Point(myLanneRect.at(k).x, myLanneRect.at(k).y + 4), 0.2, 1, cv::Scalar(255, 0, 0), 2);//��ʾ���ο��ڵ����Ⱥͣ�

				}//for k
			}////if
			imshow("Out_KNN", showMat);
		}
			

		//===>>> ��ʾԭʼͼ����ʾ���������Ϣ + ���ο������Ⱥ� + ������ͳ��
		#ifndef SHOW_RAW_MAT
			cv::Mat showMat;
			newframe.copyTo(showMat); //������
			sprintf(errorMSG, "mL=add Point; mR=add Rect; mM=delete Point;");
			cv::putText(showMat, errorMSG, cv::Point(8, 32), 0.2, 1, cv::Scalar(255, 0, 0), 2);//��ʾ��ʾ��Ϣ��
			//==>> ��ʾ�������ο�Ϊ��ɫ + ������ͳ�� + ��������ʾ
			if (myMouseEventBusy == 0)
			{
				for (int k = 0; k < myLanneRect.size(); k++)
				{
					cv::rectangle(showMat, myLanneRect.at(k), cv::Scalar(0, 0, 255), 3);
					cv::Mat subMat = mog2RES(myLanneRect.at(k));	//��MOG2��ǰ����ȡ����У�ȡ������Ǿ��ο�����ΪsubMat����
					cv::Mat sumMat;		//����ͼ == subMat�Ļ��־���
					cv::integral(subMat, sumMat, CV_32S);		//���û��־������������Ϊint�����㳵�����ο������Ȼ���ͼ��
					int sumValue = (int)sumMat.at<int>((int)sumMat.rows - 1, (int)sumMat.cols - 1);  //��ȡ����ͼ���½ǵ�ֵ�����Ǿ��ο������Ⱥͣ�
					sprintf(errorMSG, "sum = %d;", sumValue);
					cv::putText(showMat, errorMSG, cv::Point(myLanneRect.at(k).x, myLanneRect.at(k).y + 4), 0.2, 1, cv::Scalar(255, 255, 0), 2);//��ʾ���ο��ڵ����Ⱥͣ�
					//===>>> ������ͳ�ƣ�
					if (myLanneLightSum_Last.at(k) > detectTHD && sumValue <= detectTHD)
					{
						//:: ����ͨ���˾��ο���һ֡���Ⱥʹ�����ֵ����֡���Ⱥ�С����ֵ�������������Լӣ�
						myLanneVihicleCnt.at(k)++;
						myLanneLightSum_Last.at(k) = sumValue;
					}
					else 
						myLanneLightSum_Last.at(k) = sumValue;  //�洢��ǰ���Ⱥ͵�myLanneLightSum_Last				
				}//for k

				//===>> ������ͳ�ƽ����ʾ
				cv::Mat topareaMat = showMat(cv::Rect(0, 0, showMat.cols, 75));		//���48����0��
				topareaMat *= 255;

				std::string strVihicleCnt = "VihicleCnt: ";
				for (int k = 0; k < myLanneRect.size(); k++)
				{
					sprintf(errorMSG, "L%d = %d;", k, myLanneVihicleCnt.at(k));
					strVihicleCnt += errorMSG;
				}
				cv::putText(showMat, strVihicleCnt.c_str(), cv::Point(8, 64), 0.2, 1, cv::Scalar(0, 0, 255), 2); //����ͳ����ʾ����ɫͼƬ��

			}////if
			 //==>> ��ʾ���ڱ�ǵ������Ϊ��ɫ��
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
		int keycode = cv::waitKey(100);		//�ȴ�100ms
		if (keycode == 'q')
			break;
		else if (keycode == ' ')
		{
			update_bg_model = !update_bg_model;
			printf("Learn background is in state = %d\n", update_bg_model);
		}
		else if (keycode == 'w')
		{
			//д�ļ�����¼��ǵľ��ο��ļ��У�
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

