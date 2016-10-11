
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\objdetect\objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

#include "CompressiveTracker.h"
#include "FaceDete.h"
using namespace cv;
using namespace std;






Rect box; // tracking object
bool drawing_box = false;
bool gotBB = false;	// got tracking box or not
bool fromfile = false;
string video;

void readBB(char* file)	// get tracking box from file
{
	ifstream tb_file (file);
	string line;
	getline(tb_file, line);
	istringstream linestream(line);
	string x1, y1, w1, h1;
	getline(linestream, x1, ',');
	getline(linestream, y1, ',');
	getline(linestream, w1, ',');
	getline(linestream, h1, ',');
	int x = atoi(x1.c_str());
	int y = atoi(y1.c_str());
	int w = atoi(w1.c_str());
	int h = atoi(h1.c_str());
	box = Rect(x, y, w, h);
}

// tracking box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param)
{
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box)
		{
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0)
		{
			box.x += box.width;
			box.width *= -1;
		}
		if( box.height < 0 )
		{
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	default:
		break;
	}
}

void print_help(void)
{
	printf("use:\n     welcome to use CompressiveTracking\n");
	printf("Kaihua Zhang's paper:Real-Time Compressive Tracking\n");
	printf("C++ implemented by yang xian\nVersion: 1.0\nEmail: yang_xian521@163.com\nDate:	2012/08/03\n\n");
	printf("-v    source video\n-b        tracking box file\n");
}

void read_options(int argc, char** argv, VideoCapture& capture)
{
	for (int i=0; i<argc; i++)
	{
		if (strcmp(argv[i], "-b") == 0)	// read tracking box from file
		{
			if (argc>i)
			{
				readBB(argv[i+1]);
				gotBB = true;
			}
			else
			{
				print_help();
			}
		}
		if (strcmp(argv[i], "-v") == 0)	// read video from file
		{
			if (argc > i)
			{
				video = string(argv[i+1]);
				capture.open(video);
				fromfile = true;
			}
			else
			{
				print_help();
			}
		}
	}
}

//int main(int argc, char * argv[])
//{
//	VideoCapture capture;
//	FaceDetect fd;
//	capture.open(0);
//	// Read options
//	//read_options(argc, argv, capture);
//	// Init camera
//	if (!capture.isOpened())
//	{
//		cout << "capture device failed to open!" << endl;
//		return 1;
//	}
//	//namedWindow("CT", CV_WINDOW_AUTOSIZE);
//	Mat frame;
//	capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
//	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
//	vector<Rect> faces;
//	Mat croppedMat;
//	Mat Image;
//	int p[5];
//	while (faces.size() == 0){
//		capture >> frame;
//
//		cv::CascadeClassifier FaceCascade;
//		FaceCascade.load("../haarcascade_frontalface_alt_cpu.xml");
//
//		
//		frame.copyTo(croppedMat);
//		rectangle(frame, box, Scalar(0, 0, 255));
//
//
//		
//		cvtColor(croppedMat, Image, CV_BGR2GRAY);
//		double t = (double)cvGetTickCount();
//		//FaceCascade.detectMultiScale(croppedMat, faces, 1.1, 2, 0, Size(30, 30));
//		FaceCascade.detectMultiScale(Image, faces, 1.5, 2, 0, Size(30, 30));
//		t = (double)cvGetTickCount() - t;
//		printf("Info10: the Face detect cost time is :%g ms.\n", t / ((double)cvGetTickFrequency()*1000.0));
//		for (int i = 0; i < faces.size(); i++){
//			Rect r = faces[i];
//			//fd.param.est = faces[i];
//			rectangle(croppedMat, faces[i], Scalar(255, 255, 0), 2, 8, 0);
//			printf("Info11:%d face is found ad Rect(%d,%d,%d,%d).\n", i + 1, r.x, r.y, r.width, r.height);
//			p[0] = r.x + r.width*0.5;
//			p[1] = r.y + r.height*0.5;
//			p[2] = r.width;
//			p[3] = r.height;
//			p[4] = 0;
//		}
//	}
//	for (int i = 0; i < 5; i++) std::cout << p[i] << endl;
//	//imshow("FaceDete", croppedMat);
//	//cvWaitKey(5000);
//
//	//capture.release();
//
//	
//	
//	
//	float data[6] = { (float)p[0], (float)p[1], (float)(p[2] / 32.0),(float)p[4], (float)(p[3] / p[2]*1.0), 0.0 };	
//	//Mat para = (Mat_<int>(6, 1) << param0[0], param0[1], param0[2] / 32, 0, param0[3] / 32, 0);
//	//Mat para = Mat(6, 1, CV_32S, data);
//	//cout << para.t() << endl;
//
//	Mat param0 = fd.affparam2mat(data);
//	//cout << paraMat << endl;
//	Mat image = fd.warpimage(Image, param0, Size(32, 32));
//	
//	fd.tmpl.mean = image;
//	fd.tmpl.numsample = 0;
//	fd.tmpl.reseig = 0;
//	int N = 1024;
//	
//	
//	printf("1:*********Initial block image tensor*********\n");
//	vector<Mat> F0;
//
//	double t = (double)cvGetTickCount();
//
//	int updateNFrame = 15;
//	Mat show;
//	for (int i = 0; i < updateNFrame; i++){
//			
//		Mat window = fd.warpimage(Image, param0, Size(32, 32));
//		
//		F0.push_back(window);
//	}
//
//	t = (double)cvGetTickCount() - t;
//	printf("Info15: the detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
//	hconcat(F0, show);
//
//	//imshow("the croppedmat face", show);
//	//cvWaitKey(6000);
//	//cvDestroyAllWindows();
//	/*perform initial svd in the mode-1*/
//	printf("\n2:*********start to the svds*********\n");
//	Mat w, u, vt;
//	Mat A;
//	
//	show.convertTo(A, CV_32F,1.0/255);
//	//cout << A.cols<<" "<<A.rows<< endl;
//	//cout << show.type() << endl;
//	SVD::compute(A, w, u, vt,SVD::MODIFY_A);
//
//	//cout << w << endl;
//	
//	fd.opt = { 200, 0.25, 0.99, 1, { 4, 4, 0.02, 0.02, 0.005, 0.001 } };
//	fd.param.wimg = fd.tmpl.mean;
//
//	//draw track result 
//	
//	//fd.drawtrackresult(false, 0, frame);
//
//	// recircle 100 times
//	int f = 0;
//	fd.param.est = param0;
//	Mat frame1;
//	while(f<10000){
//		
//
//		capture >> frame;
//		cout << "\n3:*********start to update the parameters*********" << endl;
//		
//		cvtColor(frame, frame1, CV_BGR2GRAY);
//		double t = (double)cvGetTickCount();
//		fd.estwarp_condens(frame1);
//
//		t = (double)cvGetTickCount() - t;
//		printf("Info15: the detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
//
//		fd.drawtrackresult(false, f, frame);
//		f++;
//
//	
//	}
//
//
//
//	capture.release();
//	system("pause");
//	return 0;
//}