#pragma once
#include "CompressiveTracker.h"
using namespace cv;

class FaceDetect
{
public:
	FaceDetect();
	~FaceDetect();

	struct Opt{

		int numsample;
		double condenssig;
		double ff;
		int batchsize;
		float affsig[6];

	}opt;;// = { 1500, 0.25, 0.99, 1, { 4, 4, 0.02, 0.02, 0.005, 0.001 } };

	struct  Tmpl
	{
		Mat mean;
		Mat basis;
		Mat eigval;
		int numsample;
		int reseig;

	}tmpl;

	struct Param{

		Mat est;
		Mat wimg;
		Mat param;
		Mat conf;
		Mat err;
		Mat recon;
		Mat coef;

	}param;


	struct Drawopt
	{
		Rect frm;
		Rect window;
		Rect basis;
		int showcoef;
		int matcoef;
		int showcondens;
		int thcondens;

	}drawopt;

	string wchar_to_string(char *wchar);
	
	void meshgrid(const cv::Mat &xgv, const cv::Mat &ygv, cv::Mat1i &X, cv::Mat1i &Y);
	cv::Mat DetectNose(cv::CascadeClassifier cascade, cv::Mat img, cv::Mat Image);
	cv::Mat FaceDetection(cv::CascadeClassifier FaceCascade, cv::Mat croppedMat);
	cv::Mat face_detect_draw(cv::CascadeClassifier FaceCascade, cv::Mat croppedMat);
	void FaceDetect::FaceDeteForTracking(cv::CascadeClassifier FaceCascade, cv::Mat &croppedMat, std::vector<cv::Rect> &faces);
	
	Mat FaceDetect::getInitAffineMat(int a[]);
	Mat getInitAffineMat(cv::Mat M0, float a[][3]);


	void FaceDetect::drawtrackresult(bool flags, int fno, Mat frame);

	Mat FaceDetect::warpimage(Mat image, Mat para, Size sz);
	
	// this function convert the 'geometric' affine parameter to matrix form(2x3)
	Mat FaceDetect::affparam2mat(float *p);
	// this function convert the 'geometric' affine parameter from a matrix form(2x3)
	void FaceDetect::affparam2geom(Mat p, float* &q);

	// key function
	void FaceDetect::estwarp_condens(Mat frm);


	//sequential Karhunen-Loeve algorithm
	void FaceDetect::mySklm(Mat oldTensor, Mat newframe, int k);

	void FaceDetect::QRDecomp(Mat& m, Mat& Q, Mat& R);
private:




};

