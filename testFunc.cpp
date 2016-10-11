#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

#include "CompressiveTracker.h"
#include "FaceDete.h"
#include "FaceDete_CL.h"


#include <opencv2/ocl/ocl.hpp>
#include <arrayfire.h>

using namespace cv;
using namespace std;
using namespace ocl;


//int main(int argc, char **argv){
//
//
//
//	//Mat p = (Mat_<float>(6, 1) << 469.3984, 284.4926, 8.5796, -0.1701, 0.1701, 8.5796);
//	//Mat p = (Mat_<float>(6, 1) << 494.1155, 283.3942, 8.5239, -0.1330, 0.1330, 8.5239);
//	Mat p = (Mat_<float>(6, 1) << 595.8101, 287.0912, 6.5823, -1.7038, 1.7038, 6.5823);
//	FaceDetect ff;
//	
//	float* q  = (float *)malloc(sizeof(float *)*6);
//	double t = (double)cvGetTickCount();
//	for (int i = 0; i < 1500; i++){
//		ff.affparam2geom(p,q);
//	}
//	t = (double)cvGetTickCount() - t;
//	printf("Info5: the detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
//	for (int i = 0; i < 6; i++) cout << q[i]<<endl;
//	
//	delete q;
//
//	
//
//	//Mat M(6, 1, CV_32F);
//	//float *a = new float[6];
//
//	//for (int i = 0; i < M.cols; i++){
//	//	for (int j = 0; j < M.rows; j++){
//	//		M.at<float>(j, i) = (float)(2 * j + i+1);
//	//	}
//	//}
//
//	//cout << M << endl;
//
//
//	//FaceDetect ff;
//	//ff.affparam2geom(M, a);
//	//for (int i = 0; i < 6; i++) cout << a[i] << endl;
//
//	//Mat b(1500, 1500, CV_32F);
//
//	//float *a = new float[6];
//
//	//for (int i = 0; i < b.rows; i++){
//	//	for (int j = 0; j < b.cols; j++){
//	//		b.at<float>(i, j) = (float)(2 * i + j);
//	//	}
//	//}
//
//	//Mat cumconf(b.rows, b.cols, CV_32F);
//
//
//
//	//int nr = b.rows;
//	//int nc = b.cols;
//	//Mat outb;
//	//outb = b.clone();
//	//if (b.isContinuous() && outb.isContinuous())
//	//{
//	//	nr = 1;
//	//	nc = nc*b.rows*b.channels();
//	//}
//	//double t = (double)cvGetTickCount();//get the runtime time count;
//
//	//for (int j = 0; j < b.cols; j++)
//	//{
//	//	float *pt = b.ptr<float>(0) + j;
//	//	float *pt1 = cumconf.ptr<float>(0) + j;
//	//	float tem = *pt;
//	//	*pt1 = *pt;
//	//	for (int i = 0; i < b.rows - 1; i++)
//	//	{
//	//		pt = pt + b.cols;
//	//		pt1 = pt1 + b.cols;
//	//		tem = tem + *pt;
//	//		*pt1 = tem;
//	//	}
//	//}
//
//
//	//for (int j = 0; j < b.cols; j++)
//	//{
//	//	float *pt1 = b.ptr<float>(j - 1);
//	//	float *pt2 = b.ptr<float>(j);
//	//	for (int i = 0; i < b.rows; i++){
//	//		pt2[i] = pt2[i] + pt1[i];
//	//	}
//	//}
//
//
//	//for (int j = 0; j < nr; j++)
//	//{
//	//	const float* inData = b.ptr<float>(j);
//	//	float* outData = outb.ptr<float>(j);
//	//	for (int i = 0; i < nc-1500; i++)
//	//	{			
//	//		outData[i+1500] = inData[i] + inData[i+1500];
//	//	}
//
//	//}
//
//	
//	//t = (double)cvGetTickCount() - t;
//	//printf("Info5: the detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
//	////cumconf = b.clone();
//	////cout << cumconf;
//	//cout << outb << endl;
//
//
//
//
//
//	system("pause");
//	return 0;
//
//
//
//}
//

//int main(int argc, char **argv){
//
//	int device = argc > 1 ? atoi(argv[1]) : 0;
//	af::setDevice(device);
//	af::info();
//
//
//	//Mat H(1500,1500,CV_32FC1);
//	//cv::randn(H, 0, 1); /*(Mat_<double>(9, 6) <<
//	//	0.88160098, -1.64336536, -2.11347045, 0.56165012, 0.76329896, -0.87436239,
//	//	-0.1428785, -0.48602197, -0.68572762, -0.57335307, 0.2387667, 1.46340472,
//	//	-1.28804925, -1.12481576, 0.12405203, -1.05424422, 2.33157994, 0.22642909,
//	//	-2.2995803, -0.54955355, -2.03897209, 0.34974659, -1.45958509, -0.32050344,
//	//	1.32910356, -0.20128925, -0.22179473, -0.91015079, 0.58612652, -0.28173648,
//	//	-0.96329815, -0.10660642, -0.03960701, 0.37655642, -0.99703022, -1.06457146,
//	//	1.08492705, -1.21790987, 0.47968554, -0.40010569, -1.79978356, 0.0318384,
//	//	0.93514346, 0.26986319, 1.10777352, -0.67400066, -1.35683187, -1.42736793,
//	//	0.28383533, 0.52944351, 0.58851865, 0.66319279, 1.2344684, 0.09793161);*/
//
//	////H = H.t();
//	//
//	//Mat Q;
//	//Mat R;
//	//FaceDetect ff;
//	//// Select a device and display arrayfire info
//
//	//
//	//FaceDete_CL fc;
//	//array in;
//	//printf("Create a 5-by-3 matrix of random floats on the GPU\n");
//	//array A = randu(1500, 1500, f32);
//	////af_print(A);
//
//	//double t = (double)cvGetTickCount();
//	//fc.mat1F_to_array(H, in);
//	////
//	////af_print(in);
//
//
//	////af::array q, r, tau;
//	////af::qr(q, r, tau, in);
//
//	////af_print(q);
//	////af_print(r);
//	////af_print(tau);
//
//	//////ff.QRDecomp(H, Q, R);
//	//t = (double)cvGetTickCount() - t;
//	//printf("Info5: the detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
//	////
//	//double t1 = (double)cvGetTickCount();
//	//Mat Image;
//	//fc.array_to_mat1F(A,Image);
//	//t1 = (double)cvGetTickCount() - t1;
//	//printf("Info5: the detection time = %g ms\n", t1 / ((double)cvGetTickFrequency()*1000.0));
//	//imshow("vvv", Image);
//	//cvWaitKey(6000);
//	//
//	//cout << Image;
//
//	//cout << Q << "\n";
//	//cout << R << "\n";
//	//cout << Q*R << "\n";
//
//	
//	FaceDete_CL fc;
//	array A = randu(6, 1500, f32);
//	timer t = timer::start();
//	array p = fc.array_affparam2mat(A);
//	fc.array_warpimage(A, p, 32);
//
//	printf("elapse seconds: %g ms\n", timer::stop()*1000);
//
//	system("pause");
//	return 0;
//
//
//
//
//}


int main(int argc, char **argv){

	
	//af::sync();
	//******************test in avi file*********************
	string filename = "../dudek.avi";
	VideoCapture capture(filename);
	if (!capture.isOpened())
		throw "Error when reading steam_avi";

	//******************test in cam*************************
	//VideoCapture capture;
	//FaceDetect fd;
	//capture.open(0);
	//// Read options
	////read_options(argc, argv, capture);
	//// Init camera
	//if (!capture.isOpened())
	//{
	//	cout << "capture device failed to open!" << endl;
	//	return 1;
	//}
	//namedWindow("CT", CV_WINDOW_AUTOSIZE);
	//*****************************************************


	Mat frame;
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	vector<Rect> faces;
	Mat croppedMat;
	Mat Image;
	int p[5];
	while (faces.size() == 0){
		capture >> frame;
	
		cv::CascadeClassifier FaceCascade;
		FaceCascade.load("../haarcascade_frontalface_alt_cpu.xml");
	
			
		frame.copyTo(croppedMat);

	
		cvtColor(croppedMat, Image, CV_BGR2GRAY);
		double t = (double)cvGetTickCount();
		//FaceCascade.detectMultiScale(croppedMat, faces, 1.1, 2, 0, Size(30, 30));
		FaceCascade.detectMultiScale(Image, faces, 1.5, 2, 0, Size(30, 30));
		t = (double)cvGetTickCount() - t;
		printf("Info10: the Face detect cost time is :%g ms.\n", t / ((double)cvGetTickFrequency()*1000.0));
		for (int i = 0; i < faces.size(); i++){
			Rect r = faces[i];
			//fd.param.est = faces[i];
			//rectangle(croppedMat, faces[i], Scalar(255, 255, 0), 2, 8, 0);
			printf("Info11:%d face is found ad Rect(%d,%d,%d,%d).\n", i + 1, r.x, r.y, r.width, r.height);
			p[0] = r.x + r.width*0.5;
			p[1] = r.y + r.height*0.5;
			p[2] = r.width;
			p[3] = r.height;
			p[4] = 0;
		}
	}
	for (int i = 0; i < 5; i++) std::cout << p[i] << endl;
	//imshow("FaceDete", croppedMat);
	//cvWaitKey(5000);
	
	//capture.release();
		float data[6] = { (float)p[0], (float)p[1], (float)(p[2] / 32.0),(float)p[4], (float)(p[3] / p[2]*1.0), 0.0 };	
	
		int updateNFrame = 15;
	int device = argc > 1 ? atoi(argv[1]) : 0;
	af::setDevice(device);
	af::info();		
	FaceDete_CL fc;
	FaceDetect fd;
	array image;


	
	array param0(6, 1, data);
	array paraArray = fc.array_affparam2mat(param0);

	array paraArraynum = tile(paraArray, 1, updateNFrame);
	//af_print(paraMatnum);
	array F0(32 * 32, updateNFrame);
	fc.mat1F_to_array(Image, image);

	//***************set up the seetings******************* 

	printf("\n1:*********Initial block image tensor*********\n");
	F0 = fc.array_warpimage(image, paraArraynum, 32);

	//af::Window wnd(32,32,"Fast Feature");
	//wnd.image(af::moddims(F0.col(0), 32, 32).T().as(u8));
	   
	printf("\n2:*********start to the skl*****************\n");
	//float affsigma[6] = { 4, 4, 0.002, 0.002, 0.005, 0.001 };	

	float affsigma[6] = { 9, 9, 0.05, 0.05, 0.005, 0.001 };
	fc.opt = { 1000, 0.7, 1, 2, 16, array(6, 1, affsigma) };

	// ***************For test :calculate the mean face: p = [188,192,110,130,-0.08];
	

	//float abb[5] = { 188, 192, 110, 130, -0.08 };
	//float datappppp[6] = { (float)abb[0], (float)abb[1], (float)(abb[2] / 32.0), (float)abb[4], (float)(abb[3] / p[2] * 1.0), 0.0 };

	//array param01(6, 1, data);
	//array paraArray1 = fc.array_affparam2mat(param01);
	//array F01 = fc.array_warpimage(image, paraArray1, 32);
	//fc.tmpl.mean = F01.col(0);
	//fc.param.wimg = fc.tmpl.mean;


    //**********************************************************************************
	
	fc.tmpl.mean = F0.col(0);
	fc.param.wimg = fc.tmpl.mean;
	fc.tmpl.numsample = 0;
	fc.param.est = paraArray;
	//af_print(paraArray);
	int f = 0;
	array wings = constant(0,1024,1);
	Mat fra;
	while (f < 1000000){
	//while (f < 100 ){
		
		capture >> frame;
		//cout << "\n3:*********start to update the parameters************\n" << endl;

		cvtColor(frame, fra, CV_BGR2GRAY);
		array NewFrame;
		//timer tbb = timer::start();
		fc.mat1F_to_array(fra, NewFrame);
		
		/*fc.array_estwarp_condens(NewFrame);*/

		//***************************complete estwarp_condens function : start *****************************

		
		fc.array_estwarp_condens(NewFrame);
		/*
		//int n = fc.opt.numsample;//1500
		//int width = fc.tmpl.mean.dims(0);
		//int height = fc.tmpl.mean.dims(1);
		//cout << width << " " << height << " " << endl;
		//int N = height*width;//1024

		//array q;
		//if (fc.param.param.dims(0) != 6)
		//{
		//	q = fc.array_affparam2geom(fc.param.est);
		//	fc.param.param = tile(q, 1, n);
		//}
		//else
		//{
		//	array cumconf = accum(fc.param.conf);
		//	array  idx = floor(sum(tile(randn(1, n), n, 1) > tile(cumconf, 1, n)));
		//	fc.param.param = fc.param.param(span, idx);
		//}

		////af_print(q);
		////af_print(param.param);


		//fc.param.param = fc.param.param + randn(6, n)*tile(fc.opt.affsig, 1, n);

		//array para = fc.array_affparam2mat(fc.param.param);

		//timer ti = timer::start();
		//array wimgs(32 * 32, n);
		////cout << para.dims(0) << " " << para.dims(1) << " " << para.dims(2) << endl;
		//wimgs = fc.array_warpimage(NewFrame, para, 32);


		//printf("the most cost time  warpimage is  seconds: %g ms\n", timer::stop() * 1000);

		////timer tij  = timer::start();
		////array bbb = tile(ccc, 1,1,n);
		////array ccc = af::moddims(tmpl.mean,dim4(32,32));
		////array bbb(1024, 1, n);
		////cout << tmpl.mean.dims(0) << " " << tmpl.mean.dims(1) << " " << tmpl.mean.dims(2) << endl;
		////gfor(seq i, n){
		//// bbb(span, span, i) = tmpl.mean;
		////}
		////af::sync();
		////printf("the most tile time is  seconds: %g ms\n", timer::stop() * 1000);
		////int gg;
		////cin >> gg;
		//timer::start();
		//array diff = tile(fc.tmpl.mean, 1, n) - wimgs;
		//array coefdiff;

		//

		//if (fc.tmpl.basis.dims(2) > 0)
		//{
		//	array coef = transpose(fc.tmpl.basis)*diff;
		//	diff = diff - fc.tmpl.basis*coef;
		//	if (fc.param.param.dims(0) != 0)
		//	{
		//		coefdiff = fc.tmpl.reseig*(abs(coef) - abs(fc.param.coef)) / tile(fc.tmpl.eigval, 1, n);
		//	}
		//	else
		//		coefdiff = coef*fc.tmpl.reseig / tile(fc.tmpl.eigval, 1, n);
		//	fc.param.coef = coef;
		//}

		//fc.param.conf = exp(transpose(sum(diff*diff)*(-1) / fc.opt.condenssig));
		//fc.param.conf = fc.param.conf / tile(sum(fc.param.conf), n, 1);


		//float maxElement = 0.0;
		//int maxId = 0;
		//timer tamab = timer::start();

		//float *aaa = fc.param.conf.host<float>();
		//for (int i = 0; i < n; i++)
		//{
		//	if (aaa[i] > maxElement)
		//	{
		//		maxElement = aaa[i];
		//		maxId = i;
		//	}
		//}

		//fc.param.est = fc.array_affparam2mat(fc.param.param.col(maxId));
		//fc.param.wimg = wimgs.col(maxId);
		//fc.param.err = diff.col(maxId);
		//fc.param.recon = fc.param.wimg + fc.param.err;

		printf("the most cost time  warpimage is  seconds: %g ms\n", timer::stop() * 1000);
		// *****************drawresult********************
		*/

		
		//Mat paramest;
		//fc.array_to_mat1F(fc.param.est, paramest);
		//fd.param.est = paramest;
		//fd.drawtrackresult(false, f, frame);
		

		/*array wings	*/
		//***************************complete estwarp_condens function : end*****************************


		
		//af::Window wnd(32,32,"Fast Feature");
		//do{
		//	wnd.image(af::moddims(fc.param.wimg, 32, 32).T().as(u8));
		//} while (!wnd.close());

		//***************************we must add the skl algorithm , by the hand of U(i) to calculate.
		//cout << "fc.param.wimg :" << fc.param.wimg.dims(0) << " " << fc.param.wimg.dims(1) << endl;
		//cout << "wings 0:" << wings.dims(0) << " " << wings.dims(1) << endl;
		wings = join(1, wings, fc.param.wimg);
		//cout << "wings 1:"<<wings.dims(0) << " " << wings.dims(1) << endl;
		if (wings.dims(1) >= fc.opt.batchsize+1){
			
			//if (fc.param.coef.dims(0) == -1)
			//{}
			//cout << "222222" << endl;
			fc.array_sklm(wings.cols(1, fc.opt.batchsize));
			
			wings = constant(0,1024,1);
			//if (fc.tmpl.basis.dims(1) > fc.opt.maxbasis)
			//{
			//	//fc.tmpl.reseig = fc.opt.ff*fc.tmpl.reseig 
			//	fc.tmpl.basis = fc.tmpl.basis.cols(1, fc.opt.maxbasis);
			//	fc.tmpl.eigval = fc.tmpl.eigval.cols(1, fc.opt.maxbasis);
			//}
		}
		
		Mat paramest;
		fc.array_to_mat1F(fc.param.est, paramest);
		fd.param.est = paramest;
		fd.drawtrackresult(false, f, frame);





		//af::Window wnd(32,32,"Fast Feature");
		//wnd.image(af::moddims(wimgs.col(1000), 32, 32).T().as(u8));
		//printf("the total time is : %g ms\n", timer::stop() * 1000);
		//af::Window wnd(32,32,"Fast Feature");
		//do{
		//	wnd.image(af::moddims(fc.param.wimg, 32, 32).T().as(u8));
		//} while (!wnd.close());


	
		f++;







	}
	//******************************test the array_affparam2geom in 1 times***************************************
	//array dataB(6, 1, data);
	////af_print(dataB);
	//array mean;
	//Vector<Mat> imageVecMat;
	//Vector<array> imageVecArr;
	//float loadin[6] = { 528.4160, 229.8781, 8.0433, 0.9384, -0.9384, 8.0433 };
	//array paraMatc = array(6, 1, loadin);
	//timer t = timer::start();
	//array paraMat = fc.array_affparam2mat(dataB);
	//
	////float loadin[6] = { 469.3984, 284.4926, 8.5796, -0.1701, 0.1701, 8.5796 };
	//
	//array q = fc.array_affparam2geom(paraMatc);
	//printf("elapse seconds: %g ms\n", timer::stop() * 1000);
	////***************set up the seetings******************* 
	//fc.param.est = paraMatc;





	///* 
	//******************************test the array_warpimage in 1500 times***************************************
	////af_print(paraMat);
	///*imshow("vao", Image);*/
	//fc.mat1F_to_array(Image, in);
	////cout << Image(Range(0,1),Range::all());
	////af_print(in);
	////imshow("win", Image);
	////cvWaitKey(6000);
	////
	////cout << in.dims(0) << " " << in.dims(1) << " " << in.dims(2) << endl;
	////af::Window wnd("Fast Feature");
	////wnd.image(in.as(u8));


	//int n = 1500;
	//array paraMatnum = tile(paraMat, 1, n);
	////af_print(paraMatnum);
	//array posTe(32 * 32, n);
	//posTe = fc.array_warpimage(in, paraMatnum, 32);
	//printf("elapse seconds: %g ms\n", timer::stop()*1000);
	//cout << posTe.dims(0) << " " << posTe.dims(1) << " " << posTe.dims(2) << endl;




	////af_print(posTe.row(0));
	////af_print(imageVecArr[0]);
	////cout << "ImageVec size" << imageVecArr[0].dims(0) << " " << imageVecArr[0	].dims(1) << endl;
	////cout << "ImageVec size" << imageVecArr.size() << endl;


	//Mat show(32,32,CV_32F);
	//array cao = posTe.col(1000);
	//af::Window wnd(32,32,"Fast Feature");
	//wnd.image(af::moddims(cao, 32, 32).T().as(u8));
	//*********************************************************************

	



	//fc.array_to_mat1F(cao, show);
	//imshow("win", show);
	//cvWaitKey(6000);
	//
	capture.release();
	system("pause");
	return 0;



}