
#include <string>
#include <Windows.h>
#include <direct.h>
#include <filesystem>
#include <stdarg.h>
#include "FaceDete.h"
#include <math.h>
#include <cstdlib>
#include <ctime>
#define check_mem(s,name) (void)(s.name)
#define pi 3.14159265358979323846
template <class T>
int getarraylen(T& array){

	return (sizeof(array) / sizeof(array[0]));
}



FaceDetect::FaceDetect()
{
}

FaceDetect::~FaceDetect()
{
}

string FaceDetect::wchar_to_string(char *wchar)  {
	string str = "";
	for (int i = 0; i < strlen(wchar); i++)
	{
		str += wchar[i];
	}
	return str;
}

Mat FaceDetect::getInitAffineMat(Mat M0, float a[][3]){
	
	for (int i = 0; i < M0.cols; i++){
		for (int j = 0; j < M0.rows; j++){
			M0.at<float>(j, i) = (float)a[j][i];
		}
	}
	return M0;
}
//Mat FaceDetect::getInitAffineMat(int a[6]){
//
//	Mat M0 = Mat(1, 6, CV_16U);
//	for (int i = 0; i < M0.cols; i++){
//		for (int j = 0; j < M0.rows; j++){
//			M0.at<unsigned>(i,j) = (unsigned int)a[j];
//		}
//	}
//	return M0;
//}
void FaceDetect::meshgrid(const Mat &xgv, const Mat &ygv, Mat1i &X, Mat1i &Y){

	repeat(xgv.reshape(1, 1), ygv.total(), 1, X);
	repeat(ygv.reshape(1, 1).t(), 1, xgv.total(), Y);

}

Mat FaceDetect::DetectNose(CascadeClassifier cascade, Mat img, Mat Image){


	vector<Rect> Noses;
	Mat croppedimage;
	//cascade.detectMultiScale(img, faces, 1.1, 2, CV_HAAR_FIND_BIGGEST_OBJECT, Size(10, 10), Size(200, 200));
	double t = (double)cvGetTickCount();//get the runtime time count;
	cascade.detectMultiScale(img, Noses, 1.1, 11, 0
		| CV_HAAR_FIND_BIGGEST_OBJECT //| CV_HAAR_SCALE_IMAGE 
		, Size(30, 30), Size(90, 90));

	t = (double)cvGetTickCount() - t;
	printf("Info5: the detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
	printf("Info6:%d Nose(s) are found\n", int(Noses.size()));
	for (int i = 0; i < Noses.size(); i++){

		Rect r = Noses[i];
		printf("Info7:%d Nose is found ad Rect(%d,%d,%d,%d).\n", i + 1, r.x, r.y, r.width, r.height);
		rectangle(img, Point(r.x, r.y), Point(r.width + r.x, r.height + r.y), Scalar(0, 0, 0), 2, 8, 0);
		imshow("the cropped Nose", img);

		//****************get the center**************
		Point center = Point(r.x + r.width*0.5, r.y + r.height*0.5);
		printf("Info8: The center is : %d,%d \n", center.x, center.y);
		
		// init the affineMat by 2-dimension array 
		float a[2][3] = {
			1, 0, 200 - center.x, 0, 1, 200 - center.y
		};
		cout << "Info9:" << a[0][2] << a[1][2] << endl;
		Mat affineMat(2, 3, CV_32F);
		affineMat = getInitAffineMat(affineMat, a);
		cout << affineMat << endl;
		warpAffine(Image, croppedimage, affineMat, Size(Image.size().width - abs(200 - center.x), Image.size().height - abs(200 - center.y)));
		//Rect myRect = cvRect(center.x-20, 200 - center.y, 2 * (200 - center.x), 2 * (200 - center.y));
		/*Rect myRect = cvRect(center.x,)
		Mat croppedimage = Image(myRect);*/
		imshow("the cropped face", croppedimage);
		Noses.clear();
		cvWaitKey(1000);

	}
	return croppedimage;
}

Mat FaceDetect::FaceDetection(CascadeClassifier FaceCascade, Mat croppedMat)
{
	vector<Rect> faces;
	Mat Image = croppedMat.clone();
	double t = (double)cvGetTickCount();
	FaceCascade.detectMultiScale(croppedMat, faces,1.1,2,0,Size(30,30));
	t = (double)cvGetTickCount() - t;
	printf("Info10: the Face detect cost time is :%g ms.\n", t / ((double)cvGetTickFrequency()*1000.0));
	for (int i = 0; i < faces.size(); i++){


		Rect r = faces[i];
		printf("Info11:%d face is found ad Rect(%d,%d,%d,%d).\n", i + 1, r.x, r.y, r.width, r.height);
		Mat roi;
		roi = croppedMat(Rect(r.x, r.y, r.width, r.height));

		Mat roisize(250, 250, CV_8UC1);
		resize(roi, roisize, Size(250, 250), 0, 0, INTER_LINEAR);

		//imshow("The roisize of face Demo", roisize);
		//ellipse mask the image

		Mat im2(250, 250, CV_8UC1, Scalar(255, 255, 255));

		//ellipse(im2, Point(r.x+r.width*0.5,r.y+r.height*0.5), Size(130, 125), 0, 0, 360, Scalar(0, 0, 0), -1, 8);
		ellipse(im2, Point(130, 125), Size(250/2.5, 250/2), 0, 0, 360, Scalar(255,255,255), -1, 8);

		Mat result(250,250,CV_8UC3);
		
		roisize.copyTo(result, im2);
		cv::imshow("The result of face Demo", result);
		
		printf("The result image channels:%d\n", result.channels());
		vector<Point> ellipse_maskVector;

		for (int i = 0; i < 250; i++){
			for (int j = 0; j < 250; j++){
				//printf("%d,%d\n", i, j);
				//cout << ((125 * 125 * (i - 130)*(i - 130) + 130 * 130 * (j - 125)*(j - 125)) <= (16900 * 15625)) << endl;
				if (((100 * 100 * (i - 130)*(i - 130) + 125 * 125 * (j - 125)*(j - 125)) <= (10000 * 15625)) != 0){
					//ellipse_mask.at<int>(i,j) = 1;
					ellipse_maskVector.push_back(Point(i, j));
				}
				else{
					
					result.at<uchar>(i, j) = 255;

				}
				//else{
				//	ellipse_mask.at<int>(i, j) = 0;
				//}
			}
		}

		
		Rect VV(28, 4, 199, 244);
		Mat Image;
		Image = result(VV);
		cv::imshow("The Image of face Demo", Image);
		imwrite("../result/Image.jpg", Image);
		cvWaitKey(6000);
	}
	return croppedMat;
}


Mat FaceDetect::face_detect_draw(CascadeClassifier FaceCascade, Mat croppedMat){


	

		vector<Rect> faces;
		Mat Image;
		cvtColor(croppedMat, Image, CV_BGR2GRAY);
		double t = (double)cvGetTickCount();
		//FaceCascade.detectMultiScale(croppedMat, faces, 1.1, 2, 0, Size(30, 30));
		FaceCascade.detectMultiScale(Image, faces, 1.5, 2, 0, Size(30, 30));
		t = (double)cvGetTickCount() - t;
		printf("Info10: the Face detect cost time is :%g ms.\n", t / ((double)cvGetTickFrequency()*1000.0));
		for (int i = 0; i < faces.size(); i++){

			//Rect r = faces[i];
			//printf("Info11:%d face is found ad Rect(%d,%d,%d,%d).\n", i + 1, r.x, r.y, r.width, r.height);
			//rectangle(croppedMat, Point(r.x, r.y), Point(r.width + r.x, r.height + r.y), Scalar(0, 0, 0), 2, 8, 0);
			rectangle(croppedMat, faces[i], Scalar(255, 255, 0), 2, 8, 0);
			/*imshow("FaceDetection Demo", croppedMat);*/
			//char c = cvWaitKey(33);
			//if (c == 27){
			//	break;
			//}
			
			/*
			Mat im2(croppedMat.rows, croppedMat.cols, CV_8UC1, Scalar(0, 0, 0));
			//ellipse(im2, center, Size(r.width, r.height), 0, 0, 360, Scalar(255, 255, 255), -1, 8);
			ellipse(im2, Point(r.x+r.width*0.5,r.y+r.height*0.5), Size(120, 160), 0, 0, 360, Scalar(255, 255, 255), -1, 8);
			imshow("The im2 of face Demo", im2);
			Mat result;
			Mat roi;
			bitwise_and(croppedMat, im2, result);
			roi = result(Rect((r.x + r.width*0.5) - 150, r.y + r.height*0.5 - 150, 300, 300));
			imshow("The roi of face Demo", roi);

			imshow("The face Demo", result);
			cvWaitKey(6000);
			*/
		}

	return croppedMat;

}


void FaceDetect::FaceDeteForTracking(CascadeClassifier FaceCascade, Mat &croppedMat, vector<Rect> &faces){

	Mat Image;
	cvtColor(croppedMat, Image, CV_BGR2GRAY);
	double t = (double)cvGetTickCount();
	//FaceCascade.detectMultiScale(croppedMat, faces, 1.1, 2, 0, Size(30, 30));
	FaceCascade.detectMultiScale(Image, faces, 1.5, 2, 0, Size(30, 30));
	t = (double)cvGetTickCount() - t;
}

Mat FaceDetect::warpimage(Mat image, Mat paraMat, Size sz){

	double t = (double)cvGetTickCount();
	int width = sz.width;
	int height = sz.height;
	int *data = (int *)malloc(sizeof(int)*width);
	//int data[32];
	for (int i = 0; i < width; i++) data[i] = i + 1 - int(width/2);
	//for (int i = 0; i < width; i++) cout << data[i] << endl;
	
	//double m[3][3] = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
	//Mat M = Mat(3, 3, CV_64F, m);
	//cout << M << endl;
	
	Mat xgv, ygv;
	xgv = Mat(width, 1, CV_32S, data);
	ygv = Mat(width, 1, CV_32S, data);

	Mat1i X, Y;
	meshgrid(xgv, ygv,X,Y);

	vector<Mat1i> matrices = {

		Mat::ones(Size(width*height, 1), CV_32S), Y.reshape(1, 1), X.reshape(1, 1),
	};
	
	Mat pos;
	cv::vconcat(matrices, pos);
	
	/*cv::vconcat( Mat::ones( Size(width*height, 1) , CV_32S), Y.reshape(1, 1), pos);
	cv::vconcat(pos, X.reshape(1, 1),pos);*/
	//cout << pos.cols<<"  "<<pos.rows << endl;
	//cout << pos << endl;
	//cout << (Mat_<float>(3, 2) << para[0], para[1], para[2], para[3], para[5], para[4]) << endl;


	//float indata[] = { para[0], para[1], para[2], para[3], para[5], para[4] };
	//Mat paraMat = Mat(3, 2, CV_32FC1, indata);
	//cout << paraMat << endl;


	Mat result;
	pos.convertTo(result, CV_32FC1);
	Mat resultMat = result.t()*paraMat;

	//cout << resultMat.cols << "  " << resultMat.rows << endl;
	//cout << resultMat(Range::all(), Range(1, 2)).rows << endl; 
	//cout << resultMat(Range::all(), Range(1, 2)).cols << endl;
	
	Mat col = resultMat(Range::all(), Range(0, 1));
	Mat row = resultMat(Range::all(), Range(1, 2));
	Mat wingPos1 = col.t();
	Mat wingPos2 = row.t();
	vector<Mat> position = {

		wingPos1.reshape(1, width), wingPos2.reshape(1, width),
	};
	
	//cout << position[1];
	Mat output;
	cv::remap(image, output, position[0].t(), position[1].t(),cv::INTER_LINEAR);
	return output;
	t = (double)cvGetTickCount() - t;
	printf("Info1:  time1 = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
}


/*
void cvShowManyImages(char* title, int nArgs, ...) {

	// img - Used for getting the arguments 
	IplImage *img;

	// [[DispImage]] - the image in which input images are to be copied
	IplImage *DispImage;

	int size;
	int i;
	int m, n;
	int x, y;

	// w - Maximum number of images in a row 
	// h - Maximum number of images in a column 
	int w, h;

	// scale - How much we have to resize the image
	float scale;
	int max;

	// If the number of arguments is lesser than 0 or greater than 12
	// return without displaying 
	if (nArgs <= 0) {
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nArgs > 12) {
		printf("Number of arguments too large....\n");
		return;
	}
	// Determine the size of the image, 
	// and the number of rows/cols 
	// from number of arguments 
	else if (nArgs == 1) {
		w = h = 1;
		size = 300;
	}
	else if (nArgs == 2) {
		w = 2; h = 1;
		size = 300;
	}
	else if (nArgs == 3 || nArgs == 4) {
		w = 2; h = 2;
		size = 300;
	}
	else if (nArgs == 5 || nArgs == 6) {
		w = 3; h = 2;
		size = 200;
	}
	else if (nArgs == 7 || nArgs == 8) {
		w = 4; h = 2;
		size = 200;
	}
	else {
		w = 4; h = 3;
		size = 150;
	}

	// Create a new 3 channel image
	[[DispImage]] = cvCreateImage(cvSize(100 + size*w, 60 + size*h), 8, 3);

	// Used to get the arguments passed
	va_list args;
	va_start(args, nArgs);

	// Loop for nArgs number of arguments
	for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {

		// Get the Pointer to the IplImage
		img = va_arg(args, IplImage*);

		// Check whether it is NULL or not
		// If it is NULL, release the image, and return
		if (img == 0) {
			printf("Invalid arguments");
			cvReleaseImage(&DispImage);
			return;
		}

		// Find the width and height of the image
		x = img->width;
		y = img->height;

		// Find whether height or width is greater in order to resize the image
		max = (x > y) ? x : y;

		// Find the scaling factor to resize the image
		scale = (float)((float)max / size);

		// Used to Align the images
		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		// Set the image ROI to display the current image
		cvSetImageROI(DispImage, cvRect(m, n, (int)(x / scale), (int)(y / scale)));

		// Resize the input image and copy the it to the Single Big Image
		cvResize(img, DispImage);

		// Reset the ROI in order to display the next image
		cvResetImageROI(DispImage);
	}

	// Create a new window, and show the Single Big Image
	cvNamedWindow(title, 1);
	cvShowImage(title, DispImage);

	cvWaitKey();
	cvDestroyWindow(title);

	// End the number of arguments
	va_end(args);

	// Release the Image Memory
	cvReleaseImage(&DispImage);
}
*/



void FaceDetect::drawtrackresult(bool flags,int fno, Mat frame){
	namedWindow("Face Tracking!");
	int fh = frame.cols;
	int fw = frame.rows;
	//int th = tmpl.mean.cols;
	//int tw = tmpl.mean.rows;

	int th = 32;
	int tw = 32;


	float hb = th / (fh / fw*(5 * tw) + 3 * th);

	if (flags == false){
	//respresent the drawopt is empty ,so init the drawopt
		drawopt.frm = Rect(Point(0.00, 3 * hb), Point(1.00, 1 - 3 * hb));
		drawopt.window = Rect(Point(0.00, 2 * hb), Point(1.00, hb));
		drawopt.basis = Rect(Point(0.00, 0.00), Point(1.00, 2 * hb));
		drawopt.showcoef = 0;
		drawopt.matcoef = 3;
		drawopt.showcondens = 0;
		drawopt.thcondens = 0.001;
	}


	//int width = tmpl.mean.cols;
	//int height = tmpl.mean.rows;
	int width = 32;
	int height = 32;
	int N = width*height;

	//applyColorMap(frame, frame, cv::COLORMAP_JET);
	stringstream s;
	s << fno;
	string fnoi;
	s >> fnoi;
	putText(frame, fnoi, Point(5, 18), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(200, 200, 250), 1, CV_AA);

	//cout << param.est << endl;
	//Mat M = param.est.t();
	
	Mat M = param.est;
	Mat corners = (Mat_<float>(3, 5) << 1, 1,1,1, 1,  -width / 2,width / 2, width / 2,-width / 2, -width / 2,   -height / 2,  -height / 2, height / 2,  height / 2,-height / 2);
	corners = M*corners;
	line(frame, Point(corners.at<float>(0, 0), corners.at<float>(1, 0)), Point(corners.at<float>(0, 1), corners.at<float>(1, 1)), Scalar(255, 255, 0), 2, 8, 0);
	line(frame, Point(corners.at<float>(0, 1), corners.at<float>(1, 1)), Point(corners.at<float>(0, 2), corners.at<float>(1, 2)), Scalar(255, 255, 0), 2, 8, 0);
	line(frame, Point(corners.at<float>(0, 2), corners.at<float>(1, 2)), Point(corners.at<float>(0, 3), corners.at<float>(1, 3)), Scalar(255, 255, 0), 2, 8, 0);
	line(frame, Point(corners.at<float>(0, 0), corners.at<float>(1, 0)), Point(corners.at<float>(0, 3), corners.at<float>(1, 3)), Scalar(255, 255, 0), 2, 8, 0);

	//Rect face = Rect(param.est.at<float>(0, 0) - 50, param.est.at<float>(0, 1) - 50, 100, 100);
	//rectangle(frame, face, Scalar(255, 255, 0), 2, 8, 0);

	imshow("Face Tracking!", frame);
	cvWaitKey(30);



}


Mat FaceDetect::affparam2mat(float *p){

	int sz = 6;
	float sc = p[2];
	float th = p[3];
	float sr = p[4];
	float phi = p[5];
	float cth = cos(th);
	float sth = sin(th);
	float cph = cos(phi);
	float sph = sin(phi);

	float ccc = cth*cph*cph;  
	float ccs = cth*cph*sph;
	float css = cth*sph*sph;
	float scc = sth*cph*cph;
	float scs = sth*cph*sph;
	float sss = sth*sph*sph;


	//float data[6] = { p[0], p[1], sc*(ccc + scs + sr*(css - scs)), sc*(sr*(ccs - scc) - ccs - sss), sc*(scc - ccs + sr*(ccs + sss)) , sc*(sr*(ccc + scs) - scs + css) };
	//for (int i = 0; i < 6; i++)cout << data[i] << endl;
	//float dataP[6] = { p[0], p[1], p[2], p[3], p[4], p[5] };
	//for (int i = 0; i < 6; i++)cout << dataP[i] << endl;
	//Mat paraMat = Mat(3, 2, CV_16U, dataP);
	Mat paraMat = (Mat_<float>(3, 2) << p[0], p[1], sc*(ccc + scs + sr*(css - scs)), sc*(sr*(ccs - scc) - ccs - sss), sc*(scc - ccs + sr*(ccs + sss)), sc*(sr*(ccc + scs) - scs + css));
	return paraMat;
}


void FaceDetect::affparam2geom(Mat p,float* &q){
	p = p.reshape(0, 6);
	Mat A = (Mat_<float>(2, 2) << p.at<float>(2, 0), p.at<float>(3, 0), p.at<float>(4, 0), p.at<float>(5, 0));
	
	Mat w, u, vt;
	SVD::compute(A, w, u, vt, SVD::MODIFY_A);
	//cout << (Mat_<float>(2, 2) << w.at<float>(0, 0), 0.0, 0.0, w.at<float>(1, 0)) << endl;
	//cout << u*(Mat_<float>(2, 2) << w.at<float>(0, 0), 0.0, 0.0,w.at<float>(1, 0))*vt << endl;

	//cout << u << endl << w << endl << vt << endl;
	if (determinant(u) > 0){
		//cout << "less than" << endl;
		//u = u(Range::all(), Range(1, 0));
		//vt = vt(Range::all(), Range(1, 0));
		//w = w(Range(1, 0), Range(1, 0));
		u = (Mat_<float>(2, 2) << u.at<float>(0, 1), u.at<float>(0, 0), u.at<float>(1, 1), u.at<float>(1, 0));
		vt = (Mat_<float>(2, 2) << vt.at<float>(0, 1), vt.at<float>(0, 0), vt.at<float>(1, 1), vt.at<float>(1, 0));
		w = (Mat_<float>(2, 2) << w.at<float>(1, 0), 0.0, 0.0, w.at<float>(0, 0));
		//cout << vt<< endl;
	}
	//float q[6];
	q[0] = p.at<float>(0, 0);
	q[1] = p.at<float>(1, 0);

	q[3] = atan2(u.at<float>(1, 0) *vt.at<float>(0, 0) + u.at<float>(1, 1) *vt.at<float>(0, 1), u.at<float>(0, 0) *vt.at<float>(0, 0) + u.at<float>(0, 1) *vt.at<float>(0, 1));

	//cout << q[0] <<" "<< q[1]<<"  " << q[3] << endl;

	float phi = atan2(-vt.at<float>(0, 1), vt.at<float>(0, 0));
	//cout << phi<<endl;
	//cout << sin(- pi/ 2.0) << " " << cos(-pi/2.0) << endl;
	if (phi <= -pi / 2.0){
		float c = cos(-pi/2.0);
		float s = sin(-pi/2.0);
		Mat R = (Mat_<float>(2, 2) << c, s, s, c);
		vt = vt*R;
		w = R.t()*w*R;
	}
	if (phi >= pi / 2.0){
		float c = cos(pi / 2.0);
		float s = sin(pi / 2.0);
		Mat R = (Mat_<float>(2, 2) << c, s, s, c);
		vt = vt*R;
		w = R.t()*w*R;
	}

	q[2] = w.at<float>(0, 0);
	
	q[4] = w.at<float>(1, 1) / w.at<float>(0, 0)*1.0;
	//cout << q[4] << endl;
	q[5] = atan2(-vt.at<float>(0, 1), -vt.at<float>(0, 0));

}


void FaceDetect::estwarp_condens(Mat frm)
{
	int n = opt.numsample;//1500
	Size sz(tmpl.mean.rows, tmpl.mean.cols);//32*32
	int N = sz.height*sz.width;//1024




	if (param.param.size().height == 0)
	{
		float *q = new float[6];
		cout << param.est<< endl;
		affparam2geom(param.est, q);
		
		Mat tem(6, 1, CV_32F);
		for (int i = 0; i < tem.rows; i++)
		{
			cout << q[i] << endl;
			tem.at<float>(i, 0) = q[i];
		}
		param.param = repeat(tem, 1, n);
		
	}
	else
	{
		double t = (double)cvGetTickCount();


	
		Mat cumconf(param.conf.rows, param.conf.cols, CV_32FC1);//  得到cumconf,按列累加
		float *pt = param.conf.ptr<float>(0);
		float *ptt = cumconf.ptr<float>(0);
		for (int i = 0; i < cumconf.cols; i++)
		{
			*ptt++ = *pt++;
		}
		for (int j = 1; j < param.conf.rows; j++)
		{
			float *pt1 = cumconf.ptr<float>(j - 1);
			float *pt2 = param.conf.ptr<float>(j);
			float *pt3 = cumconf.ptr<float>(j);
			for (int i = 0; i < param.conf.cols; i++){
				pt3[i] = pt2[i] + pt1[i];
			}
		}
		Mat repcumconf = repeat(cumconf, 1, n);

		//生成随机矩阵
		float * r = new float[n];
		#define M1 10000 //四位小数。
		srand(time(NULL));
		for (int i = 0; i < n; i++)
			r[i] = rand() % (M1) / (float)(M1);

		Mat randmat = repeat(Mat(1, n, CV_32FC1, r), n, 1);


		//得到idx
		int *sum = new int[n];
		for (int i = 0; i < n; i++)
			sum[i] = 0;

		for (int i = 0; i < n; i++){
			float * pr = randmat.ptr<float>(i);
			float * pc = repcumconf.ptr<float>(i);
			for (int j = 0; j < n; j++){
				if (pr[j]>pc[j])
					sum[j]++;
			}
		}

		//更新param.param
		Mat cloneparam = param.param.clone();
		for (int i = 0; i < param.param.rows; i++)
		{
			float *pp = param.param.ptr<float>(i);
			float *pcl = cloneparam.ptr<float>(i);
			for (int j = 0; j < param.param.cols; j++)
			{
				pp[j] = pcl[sum[j]];
			}
		}
		t = (double)cvGetTickCount() - t;
		printf("Info1:  time1 = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));
	}


	double t = (double)cvGetTickCount();

	//param.param = param.param + randn(6, n).*repmat(opt.affsig(:), [1, n]);
	Mat te(6, n, CV_32F);
	randn(te, 0, 1);
	param.param = param.param + te.mul(repeat(Mat(6, 1, CV_32F, opt.affsig), 1, n));
	cout << param.param.cols << endl;;
	//wimgs = warpimg(frm, affparam2mat(param.param), sz);
	vector<Mat> vMat;//得到wimgs   vMat中每个Mat的size为1024*1

	vector<Mat> vMat1;

	Mat tparam = param.param.t();
	for (int i = 0; i < param.param.cols; i++)
	{
		float *pt = tparam.ptr<float>(i);
		float tee[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		for (int j = 0; j < param.param.rows; j++)
		{
			tee[j] = pt[j];
		}
		//vMat1.push_back((warpimage(frm, affparam2mat(tee), sz)));
		vMat.push_back((warpimage(frm, affparam2mat(tee), sz)).reshape(0, N));

	}
	t = (double)cvGetTickCount() - t;
	printf("Info2: cropped image time2 = %g ms\n", t / ((double)cvGetTickFrequency()*1000.0));

	Mat merg(N, n, CV_32F);//1500*1024
	//Mat merg;//1024*1500
	Mat merg1;
	Mat mergMat;
	Mat repeMat;
	hconcat(vMat, merg);

	//hconcat(vMat1, merg1);
	//imshow("rrr", merg1);
	//cvWaitKey(600);
	/*merg.convertTo(mergMat, CV_32FC1);*/
	repeat(tmpl.mean.reshape(0, N), 1, n, repeMat);
	//cout << repeMat.cols << " " << repeMat.rows << endl;	
	//cout << repeMat << endl;
	//imshow("vvv", repeMat);
	//cvWaitKey(6000);
	Mat re1;
	Mat re2;
	repeMat.convertTo(re1, CV_32F, 1.0 / 256);
	merg.convertTo(re2, CV_32F, 1.0 / 256);
	Mat diff = re1 - re2;
	
	//cout << diff.cols << " " << diff.rows << endl;
	Mat coefdiff;


	if (tmpl.basis.size().height> 0)
	{
		Mat coef = tmpl.basis.t()*diff;
		diff = diff - tmpl.basis*coef;
		if (param.coef.rows == 0)
			coefdiff = ((abs(coef) - abs(param.coef))*tmpl.reseig).mul(1 / (repeat(tmpl.eigval, 1, n)));
		else
			coefdiff = (coef.mul(tmpl.reseig)).mul(1 / (repeat(tmpl.eigval, 1, n)));
		param.coef = coef.clone();
	}

	Mat dianmi;
	pow(diff, 2.0, dianmi);
	//cout << diff << endl;
	Mat rdcsum; 
	reduce(dianmi, rdcsum, 0, CV_REDUCE_SUM, CV_32F);
	//cout << "rdcsum " << rdcsum.cols << " " << rdcsum.rows << endl;
	
	Mat opprdc = (-rdcsum);
	Mat ex;//e为底的指数
	exp(opprdc.mul(1.0/ opt.condenssig), ex);
	param.conf = ex.t();
	
	/*param.conf = param.conf . / sum(param.conf);
	[maxprob, maxidx] = max(param.conf);
	param.est = affparam2mat(param.param(:, maxidx));
	param.wimg = wimgs(:, : , maxidx);
	param.err = reshape(diff(:, maxidx), sz);
	param.recon = param.wimg + param.err;*/

	Mat rep;
	reduce(param.conf, rep, 0, CV_REDUCE_SUM,CV_32F);
	float chu = 1.0 / rep.at<float>(0, 0);
	param.conf = param.conf.mul(chu);
	//cout <<"param.conf " <<param.conf.cols << " " << param.conf.rows << endl;;
	int maxidx = 0; //[maxprob, maxidx] = max(param.conf);
	float *pp = param.conf.ptr<float>(0);

	float maxprob = pp[1];
	//cout << maxprob << endl;
	for (int i = 0; i < param.conf.rows; i++)
	{
		if (*pp>maxprob)
		{
			maxprob = *pp;
			maxidx = i;
		}
		pp++;
	}

	//cout << maxidx << " " << maxprob << endl;
	//cout << param.param << endl;
	
	Mat tt = param.param(Range::all(), Range(maxidx, maxidx+1));
	//cout << tt;
	float par[6] = { tt.at<float>(0, 0), tt.at<float>(1, 0), tt.at<float>(2, 0), tt.at<float>(3, 0), tt.at<float>(4, 0), tt.at<float>(5, 0)};
	//param.est = affparam2mat(param.param(:, maxidx));
	/*float *ptt = tt.ptr<float>(0);*/
	//for (int i = 0; i < 6; i++)
	//{
	//	par[i] = ptt[i];
	//	/*ptt++;*/
	//	cout <<"wocao" <<par[i];
	//}
	param.est = affparam2mat(par);

	param.wimg = vMat[maxidx].reshape(0, 32); //param.wimg = wimgs(:, : , maxidx);
	Mat bb = diff(Range::all(), Range(maxidx, maxidx + 1)).clone();
	param.err = bb.reshape(0, 32);//param.err = reshape(diff(:, maxidx), sz);
	//cout << param.wimg << endl;
	//param.recon = param.wimg + param.err;



}


void FaceDetect::mySklm(Mat oldTensor, Mat newframe, int num){

	Mat newTensor = oldTensor.clone();
	newTensor(Range::all(), Range(0, 13 * 32)) = oldTensor(Range::all(), Range(32, 14 * 32));
	newTensor(Range::all(), Range(13 * 32, 14 * 32)) = newframe;

	Mat S(num,1,CV_32F);
	int k = 0;
	int m = 32;
	int n = 480;
	int kmax = 32;
	
	int lmin = 1; 
	int lmax = round(kmax / sqrt(2));
	int kmin = 1;
	int kstart = lmax;
	int lambda = 1;
	Mat SEQKL_U,SEQKL_V;


	int maxudim = max(kmax, kstart) + lmax;
	int maxvdim = max(kmax, kstart);

	S = Mat::zeros(Size(maxudim, 1),CV_32F);


	int i = 1;
	while (i <= n){
		
		if (i == 1 && k == 0){

			k = kstart;
			printf("Info16: Performing initial QR decomposition of first %d columns..\n", k);
			
			Mat output;
		


		}



		
	
	
	}



}
void FaceDetect::QRDecomp(Mat& m, Mat& Q, Mat& R)
{

	vector<Mat> q(m.rows);
	Mat z = m.clone();


	for (int k = 0; k < m.cols && k < m.rows - 1; k++){
		vector<float> e(m.rows, 0);
		vector<float> x(m.rows, 0);
		double a = 0;

		Mat z1 = Mat::zeros(z.rows, z.cols, CV_32F);

		for (int i = 0; i < k; i++){
			z1.at<float>(i, i) = 1;
		}
		for (int y = k; y < z1.rows; y++){
			for (int x = k; x < z1.cols; x++){
				z1.at<float>(y, x) = z.at<float>(y, x);
			}
		}
		z = z1.clone();

		for (int i = 0; i < z.rows; i++){
			a += pow(z.at<float>(i, k), 2);
			x[i] = z.at<float>(i, k);
		}

		a = sqrt(a);
		if (m.at<float>(k, k) > 0){
			a = -a;
		}
		for (int i = 0; i < m.rows; i++){
			if (i == k){
				e[i] = 1;
			}
			else{
				e[i] = 0;
			}
		}


		for (int i = 0; i < m.rows; i++){
			e[i] = x[i] + a * e[i];
		}

		float norm = 0;
		for (int i = 0; i < e.size(); i++){
			norm += pow(e[i], 2);
		}
		norm = sqrt(norm);
		for (int i = 0; i < e.size(); i++){
			if (norm != 0){
				e[i] /= norm;
			}
		}
		Mat E(e.size(), 1, CV_32F);
		for (int i = 0; i < e.size(); i++){
			E.at<float>(i, 0) = e[i];
		}

		q[k] = Mat::eye(m.rows, m.rows, CV_32F) - 2 * E * E.t();
		z1 = q[k] * z;
		z = z1.clone();
	}

	Q = q[0].clone();
	R = q[0] * m;
	for (int i = 1; i < m.cols && i < m.rows - 1; i++){
		Q = q[i] * Q;
	}
	R = Q * m;
	Q = Q.t();

	if (m.rows > m.cols){
		R = R(Rect(0, 0, m.cols, m.cols)).clone();
		Q = Q(Rect(0, 0, m.cols, m.rows)).clone();
	}


}

/*
	int main(int argc, char ** argv){


		CGaborFilter cgabor;
		Mat I = imread("C:\\Users\bcc\Documents\Visual Studio 2013\Projects\TestDete\result\Image.jpg");
		cgabor.init();
		printf("The gaborresult size:%d \n", cgabor.gaborarray.size());
		cgabor.CgaborFeatures(I,12,12);

		/*

		if (argc == 2)
		{
			printf("a single argument is required.\n");

		}	

		char _path[_MAX_PATH] = "../data";
		char *_currentPath = (char *)malloc(sizeof(char)*_MAX_PATH);
		printf("Info1 :the current path is :%s\n", _getcwd(_currentPath, _MAX_PATH));

		if (_chdir((const char*)_path) != 0)
		{
			printf("Info2 :the path is not exist:\n");
		}
		int len = strlen(_path);
		printf("%d\n", len);
		if (_path[len - 1] != '\/'){
			strcat(_path, "/*.*");
		}
		printf("Info3 :The data path is :%s\n", _path);




		WIN32_FIND_DATAA FindFileData;
		HANDLE handle;
		handle = FindFirstFileA(_path, &FindFileData);

		std::vector<string> fileName;

		while (FindNextFileA(handle, &FindFileData))
		{
			printf("%s\n", FindFileData.cFileName);
			//cout << strlen(FindFileData.cFileName) << endl;
			if (strlen(FindFileData.cFileName) != 2){
				fileName.push_back(wchar_to_string(FindFileData.cFileName));
			}
		}
		FindClose(handle);
		printf("Info4: the fileName size is :%d \n", fileName.size());
		//Mat img = imread("C:\\Users\\bcc\\Desktop\\opencvFaceDetector\\Face-Detection-master\\Test Inputs\\customers\\4.tiff",CV_LOAD_IMAGE_GRAYSCALE);fileName.size()
		namedWindow("FaceDetection Demo");
		for (int i = 0; i < 1; i++)
		{

			Mat img = imread("..\\data\\" + fileName[i], CV_LOAD_IMAGE_GRAYSCALE);
			Mat Image = img.clone();
			if (!img.data){
				printf("Error1 :Failed to load the image data");
			}
			CascadeClassifier cascade;
			CascadeClassifier FaceCascade;
			try{


				
				//imshow("FaceDetection Demo", img);
				Mat croppedMat;
				//*********************1:Detect the Nose******************
				try{
						if (cascade.load("../Nariz.xml")){
					
							croppedMat = DetectNose(cascade, img, Image);
						}else{
							printf("Error2: Failed to load the Nose xml file. \n");
						}

				}catch(Exception& e){
					printf("Error2: Could not check the Nose.%s\n", e.what());
				}
				if (croppedMat.size().width == 0)
				{
					croppedMat = Image.clone();
				}
				 //********************2:Detect the face******************
				if (FaceCascade.load("../haarcascade_frontalface_alt.xml")){

					FaceDetect(FaceCascade, croppedMat);

				}
				else
				{
					printf("Error2: Failed to load the Face xml file. \n");
				}


			}
			catch (Exception& e){
				printf("%s\n", e.what());

			}


			//imshow("Facedetection demo", img);
			//cvWaitKey(600);
			//cvDestroyAllWindows();
			img.release();
		}

		cvDestroyAllWindows();
		fileName.clear();
		


	
		system("pause");
		return 0;
	}
	*/	