/************************************************************************
* File:	FaceDete_CL.cpp
* Author: Xu Guoxia
* Email: guoxiaxu@cityu.edu.hk
* Date:	2016/09/13
* History:
************************************************************************/

#include "FaceDete_CL.h"


#define pi 3.14159265358979323846
FaceDete_CL::FaceDete_CL()
{
}

FaceDete_CL::~FaceDete_CL()
{
}


//void FaceDete_CL::FaceDete_CL_init(){
//
//	VideoCapture capture;
//
//	capture.open(0);
//
//	if (!capture.isOpened())
//	{
//		cout << "capture device failed to open!" << endl;
//
//	}
//	namedWindow("CT", CV_WINDOW_AUTOSIZE);
//	Mat frame;
//	capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
//	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
//	vector<Rect> faces;
//	oclMat croppedMat;
//	int p[5];
//	capture >> frame;
//
//
//	oclMat Image;
//	Image.upload(frame);
//	ocl::cvtColor(Image, croppedMat, CV_BGR2GRAY);
//
//	ocl::OclCascadeClassifier cascade;
//	if (!cascade.load("../haarcascade_frontalface_alt_gpu.xml")){
//
//		cout << "The haar xml load failed" << endl;
//	}
//
//	//CascadeClassifier  facedCascade;
//	//facedCascade.load("../haarcascade_frontalface_alt_cpu.xml");
//
//
//	double t = (double)cvGetTickCount();
//
//	cascade.detectMultiScale(croppedMat, faces, 1.7, 2, 0, Size(30, 30));
//	//facedCascade.detectMultiScale(frame, faces, 1.5, 2, 0, Size(30, 30));
//	t = (double)cvGetTickCount() - t;
//	printf("Info10: the Face detect cost time is :%g ms.\n", t / ((double)cvGetTickFrequency()*1000.0));
//	frame = Mat(croppedMat);
//	for (int i = 0; i < faces.size(); i++){
//		Rect r = faces[i];
//		//fd.param.est = faces[i];
//		rectangle(frame, faces[i], Scalar(255, 255, 0), 2, 8, 0);
//		printf("Info11:%d face is found ad Rect(%d,%d,%d,%d).\n", i + 1, r.x, r.y, r.width, r.height);
//		p[0] = r.x + r.width*0.5;
//		p[1] = r.y + r.height*0.5;
//		p[2] = r.width;
//		p[3] = r.height;
//		p[4] = 0;
//	}
//
//	for (int i = 0; i < 5; i++) std::cout << p[i] << endl;
//	imshow("FaceDete", frame);
//	cvWaitKey(5000);
//
//	capture.release();
//
//	cvDestroyAllWindows();
//
//
//
//}

void FaceDete_CL::mat3F_to_array(Mat input, af::array& output){


	input.convertTo(input, CV_32FC3);
	int size = input.rows * input.cols;
	int w = input.rows;
	int h = input.cols;
	int temp = 0;
	cout << size << w << h << endl;
	float* r = (float *)malloc(sizeof(float)* size);
	float* g = (float *)malloc(sizeof(float)* size);
	float* b = (float *)malloc(sizeof(float)* size);

	for (int i = 0; i < w; i++){
		for (int j = 0; j < h; j++){
			Vec3f ip = input.at<Vec3f>(i, j);
			r[temp] = ip[2];
			g[temp] = ip[1];
			b[temp] = ip[0];
			temp++;
		}
	}
	output = af::join(2, af::array(h, w, r), af::array(h, w, g), af::array(h, w, b)) / 255.f;
	free(r);
	free(g);
	free(b);
}

 void FaceDete_CL::mat1F_to_array(Mat input, af::array& output){


	input.convertTo(input, CV_32F);
	int size = input.rows * input.cols;
	int w = input.rows;
	int h = input.cols;
	int temp = 0;
	//imshow("win", input);
	//cvWaitKey(6000);
	//float ** Ptr;

	//Ptr = (float **)malloc(sizeof(float **)*w);
	//for (int i = 0; i<w; i++)
	//	Ptr[i] = (float *)malloc(sizeof(float)*h);
	//float *Ptr = (float *)malloc(sizeof(float*)*w*h);
	//float *Ptr = new float[w*h];
	//Ptr = input.ptr<float>(0);
	
	//cout << size <<" " << w <<" " << h << endl;
	//for (int i = 0; i < w; i++)
	//{
	//	float *pp = input.ptr<float>(i);
	//	for (int j = 0; j < h; j++)
	//	{		
	//		Ptr[temp] = pp[i];
	//		temp++;	
	//	}
	//	
	//	
	//	//cout << Ptr[i] << endl;
	//	//for (int j = 0; j < h; j++){		\

	//	//	r[temp] = pp[j];
	//		
	//	//}
	//	delete []pp;
	//}
	//for (int i = 0; i < w*h; i++) cout << Ptr[i] << " ";
	//af_array handle = output.get();	
	dim_t dims[2] = { w, h };
	//af_create_array(&handle, input.ptr<float>(0), 2, dims,u8);
	//af_create_array(handle, Ptr, 2, dims, f32);
	output = array(h, w,  input.ptr<float>(0));

	//af_print(af::transpose(output).row(0));
	//cout << "jieshu";
	//delete Ptr;

}


 void FaceDete_CL::array_to_mat1F(array input, Mat& output){

	 int width = input.dims(0);
		 int height = input.dims(1);
		 //cout << input.dims(0) << " " << input.dims(1) << " " << input.dims(2) << endl;
		 
		float *data = input.host<float>();
		output = (Mat_<float>(2, 3) << data[0],  data[2],data[3],  data[1], data[4], data[5]);
		//cout << output;
	 //try{
	
		 //af_print(input);
		



		 /*af_array handle = input.get();*/
		 //double data[1024];
		 //float *data = (float *)malloc(sizeof(float)*width*height);
		 //af_get_data_ptr(data, handle);



		 //float *data = 
		 //af::Window window(32,32,"2D plot example title");
		 //do{
			// window.image(af::moddims(input,32,32).T().as(u8));
		 //} while (!window.close());

		 ////for (int i = 0; i < 1024;i++) cout << i<<data[i]<<" ";

		 //img = Mat(32, 32, CV_32F, data);
		 //output = img.clone();
		 ////free(data);
		 //delete[] data;
		 //free(handle);
	 //}
	 //catch (Exception& e){
		// printf( "%s \n", e.what());
		// throw;
	 //}
		 

 }

 array FaceDete_CL::array_affparam2mat(array p){
	
	 int sz = 6;
	 array sc = p.row(2);
	 array th = p.row(3);
	 array sr = p.row(4);
	 array phi = p.row(5);
	 array cth = cos(th);
	 array sth = sin(th);
	 array cph = cos(phi);
	 array sph = sin(phi);

	 array ccc = cth*cph*cph;
	 array ccs = cth*cph*sph;
	 array css = cth*sph*sph;
	 array scc = sth*cph*cph;
	 array scs = sth*cph*sph;
	 array sss = sth*sph*sph;

	 array paraMat(sz,p.dims(1));
	 //float Ptr[6] = { p[0], p[1], sc*(ccc + scs + sr*(css - scs)), sc*(sr*(ccs - scc) - ccs - sss), sc*(scc - ccs + sr*(ccs + sss)), sc*(sr*(ccc + scs) - scs + css) };
	 //dim_t dims[2] = { 2, 3 };

	 //af_array handle = paraMat.get();
	 //af_create_array(&handle, Ptr, 2, dims, f32);
	 //paraMat = array(handle);

	 paraMat.rows(0,1) = p.rows(0,1);
	 paraMat.row(2) = sc*(ccc + scs + sr*(css - scs));
	 paraMat.row(3) = sc*(sr*(ccs - scc) - ccs - sss);
	 paraMat.row(4) = sc*(scc - ccs + sr*(ccs + sss));
	 paraMat.row(5) = sc*(sr*(ccc + scs) - scs + css);
	 return paraMat;
 }

// Vector<Mat> FaceDete_CL::array_warpimage(Mat image, array paramat, int size){
//		
//	 int width = size;
//	 int height = size;
//
//	 int n = paramat.dims(1);
//
//	 cout << "length:" << n << endl;
//	 array X = (iota(dim4(1, height), dim4(width, 1)) - (float)height / 2.0);
//	 array Y = (iota(dim4(width, 1), dim4(1, height)) - (float)width / 2.0) ;
//
//	 array ones = constant(1, dim4(width*height, 1));
//	 array in = join(1, ones, flat(X), flat(Y));
//	 array on(3, n * 2);
//	 on.row(0) = join(1, paramat.row(0), paramat.row(1));
//
//	 on.rows(1,2) = join(1, paramat.rows(2,3), paramat.rows(4,5));
//
//	 
//	 array result = matmul(in, on);
//	 cout << result.dims(0) << " " << result.dims(1) << endl;
//	 array result1 = result(span, seq(n, 2*n-1));
//	 Vector<Mat> posVec;
//	 array posTe(width*height,n);
//
//	 //array img = image.as(f32);
//	 //cout << img.dims(0) << " " << img.dims(1) << endl;
//	 ////af_print(result1(span, 0));
//	 //array newImg(width, height, n, f32);
//	 //gfor(seq i, n){
//		// 
//		// //posTe = approx2(image, moddims(result(span, i), dim4(width, height)), moddims(result1(span, i), dim4(width, height)));
//		// try{
//		//	 newImg(span,span,i) = af::approx2(img, moddims(result(span, i), dim4(width, height)), moddims(result1(span, i), dim4(width, height)), AF_INTERP_LINEAR);
//
//		// }
//		// catch (af::exception& e){
//		//	 fprintf(stderr, "%s \n", e.what());
//		//	 throw;
//		// }
//		// //posVec.push_back(posTe);
//	 //}
//
//
//	 //gfor(seq i, n){
//		// Mat output;
//		// Mat pos1;
//		// Mat pos2;
//		// array_to_mat1F(result(span, i), pos1);
//		// array_to_mat1F(result1(span, i), pos2);
//		// cv::remap(image, output, pos1, pos2, cv::INTER_LINEAR);
//		// posVec.push_back(output);
//	 //}
//	 //
//
//	/* timer t = timer::start();
//	 for (int i = 0; i < n;i++){
//		 Mat output;
//		 Mat pos1;
//		 Mat pos2;
//		 array_to_mat1F(result(span, i), pos1);
//		 array_to_mat1F(result1(span, i), pos2);
//		 cv::remap(image, output, pos1, pos2, cv::INTER_LINEAR);
//		 posVec.push_back(output);
//	 }
//	 printf("remap seconds: %g ms\n", timer::stop() * 1000);
//*/	 
//	 return posVec;
//
//	 
//	 //array pos = moddims(matmul(in, on), dim4(width, height, n, 2));
// 
// }

 array FaceDete_CL::array_warpimage(array image, array paramat, int size){

	 int width = size;
	 int height = size;

	 int n = paramat.dims(1);


	 //af::Window window("2D plot example title");
	 //do{
		// window.image(image);
	 //} while (!window.close());
	 //


	 //cout << "length:" << n << endl;
	 array X = (iota(dim4(width, 1), dim4(1, height)) - (float)width / 2.0 + 1);
	 array Y = (iota(dim4(1, height), dim4(width, 1)) - (float)height / 2.0 + 1);
	 
	 //af_print(Y);
	 array ones = constant(1, dim4(width*height, 1));
	 array in = join(1, ones, flat(X), flat(Y));
	 array on(3, n * 2);
	 on.row(0) = join(1, paramat.row(0), paramat.row(1));

	 on.rows(1, 2) = join(1, paramat.rows(2, 3), paramat.rows(4, 5));


	 array result = matmul(in, on);
	 //af_print(result);
	 //cout << result.dims(0) << " " << result.dims(1) << endl;
	 array result1 = result(span, seq(n, 2 * n - 1));
	 Vector<array> posVec;
	 array posTe(width*height, n);

	 array img = image.T().as(f32);
	 //cout << img.dims(0) << " " << img.dims(1) << endl;
	 //af_print(result1(span, 0));
	 //af_print(img);
	 array newImg(width, height, n, f32);
	 //af::Window wnd("Fast Feature");
	 //while (!wnd.close()){
		// wnd.image(img);
	 //}
	 //af_print(result(span, 0));
	 //cout << result(span, 0).dims(0) << "  " << result(span, 0).dims(1) << endl;
	 //cout << result1(span, 0).dims(0) << "  " << result1(span, 0).dims(1) << endl;

	 array paraMatnum = tile(img, 1,1,n);

	 //af::Window wnd("Fast Feature");
	 //while (!wnd.close()){
		// wnd.image(paraMatnum(span, span, 5).as(u8));
	 //}
	
	 gfor(seq i, n){
		 //for (int i = 0; i < n; i++){
		 //posTe = approx2(image, moddims(result(span, i), dim4(width, height)), moddims(result1(span, i), dim4(width, height)));
		 //try{
		 posTe(span, i) = af::approx2(paraMatnum(span, span, i), result1(span, i), result(span, i), AF_INTERP_NEAREST);

		 //}
		 //catch (af::exception& e){
			// fprintf(stderr, "%s \n", e.what());
			// throw;
		 //}
		 
	 }
	 //af_print(posTe.col(0));
	 //Mat show;
	 //array_to_mat1F(posTe.col(0), show);
	 //imshow("win", show);
	 //cvWaitKey(6000); 



	 return posTe;

	 //gfor(seq i, n){
	 // Mat output;
	 // Mat pos1;
	 // Mat pos2;
	 // array_to_mat1F(result(span, i), pos1);
	 // array_to_mat1F(result1(span, i), pos2);
	 // cv::remap(image, output, pos1, pos2, cv::INTER_LINEAR);
	 // posVec.push_back(output);
	 //}
	 //

	 /* timer t = timer::start();
	 for (int i = 0; i < n;i++){
	 Mat output;
	 Mat pos1;
	 Mat pos2;
	 array_to_mat1F(result(span, i), pos1);
	 array_to_mat1F(result1(span, i), pos2);
	 cv::remap(image, output, pos1, pos2, cv::INTER_LINEAR);
	 posVec.push_back(output);
	 }
	 printf("remap seconds: %g ms\n", timer::stop() * 1000);
	 return posVec;
	 */

	 //array pos = moddims(matmul(in, on), dim4(width, height, n, 2));

 }


 array FaceDete_CL::array_affparam2geom(array p){


	 array A = moddims(p.rows(2, 5), 2, 2);
	 array q(6, 1, f32);
	 //af_print(A);

	 af::array U, S, Vt;
	 af::svd(U, S, Vt, A);
	 //timer ti = timer::start();
	 array w = diag(S, 0, false);
	 //


	 //
	 ////af_print(w);
	 ////af_print(U);
	 ////af_print(S);
	 ////af_print(Vt);
	 //float* h_x = U.host<float>();
	 //
	 //size_t t = 0;
	 //float x1 = h_x[t];
	 //float x2 = h_x[t+1];
	 //float x3 = h_x[t+2];
	 //float x4 = h_x[t+3];
	 //if ((x2*x4 - x1*x3 ) < 0){
	 //
	 //
	 //}

	 U = flip(U, 1);
	 Vt = flip(Vt, 1);
	 w = flip(w, 1);

	 //af_print(U);

	 q.rows(0, 1) = p.rows(0, 1);


	 q.row(3) = af::atan2(U(1, 0)*Vt(0, 0) + U(1, 1) *Vt(0, 1), U(0, 0) *Vt(0, 0) + U(0, 1) *Vt(0, 1))*-1;

	 //array phi = atan2(Vt(0, 1), Vt(0, 0));
	 array phi = atan2(Vt(0, 1)*-1, Vt(0, 0));

	 //float data[1] = { -pi / 2.0 };

	 // array result = phi(0, 0) - array(1, 1, data);

	 // result > 0

	 // if ()
	 //{
	 	//float c = cos(-pi / 2.0);
	 	//float s = sin(-pi / 2.0);
	 	//float cs[4] = { c, s, s, c };
	 	//array R = array(2, 2, cs);
	 //array c = af::cos(Pi / -2.0);
	 //af_print(c);
	 //array s = af::sin(Pi / -2.0);
	 //array R;
	 //R(0,0) = c;
	 //R(0, 1) = s;
	 //R(1, 0) = s;
	 //R(1, 1) = c;
	//af_print(R);
	//Vt = Vt*R;
	//w = R*w*R.T();
	//printf("get w  seconds: %g ms\n", timer::stop() * 1000);
		//af_print(w)
	//}
	//if (phi[0] * -1 >= pi / 2.0){
	//	float c = cos(pi / 2.0);
	//	float s = sin(pi / 2.0);
	//	float cs[4] = { c, s, s, c };
	//	array R = array(2, 2, cs);
	//	Vt = Vt*R;
	//	w = R*w*R.T();
	//}
	//af_print(Vt);
	q.row(2) = w(0, 1);

	q.row(4) = w(0, 1) / w(1, 0)*1.0;

	q.row(5) = atan2(Vt(0, 0),Vt(0,1));

	
	//af_print(q);
	return q;

 
 }

 //array FaceDete_CL::array_estwarp_condens(array frm){
 //
	//
	// int n = opt.numsample;//1500
	// int width = tmpl.mean.dims(0);
	// int height = tmpl.mean.dims(1);
	// cout << width << " " << height << " " << endl;
	// int N = height*width;//1024
	// array q;
	// if (param.param.dims(0) != 6)
	// {		
	//	 q = array_affparam2geom(param.est);
	//	 param.param = tile(q,1,n);		
	// }
	// else
	// {
	//	 array cumconf = accum(param.conf);
	//	 array  idx = floor(sum(tile(randn(1, n), n, 1) > tile(cumconf, 1, n)));
	//	 param.param = param.param(span, idx);
	// }
	// //af_print(q);
	// //af_print(param.param);
	//
	//
	// param.param = param.param + randn(6, n)*tile(opt.affsig, 1, n);
	// array para = array_affparam2mat(param.param);
	//
	//  timer ti = timer::start();
	//  array wimgs(32 * 32, n);
	//  //cout << para.dims(0) << " " << para.dims(1) << " " << para.dims(2) << endl;
	//  wimgs = array_warpimage(frm, para, 32);
	// printf("the most cost time is  seconds: %g ms\n", timer::stop() * 1000);
	// return q;
 //}


 array FaceDete_CL::array_estwarp_condens(array frm){


	 int n = opt.numsample;//1500
	 int width = tmpl.mean.dims(0);
	 int height = tmpl.mean.dims(1);
	 cout << width << " " << height << " " << endl;
	 int N = height*width;//1024

	 array q;
	 if (param.param.dims(0) != 6)
	 {
		 q = array_affparam2geom(param.est);
		 param.param = tile(q, 1, n);
	 }
	 else
	 {
		 array cumconf = accum(param.conf);
		 //af_print(cumconf);
		 array  idx = floor(sum(tile(randu(1, n), n, 1) > tile(cumconf, 1, n)));
		 //int *index = idx.host<int>();
		 //af_print(idx);
		 param.param = param.param(span, idx);
		 //af_print(param.param);
		 //delete []index;
	 }
	 
	 //af_print(q);
	 //af_print(param.param);


	 param.param = param.param + (randn(6, n))*tile(opt.affsig, 1, n);

	 array para = array_affparam2mat(param.param);

	 timer ti = timer::start();
	 array wimgs(32 * 32, n);
	 //cout << para.dims(0) << " " << para.dims(1) << " " << para.dims(2) << endl;
	 wimgs = array_warpimage(frm, para, 32);

	


	 //timer tij  = timer::start();
	 //array bbb = tile(ccc, 1,1,n);
	 //array ccc = af::moddims(tmpl.mean,dim4(32,32));
	 //array bbb(1024, 1, n);
	 //cout << tmpl.mean.dims(0) << " " << tmpl.mean.dims(1) << " " << tmpl.mean.dims(2) << endl;
	 //gfor(seq i, n){
	 // bbb(span, span, i) = tmpl.mean;
	 //}
	 af::sync();
	 //printf("the most tile time is  seconds: %g ms\n", timer::stop() * 1000);
	 //int gg;
	 //cin >> gg;

	 array diff = tile(tmpl.mean/256, 1, n) - wimgs/256;
	 array coefdiff;

	printf("the most cost time  warpimage is  seconds: %g ms\n", timer::stop() * 1000);

	 if (tmpl.basis.dims(1) > 0)
	 {
		 //cout << "wocaonima";
		 //cout << tmpl.basis.dims(0) << " " << tmpl.basis.dims(1) << endl;
		 array coef = matmul(tmpl.basis.T(),diff);
		 diff = diff - matmul(tmpl.basis,coef);
		 // update later&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
		 //if (param.param.dims(0) != 0)
		 //{
			// coefdiff = tmpl.reseig*(abs(coef) - abs(param.coef)) / tile(tmpl.eigval, 1, n);
		 //}
		 //else
			// coefdiff = coef*tmpl.reseig / tile(tmpl.eigval, 1, n);
		 //param.coef = coef;
	 }
	 //af_print(diff);
	 //af_print(sum(pow2(diff), -1) / opt.condenssig);
	 param.conf = exp(sum(diff*diff,-1)*(-1) / opt.condenssig).T();
	
	 //param.conf = param.conf / tile(sum(param.conf), n, 1);
	 array sumconf = sum(param.conf, 0);
	 //af_print(sumconf);
	 //try{
		 array parconf = param.conf / tile(sumconf, n, 1);
		 param.conf = parconf;
	 //}
	 //catch (af::exception& e){
		// printf("%s\n", e.what());
	 //}

	 //af_print(param.conf);
	 float maxElement = 0.0;
	 int maxId = 0;
	 timer tamab= timer::start();
	 
	 float *aaa = param.conf.host<float>();
	 for (int i = 0; i < n; i++)
	 {
		 if (aaa[i] > maxElement)
		 {
			 maxElement = aaa[i];
			 maxId = i;
		 }
	 }
	 param.est = array_affparam2mat(param.param.col(maxId));
	 param.wimg = wimgs.col(maxId);
	 //cout << "original :" << wimgs.dims(0) << " " << wimgs.dims(1) << endl;
	 //cout << "original :" << param.wimg.dims(0) << " " << param.wimg.dims(1) << endl;
	 param.err = diff.col(maxId);
	 param.recon = param.wimg + param.err;

	 printf("the most cost time  warpimage is  seconds: %g ms\n", timer::stop() * 1000);

	 return q;
 }


 void FaceDete_CL::array_sklm(array wings){
	 //array basis = tmpl.basis;
	 //array eigval = tmpl.eigval;
	 //array mean = tmpl.mean;
	 //array mean = tmpl.numsample;

	 array U0 = tmpl.basis;
	 array D0 = tmpl.eigval;
	 array mu0 = tmpl.mean;
	 int n0 = tmpl.numsample;

	 int ff = opt.ff;


	 int N = wings.dims(0);
	 int n = wings.dims(1);
	 array mu;
	 af::array U, D, Vt;
	 array S;
	 //cout << "N,n" << N << " " << n << endl;
	 if (U0.dims(0) == 0){
		 if (n == 1){

			 if (U0.dims(0) == 0)
			 {

				 U0 = constant(0, 1, n);
				 D0 = constant(0, 1, 1);
			 }
			 mu = moddims(flat(wings), mu0.dims(0), mu0.dims(1));
			 U = constant(0, N, n);
			 float ones[1] = { 1 };
			 U(1, 1) = array(1, 1, ones);
			 D = constant(0, 1, 1);
		 }
		 else{

			 mu = mean(wings, 1);
			 wings = wings - tile(mu, 1, n);
			 //cout << "wings" << wings.dims(0) << " " << wings.dims(1) << endl;
			 af::svd(U, S, Vt, wings);
			 //D = diag(S, 0, false);
			 //af_print(U);
			 //af_print(S);
			 //af_print(Vt);
			 D = S;
			 mu = moddims(mu, mu0.dims(0), mu0.dims(1));
			 //cout << "U" << U.dims(0) << " " << U.dims(1) << endl;
			 
			 tmpl.basis = U(span, seq(n));
			 tmpl.eigval = D;
			 tmpl.mean = mu;
			 tmpl.numsample = n;
		 }

	 }
	 else{
		 if (mu0.dims(0) != 0){

			 array mu1 = mean(wings, 1);
			 //cout << "mu1:" << mu1.dims(0) << " " << mu1.dims(1) << endl;
			 //cout << "wings0:" << wings.dims(0) << " " << wings.dims(1) << endl;
			 wings = wings - tile(mu1, 1, n);
			 //cout << "wings1:" << wings.dims(0) << " " << wings.dims(1) << endl;
			 wings = join(1, wings, std::sqrt(n*n0 / ((n + n0)*1.0))*(flat(mu0) - mu1));
			 mu = moddims((ff*n0*flat(mu0) + n*mu1) / (n + ff*n0), mu0.dims(0), mu0.dims(1));
			 n = n + ff*n0;
		 }

		 //D = diag(D0, 0, false);
		 //cout << "U0:" << U0.dims(0) << " " << U0.dims(1) << endl;
		 //cout << "wings:" << wings.dims(0) << " " << wings.dims(1) << endl;
		 //af_print(wings);
		 array data_proj = matmul(U0.T(), wings);
		 array data_res = wings - matmul(U0, data_proj);

		 af::array q, r, tau;
		 af::qr(q, r, tau, data_res);
		 //af_print(D);
		 //cout << "q:" << q.dims(0) << " " << q.dims(1) << endl;
		 q = q(span, seq(opt.batchsize+1));
		 array Q = join(1, U0, q);


		 //cout << "q hou:" << q.dims(0) << " " << q.dims(1) << endl;
		 //cout << "Q :" << Q.dims(0) << " " << Q.dims(1) << endl;
		 join(1, ff*diag(D0, 0, false), data_proj);
		 //af_print(constant(0, wings.dims(1), D0.dims(0)));
		 //af_print(af::matmul(q.T(), data_res));
		 join(1, constant(0, wings.dims(1), D0.dims(0)), af::matmul(q.T(), data_res));


		 array R = join(0, join(1, ff*diag(D0, 0, false), data_proj),
			 join(1, constant(0, wings.dims(1), D0.dims(0)), af::matmul(q.T(), data_res)));
		 //cout << "R:" << R.dims(0) << " " << R.dims(1) << endl;
		 af::svd(U, D, Vt, R);
		 //af_print(D);
		 //cout << "U:" << U.dims(0) << " " << U.dims(1) << endl;


		 //D = diag(D);


		 tmpl.basis = matmul(Q, U)(span,seq(opt.batchsize));
		 tmpl.eigval = D(seq(opt.batchsize),span);
		 tmpl.mean = mu;
		 tmpl.numsample = n;
	 }
	 //D = diag(D, 0, false);
 
 }

//void prepareFun(){
//	
//	try
//	{ 
//		cout << "Device name is :" << cv::ocl::Context::getContext()->getDeviceInfo().deviceVendor << endl;
//		
//	}
//	catch (const exception& e)
//	{
//		cout << "Error:" << e.what() << endl;
//	}
//	DevicesInfo info;
//	//int devnums = ocl::getOpenCLDevices(info, CVCL_DEVICE_TYPE_GPU);
//	PlatformsInfo plInfo;
//	int platnums = ocl::getOpenCLPlatforms(plInfo);
//	int devnums = ocl::getOpenCLDevices(info, CVCL_DEVICE_TYPE_GPU);
//	cout << devnums << endl;
//	cout << platnums << endl;
//	cout << "Device name is :" << cv::ocl::Context::getContext()->getDeviceInfo().deviceName << endl;
//	
//	if (devnums < 1){
//		printf("no device found\n");
//	}
//	FaceDete_CL fd;
//	//fd.FaceDete_CL_init();
//	cv::Mat mat(640, 480, CV_8UC3, cv::Scalar(0, 0, 0));
//	cv::ocl::oclMat mat_ocl;
//	//cpu->gpu
//	mat_ocl.upload(mat);
//	mat_ocl.release()
//	//gpu->cpu
//	mat = (cv::Mat)mat_ocl;
//}