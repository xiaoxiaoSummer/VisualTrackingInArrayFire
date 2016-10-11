#pragma once
#include "CompressiveTracker.h"
#include <math.h>
#include <opencv2\ocl\ocl.hpp>
#include <opencv2\ocl\matrix_operations.hpp>

#include <arrayfire.h>



using namespace af;
using namespace cv;
using namespace ocl;
using namespace std;
class FaceDete_CL
{
public:
	FaceDete_CL();
	~FaceDete_CL();

	struct Opt{

		int numsample;
		double condenssig;
		double ff;
		int batchsize;
		int maxbasis;
		array affsig;

	}opt;// = { 1500, 0.25, 0.99, 1, { 4, 4, 0.02, 0.02, 0.005, 0.001 } };

	struct  Tmpl
	{
		array mean;
		array basis;
		array eigval;
		int numsample;
		int reseig;

	}tmpl;

	struct Param{

		array est;
		array wimg;
		array param;
		array conf;
		array err;
		array recon;
		array coef;

	}param;


	struct Drawopt
	{
		array frm;
		array window;
		array basis;
		int showcoef;
		int matcoef;
		int showcondens;
		int thcondens;

	}drawopt;


	/*void FaceDete_CL::FaceDete_CL_init();*/

	void mat3F_to_array(Mat input, af::array& output);


	void mat1F_to_array(Mat input, af::array& output);
	void array_to_mat1F(af::array input, Mat& output);

	Vector<Mat> FaceDete_CL::array_warpimage(Mat image, array paramat, int size);
	array FaceDete_CL::array_warpimage(array image, array paramat, int size);
	array FaceDete_CL::array_affparam2mat(array p);
	array FaceDete_CL::array_affparam2geom(array p);

	array FaceDete_CL::array_estwarp_condens(array frm);

	void FaceDete_CL::array_sklm(array wings);
private:

};

