#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <fstream>

#include <math.h>
#include <cstdlib>

#include <MeshReconstruction.h>
#include <IO.h>

#define PI 3.14159265

using namespace cv;
using namespace std;
using namespace MeshReconstruction;

Vec3f voxel_number;
int N = 8;			//how many cameras/views
int F = 15;			//number of frames
int dim[3];
int decPoint = 1 / 0.01;
////for bounding box computation

typedef struct {
	vector<Point2f> pnts2d;
} campnts;

typedef struct {
	float x, y, z;
} Vector3f;

Vector3f Normalize(const Vector3f &V) {
	float Len = sqrt(V.x * V.x + V.y * V.y + V.z * V.z);
	if (Len == 0.0f) {
		return V;
	} else {
		float Factor = 1.0f / Len;
		Vector3f result;
		result.x = V.x * Factor;
		result.y = V.y * Factor;
		result.z = V.z * Factor;
		//return Vector3f(V.x * Factor, V.y * Factor, V.z * Factor);
		return result;
	}
}

float Dot(const Vector3f &Left, const Vector3f &Right) {
	return (Left.x * Right.x + Left.y * Right.y + Left.z * Right.z);
}

Vector3f Cross(const Vector3f &Left, const Vector3f &Right) {
	Vector3f Result;
	Result.x = Left.y * Right.z - Left.z * Right.y;
	Result.y = Left.z * Right.x - Left.x * Right.z;
	Result.z = Left.x * Right.y - Left.y * Right.x;
	return Result;
}

Vector3f operator *(const Vector3f &Left, float Right) {
	Vector3f result;
	result.x = Left.x * Right;
	result.y = Left.y * Right;
	result.z = Left.z * Right;
	return result;
}

Vector3f operator *(float Left, const Vector3f &Right) {

	Vector3f result;
	result.x = Right.x * Left;
	result.y = Right.y * Left;
	result.z = Right.z * Left;
	return result;

}

Vector3f operator /(const Vector3f &Left, float Right) {

	Vector3f result;
	result.x = Left.x / Right;
	result.y = Left.y / Right;
	result.z = Left.z / Right;
	return result;

}

Vector3f operator +(const Vector3f &Left, const Vector3f &Right) {

	Vector3f result;
	result.x = Left.x + Right.x;
	result.y = Left.y + Right.y;
	result.z = Left.z + Right.z;
	return result;
}

Vector3f operator -(const Vector3f &Left, const Vector3f &Right) {
	Vector3f result;
	result.x = Left.x - Right.x;
	result.y = Left.y - Right.y;
	result.z = Left.z - Right.z;
	return result;
}

Vector3f operator -(const Vector3f &V) {
	Vector3f result;
	result.x = -V.x;
	result.y = -V.y;
	result.z = -V.z;
	return result;
}

typedef struct {
	float a, b, c, d;
	Vector3f normal;
} Plane;

Plane ConstructFromPointNormal(const Vector3f &Pt, const Vector3f &Normal) {
	Plane Result;
	Vector3f NormalizedNormal = Normalize(Normal);
	Result.a = NormalizedNormal.x;
	Result.b = NormalizedNormal.y;
	Result.c = NormalizedNormal.z;
	//Result.d = -Dot(Pt, NormalizedNormal);
	Result.d = Dot(Pt, NormalizedNormal);
	//Result.normal = Normal;
	Result.normal = NormalizedNormal;
	return Result;
}

////from example MC
typedef struct {
	double x, y, z;
} XYZ;

typedef struct {
	XYZ p[8];
	XYZ n[8];
	double val[8];
} GRIDCELL;

typedef struct {
	XYZ p[3]; /* Vertices */
	XYZ c; /* Centroid */
	XYZ n[3]; /* Normal   */
} TRIANGLE;

#define ABS(x) (x < 0 ? -(x) : (x))

// Prototypes
int PolygoniseCube(GRIDCELL, double, TRIANGLE *, Mesh& mesh);
XYZ VertexInterp(double, XYZ, XYZ, double, double);

Mat InitializeVoxels(Vec3f voxel_size, Vec2f xlim, Vec2f ylim, Vec2f zlim,
		vector<Vec4d> voxels, int& total_number, vector<Mat>& silhouettes,
		vector<Mat>& M, Mat& voxel);

Mat VoxelConvertTo3D(Vec3f voxel_number, Vec3f voxel_size, Mat voxel,
		int& total_number, Mat& voxel3D);

int main() {

	for (int countFrame = 0; countFrame < F; countFrame++) {
		//Load data

		int total_number;	//bounding volumes's prod(dims)
		vector<Mat> M; 	//params
		vector<Mat> silhouettes;
		vector<Mat> imageData;
		vector<Vec4d> voxels;
		Mat voxel;
		Mat voxel3D;

		vector<campnts> pnts;
		vector<Mat> points3D;

		vector<Mat> cameraPos;
		vector<Mat> K;
		vector<Mat> Rt;
		vector<Mat> R;
		vector<Mat> Rvec;
		vector<Mat> t;

		vector<Vector3f> cameraOrigins;
		vector<Vector3f> planeNormals;
		vector<Plane> cameraPlanes;
		vector<Point> midpoints;

		for (int countView = 0; countView < N; countView++) {

			cv::String path("data/cam0" + to_string(countView) + "/*.pbm");
			//cout << path << endl;
			vector<String> fn;
			cv::glob(path, fn, true); // recurse

			cv::Mat im = cv::imread(fn[countFrame]);
			//cout << fn[countFrame] << endl;

			if (im.empty())
				continue; //only proceed if sucsessful

			imageData.push_back(im);

			//Compute sils

			Vec3b bgcolor = im.at<Vec3b>(Point(1, 1));

			for (int x = 0; x < im.rows; x++) {
				for (int y = 0; y < im.cols; y++) {
					if (im.at<Vec3b>(x, y) == bgcolor) {
						im.at<Vec3b>(x, y)[0] = 0;
						im.at<Vec3b>(x, y)[1] = 0;
						im.at<Vec3b>(x, y)[2] = 0;

					}
				}
			}

			// without watershed
			//Grayscale matrix
			cv::Mat grayscaleMat(im.size(), CV_8U);

			//Convert BGR to Gray
			cv::cvtColor(im, grayscaleMat, CV_BGR2GRAY);

			//Binary image
			cv::Mat binaryMat(grayscaleMat.size(), grayscaleMat.type());

			//Apply thresholding
			cv::threshold(grayscaleMat, binaryMat, 20, 255, cv::THRESH_BINARY);

			//Show the results
			//cv::namedWindow("silhouettes", cv::WINDOW_AUTOSIZE);
			//cv::imshow("sil", binaryMat);
			//waitKey(0);

			//Sils computation done_____________-------------------->>>

			////Compute Bounding Rects

			Mat threshold_output;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;

			/// Detect edges using Threshold
			threshold(grayscaleMat, threshold_output, 20, 255, THRESH_BINARY);
			//threshold(binaryMat, threshold_output, 20, 255, THRESH_BINARY);
			/// Find contours
			findContours(threshold_output, contours, hierarchy, CV_RETR_TREE,
					CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

			vector<vector<Point> > contours_poly(contours.size());
			vector<Rect> boundRect(contours.size());
			vector<Point2f> center(contours.size());
			float maxArea = 0;
			int BBindex;

			for (int i = 0; i < contours.size(); i++) {
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				boundRect[i] = boundingRect(Mat(contours_poly[i]));

				double a = contourArea(contours[i], false);
				if (a > maxArea) {
					maxArea = a;
					BBindex = i;  //Store the index of largest contour
					//bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
				}
			}

			Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);

			drawContours(drawing, contours, BBindex, Scalar(255), CV_FILLED, 8,
					hierarchy);
			rectangle(drawing, boundRect[BBindex].tl(), boundRect[BBindex].br(),
					Scalar(255, 255, 255), 2, 8, 0);
			Rect test = boundRect[BBindex];
			int x = test.x;
			int y = test.y;
			int width = test.width;
			int height = test.height;
			// Now with those parameters you can calculate the 4 points
			Point2f top_left(x, y);
			Point2f top_right(x + width, y);
			Point2f bottom_left(x, y + height);
			Point2f bottom_right(x + width, y + height);
			campnts camerapnts;
			camerapnts.pnts2d.push_back(top_left);
			camerapnts.pnts2d.push_back(top_right);
			camerapnts.pnts2d.push_back(bottom_left);
			camerapnts.pnts2d.push_back(bottom_right);

			pnts.push_back(camerapnts);

			/// Show in a window
//						namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//						imshow("Contours", drawing);
//						waitKey(0);

			silhouettes.push_back(binaryMat);

			vector<string> fid;

			std::ifstream txtfile(
					"data/cam0" + to_string(countView) + "/cam_par.txt");
			//cout << "data/cam0" + to_string(countView) + "/cam_par.txt" << endl;
			//std::ifstream txtfile("templeSR/templeSR_par.txt");
			std::string line;
			vector<string> linedata;
			//std::getline(txtfile, line);
			//cout<<line<<endl;
			//std::stringstream linestream(line);

//			int value;
			int i = 0;

//			while (linestream >> value) {
//				N = value;
//			}

			while (std::getline(txtfile, line)) {
				std::stringstream linestream(line);
				string val;
				while (linestream >> val) {
					linedata.push_back(val);
					//cout<<val<<endl;
				}
			}
			while (i < linedata.size()) {
				fid.push_back(linedata[i]);
				i++;

				Mat P(3, 4, cv::DataType<float>::type, Scalar(1));
				for (int j = 0; j < 3; j++) {
					for (int k = 0; k < 4; k++) {
						float temp = strtof((linedata[i]).c_str(), 0);

						P.at<float>(j, k) = temp;
						i++;
					}
				}
				M.push_back(P);

				Mat rotm, tvec, kk;
				decomposeProjectionMatrix(P, kk, rotm, tvec);
				K.push_back(kk);
				//		cout << kk << endl << endl;
				R.push_back(rotm);
				Mat ttemp(3, 1, cv::DataType<float>::type, Scalar(1));
				float temp4 = tvec.at<float>(3, 0);
				ttemp.at<float>(0, 0) = tvec.at<float>(0, 0) / temp4;
				ttemp.at<float>(1, 0) = tvec.at<float>(1, 0) / temp4;
				ttemp.at<float>(2, 0) = tvec.at<float>(2, 0) / temp4;

				t.push_back(ttemp);

				//Mat cameraPosition = -R[i].t() * t[i];
				Mat Rtrans = rotm.t();
				Mat cameraPosition = ttemp;
				//Mat Rtrans = rotm;

				Vector3f cameraOrigin;
				cameraOrigin.x = cameraPosition.at<float>(0, 0);
				cameraOrigin.y = cameraPosition.at<float>(0, 1);
				cameraOrigin.z = cameraPosition.at<float>(0, 2);

				Vector3f planeNormal;
				planeNormal.x = Rtrans.at<float>(0, 2);
				planeNormal.y = Rtrans.at<float>(1, 2);
				planeNormal.z = Rtrans.at<float>(2, 2);

				Plane cameraPlane = ConstructFromPointNormal(cameraOrigin,
						planeNormal);

				cameraOrigins.push_back(cameraOrigin);
				planeNormals.push_back(planeNormal);
				cameraPlanes.push_back(cameraPlane);

				//cout<<"camera"<<countView<<": "<<M[countView]<<endl;

			}

		}

		//Read Camera Params from text file ***************************************************

//		vector<cv::Mat> K;
//		vector<cv::Mat> Rt;

		//Bounding box Calculation here
		float xmin = 100;
		float xmax = -100;
		float ymin = 100;
		float ymax = -100;
		float zmin = 100;
		float zmax = -100;

		for (int a = 0; a < N - 1; a++) {

			///check the angle here before calculation
			float dot = Dot(planeNormals[a], planeNormals[a + 1]);
			float lensq1 = (planeNormals[a].x * planeNormals[a].x)
					+ (planeNormals[a].y * planeNormals[a].y)
					+ (planeNormals[a].z * planeNormals[a].z);
			float lensq2 = (planeNormals[a + 1].x * planeNormals[a + 1].x)
					+ (planeNormals[a + 1].y * planeNormals[a + 1].y)
					+ (planeNormals[a + 1].z * planeNormals[a + 1].z);
			float angle = acos(dot / sqrt(lensq1 * lensq2));
			float angleD = angle * 180.0 / PI;

			cout << "Angle in radian : " << angle << ", in Degree : " << angleD
					<< endl;

			if (angleD < 45) {
				Mat temp(4, pnts[0].pnts2d.size(), CV_32F);
				//triangulatePoints(M[0], M[a+1], pnts[0].pnts2d, pnts[a+1].pnts2d, temp);
				triangulatePoints(M[a], M[a + 1], pnts[a].pnts2d,
						pnts[a + 1].pnts2d, temp);
				//temp = temp.t();
				for (int k = 0; k < temp.cols; k++) {
					for (int k = 0; k < temp.cols; k++) {
						for (int j = 0; j < 4; j++) {
							temp.at<float>(j, k) = temp.at<float>(j, k)
									/ temp.at<float>(3, k);
							if (j == 0) {
								if (temp.at<float>(j, k) < xmin) {
									xmin = temp.at<float>(j, k);
									xmin = ceilf(xmin * decPoint) / decPoint;
								}
								if (temp.at<float>(j, k) > xmax) {
									xmax = temp.at<float>(j, k);
									xmax = ceilf(xmax * decPoint) / decPoint;
								}
							} else if (j == 1) {
								if (temp.at<float>(j, k) < ymin) {
									ymin = temp.at<float>(j, k);
									ymin = ceilf(ymin * decPoint) / decPoint;
								}
								if (temp.at<float>(j, k) > ymax) {
									ymax = temp.at<float>(j, k);
									ymax = ceilf(ymax * decPoint) / decPoint;
								}
							} else if (j == 2) {
								if (temp.at<float>(j, k) < zmin) {
									zmin = temp.at<float>(j, k);
									zmin = ceilf(zmin * decPoint) / decPoint;
								}
								if (temp.at<float>(j, k) > zmax) {
									zmax = temp.at<float>(j, k);
									zmax = ceilf(zmax * decPoint) / decPoint;
								}
							}
						}
					}
				}
				points3D.push_back(temp);
				//cout << temp << endl;
			}
		}

		Vec2f xlim(xmin, xmax);
		Vec2f ylim(ymin, ymax);
		Vec2f zlim(zmin, zmax);
//		Vec2f xlim(-0.31, 1.05);
//		Vec2f ylim(-1.69, -1.13);
//		Vec2f zlim(-0.29, 2.05);

		cout << "min is: [ " << xmin << ", " << ymin << ", " << zmin << " ]"
				<< endl;
		cout << "max is: [ " << xmax << ", " << ymax << ", " << zmax << " ]"
				<< endl;

		//Set resolution after BB calculation
		Vec3f voxel_size(0.01, 0.01, 0.01);	//resolution

		//initialize voxels
		Mat voxels_voted = InitializeVoxels(voxel_size, xlim, ylim, zlim,
				voxels, total_number, silhouettes, M, voxel);
		cout << "voxels voting done!" << endl;

		voxel3D = VoxelConvertTo3D(voxel_number, voxel_size, voxels_voted,
				total_number, voxel3D);
		cout << "voxel3D conversion done!" << endl;

		float error = 5;
		float maxv = 0;
		float minv = 10;
		for (int i = 0; i < total_number; i++) {
			if (voxels_voted.at<float>(i, 3) > maxv) {
				maxv = voxels_voted.at<float>(i, 3);
				cout << "maxv updated, now: " << maxv << endl;
			} else if (voxels_voted.at<float>(i, 3) < minv) {
				minv = voxels_voted.at<float>(i, 3);
			}
		}

		float iso_value = maxv - round((maxv / 100) * error) - 0.5;

		int i, j, k, l, n;

		short int data[dim[0]][dim[1]][dim[2]];

		GRIDCELL grid;
		TRIANGLE triangles[10];
		vector<TRIANGLE> tri;

		int ntri = 0;
		FILE *fptr;
		Mesh mesh;

		double isolevel = iso_value * 0.925;

		for (k = dim[2]; k >= 0; k--) {
			for (j = 0; j < dim[1]; j++) {
				for (i = 0; i < dim[0]; i++) {
					data[i][j][k] = voxel3D.at<float>(i, j, k);
				}
			}
		}

		fprintf(stderr, "Polygonising data ...\n");
		for (i = 0; i < dim[0] - 1; i++) {
			if (i % (dim[0] / 10) == 0)
				fprintf(stderr, "   Slice %d of %d\n", i, dim[0]);
			for (j = 0; j < dim[1] - 1; j++) {
				for (k = 0; k < dim[2] - 1; k++) {

					grid.p[0].x = i;
					grid.p[0].y = j;
					grid.p[0].z = k;
					grid.val[0] = data[i][j][k];
					grid.p[1].x = i + 1;
					grid.p[1].y = j;
					grid.p[1].z = k;
					grid.val[1] = data[i + 1][j][k];
					grid.p[2].x = i + 1;
					grid.p[2].y = j + 1;
					grid.p[2].z = k;
					grid.val[2] = data[i + 1][j + 1][k];
					grid.p[3].x = i;
					grid.p[3].y = j + 1;
					grid.p[3].z = k;
					grid.val[3] = data[i][j + 1][k];
					grid.p[4].x = i;
					grid.p[4].y = j;
					grid.p[4].z = k + 1;
					grid.val[4] = data[i][j][k + 1];
					grid.p[5].x = i + 1;
					grid.p[5].y = j;
					grid.p[5].z = k + 1;
					grid.val[5] = data[i + 1][j][k + 1];
					grid.p[6].x = i + 1;
					grid.p[6].y = j + 1;
					grid.p[6].z = k + 1;
					grid.val[6] = data[i + 1][j + 1][k + 1];
					grid.p[7].x = i;
					grid.p[7].y = j + 1;
					grid.p[7].z = k + 1;
					grid.val[7] = data[i][j + 1][k + 1];

					n = PolygoniseCube(grid, isolevel, triangles, mesh);

					for (l = 0; l < n; l++) {
						tri.push_back(triangles[l]);
					}
					ntri += n;
				}
			}
		}

		////for outputting in .off file

//		cout<<outputfilename<<endl;
//
//		if ((fptr = fopen("output/output.off", "w")) == NULL) {
//			fprintf(stderr, "Failed to open .off file!\n");
//			exit(-1);
//		}
//
//		int numVerts = ntri * 3;
//		cout << "NumVerts: " << numVerts << " NumTri: " << ntri << endl;
//
//		fprintf(fptr, "OFF\n");
//		fprintf(fptr, "%d %d %d\n", numVerts, ntri, 0);
//
//		for (i = 0; i < ntri; i++) {
//			for (k = 0; k < 3; k++) {
//				fprintf(fptr, "%f %f %f\n", tri[i].p[k].x, tri[i].p[k].y,
//						tri[i].p[k].z);
//			}
//		}
//		int vertCount = 0;
//		for (i = 0; i < ntri; i++) {
//			fprintf(fptr, "3 ");
//			fprintf(fptr, "%i %i %i\n", vertCount, vertCount + 1,
//					vertCount + 2);
//			vertCount += 3;
//		}
//
//		fclose(fptr);
//		printf("Output wrote in .off format!\n");

		///// writing output

//		string outputfilename = "output/output" + to_string(countFrame)
//				+ ".off";

		//// .off output
//		ofstream myfile;
//		myfile.open(outputfilename);
//		int numVerts = ntri * 3;
//
//		myfile << "OFF\n";
//		myfile << numVerts << " " << ntri << " " << 0 << "\n";
//		for (i = 0; i < ntri; i++) {
//			for (k = 0; k < 3; k++) {
//				myfile << tri[i].p[k].x << " " << tri[i].p[k].y << " "
//						<< tri[i].p[k].z << "\n";
//			}
//		}
//		int vertCount = 0;
//		for (i = 0; i < ntri; i++) {
//			myfile << "3 ";
//			myfile << vertCount << " " << vertCount + 1 << " " << vertCount + 2
//					<< "\n";
//			vertCount += 3;
//		}
//
//		myfile.close();
//		printf("Output wrote in .off format!\n");


		////For .obj file
		string outputfilename = "output/output" + to_string(countFrame)
						+ ".obj";
		WriteObjFile(mesh, outputfilename);
		printf("Output wrote in .obj file!\n");

		//Release & delete
		voxel.release();
		voxel3D.release();
		voxels_voted.release();

	}

	return 0;
}

Mat InitializeVoxels(Vec3f voxel_size, Vec2f xlim, Vec2f ylim, Vec2f zlim,
		vector<Vec4d> voxels, int& total_number, vector<Mat>& silhouettes,
		vector<Mat>& M, Mat& voxel) {

	voxel_number[0] = (xlim[1] - xlim[0]) / voxel_size[0];
	voxel_number[1] = (ylim[1] - ylim[0]) / voxel_size[1];
	voxel_number[2] = (zlim[1] - zlim[0]) / voxel_size[2];

	total_number = ((voxel_number[0] + 1) * (voxel_number[1] + 1)
			* (voxel_number[2] + 1));
	cout << total_number << endl;

	voxel = Mat(total_number, 4, cv::DataType<float>::type, Scalar(1));

	float sx = xlim[0];
	float ex = xlim[1];
	float sy = ylim[0];
	float ey = ylim[1];
	float sz = zlim[0];
	float ez = zlim[1];

	float x_step;
	float y_step;
	float z_step;

	if (ex > sx) {
		x_step = voxel_size[0];
	} else {
		x_step = -voxel_size[0];
	}
	if (ey > sy) {
		y_step = voxel_size[1];
	} else {
		y_step = -voxel_size[1];
	}
	if (ez > sz) {
		z_step = voxel_size[2];
	} else {
		z_step = -voxel_size[2];
	}

	dim[0] = voxel_number[0] + 1;
	dim[1] = voxel_number[1] + 1;
	dim[2] = voxel_number[2] + 2;

	cout << dim[0] << endl;

	int i = 0;
	//int a, b, c;
	//float x, y, z;

	while (i < total_number) {
		for (float z = ez; z >= sz; z -= z_step) {
			for (float x = sx; x <= ex; x += x_step) {
				for (float y = sy; y <= ey; y += y_step) {
					voxel.at<float>(i, 0) = x;
					voxel.at<float>(i, 1) = y;
					voxel.at<float>(i, 2) = z;
					voxel.at<float>(i, 3) = 1;
					i++;
				}
			}
		}
	}

	cout << total_number << endl;
	Mat obj_points_3D = voxel.t();

	for (int i = 0; i < total_number; i++) {
		voxel.at<float>(i, 3) = 0;
	}

	Mat points2d;
	int imgH = silhouettes[1].rows;
	int imgW = silhouettes[1].cols;
	int num = 0;

	while (num < N) {
		//cout<<M[0].at<float>(0,1)<<endl;
		Mat camParam = M[num];

		Mat curr_sil = silhouettes[num];

		Mat curr_sil_bin(imgH, imgW, CV_32F);

		int count1 = 0;
		for (int row = 0; row < imgH; row++) {
			for (int col = 0; col < imgW; col++) {
				if (curr_sil.at<uchar>(row, col) != 0) {
					curr_sil_bin.at<float>(row, col) = 1;
					count1++;
				} else
					curr_sil_bin.at<float>(row, col) = 0;
			}
		}

		points2d = camParam * obj_points_3D;

		for (int c = 0; c < total_number; c++) {
			for (int r = 0; r < 3; r++) {
				points2d.at<float>(r, c) = floor(
						points2d.at<float>(r, c) / points2d.at<float>(2, c));

				if (points2d.at<float>(r, c) <= 0)
					points2d.at<float>(r, c) = 0;//condition applied for negativeness

				if (r == 0 && points2d.at<float>(r, c) > imgW) {
					points2d.at<float>(0, c) = 1;
					points2d.at<float>(1, c) = 1;
					points2d.at<float>(2, c) = 1;
				} else if (r == 1 && points2d.at<float>(r, c) > imgH) {
					points2d.at<float>(0, c) = 1;
					points2d.at<float>(1, c) = 1;
					points2d.at<float>(2, c) = 1;

				}
				int row = points2d.at<float>(1, c);
				int col = points2d.at<float>(0, c);
				if (row < 0 || row > imgH - 1)
					row = 0;
				else if (col < 0 || col > imgW - 1)
					col = 0;
				voxel.at<float>(c, 3) += curr_sil_bin.at<float>(row, col);

			}

		}

		num++;
	}

	return voxel;

}

Mat VoxelConvertTo3D(Vec3f voxel_number, Vec3f voxel_size, Mat voxel,
		int& total_number, Mat& voxel3D) {

	voxel3D = Mat(3, dim, CV_32FC1, Scalar(0));

	//float sx, ex, sy, ey, sz, ez;
	int l, x1, y1, z1;

//	sx = -(voxel_number[0] / 2) * voxel_size[0];
//	ex = (voxel_number[0] / 2) * voxel_size[0];
//	sy = -(voxel_number[1] / 2) * voxel_size[1];
//	ey = (voxel_number[1] / 2) * voxel_size[1];
//	sz = 0;
//	ez = voxel_number[2] * voxel_size[2];

	cout << dim[0] << ", " << dim[1] << ", " << dim[2] << endl;

	l = 0;
	z1 = 0;
//	cout<<"problem here: "<<voxel.at<float>(185312, 3)<<endl;
//	cout<<voxel3D.at<float>(27, 39, 237)<<endl;
	while (l < total_number) {
		//z1=0;
		for (z1 = 0; z1 < dim[2]; z1++) {
			for (x1 = 0; x1 < dim[0]; x1++) {
				for (y1 = 0; y1 < dim[1]; y1++) {

					voxel3D.at<float>(x1, y1, z1) = voxel.at<float>(l, 3);
					l++;
				}
			}
		}
	}

	return voxel3D;
}

int PolygoniseCube(GRIDCELL g, double iso, TRIANGLE *tri, Mesh& mesh) {
	int i, ntri = 0;
	int cubeindex;
	XYZ vertlist[12];
	/*
	 int edgeTable[256].  It corresponds to the 2^8 possible combinations of
	 of the eight (n) vertices either existing inside or outside (2^n) of the
	 surface.  A vertex is inside of a surface if the value at that vertex is
	 less than that of the surface you are scanning for.  The table index is
	 constructed bitwise with bit 0 corresponding to vertex 0, bit 1 to vert
	 1.. bit 7 to vert 7.  The value in the table tells you which edges of
	 the table are intersected by the surface.  Once again bit 0 corresponds
	 to edge 0 and so on, up to edge 12.
	 Constructing the table simply consisted of having a program run thru
	 the 256 cases and setting the edge bit if the vertices at either end of
	 the edge had different values (one is inside while the other is out).
	 The purpose of the table is to speed up the scanning process.  Only the
	 edges whose bit's are set contain vertices of the surface.
	 Vertex 0 is on the bottom face, back edge, left side.
	 The progression of vertices is clockwise around the bottom face
	 and then clockwise around the top face of the cube.  Edge 0 goes from
	 vertex 0 to vertex 1, Edge 1 is from 2->3 and so on around clockwise to
	 vertex 0 again. Then Edge 4 to 7 make up the top face, 4->5, 5->6, 6->7
	 and 7->4.  Edge 8 thru 11 are the vertical edges from vert 0->4, 1->5,
	 2->6, and 3->7.
	 4--------5     *---4----*
	 /|       /|    /|       /|
	 / |      / |   7 |      5 |
	 /  |     /  |  /  8     /  9
	 7--------6   | *----6---*   |
	 |   |    |   | |   |    |   |
	 |   0----|---1 |   *---0|---*
	 |  /     |  /  11 /     10 /
	 | /      | /   | 3      | 1
	 |/       |/    |/       |/
	 3--------2     *---2----*
	 */
	int edgeTable[256] = { 0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
			0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99,
			0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
			0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636,
			0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33,
			0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
			0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460,
			0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f,
			0xf66, 0x86a, 0x963, 0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa,
			0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3,
			0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
			0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0,
			0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf,
			0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca,
			0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3,
			0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55,
			0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0,
			0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff,
			0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a,
			0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663,
			0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5,
			0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
			0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435,
			0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a,
			0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a,
			0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905,
			0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };

	/*
	 int triTable[256][16] also corresponds to the 256 possible combinations
	 of vertices.
	 The [16] dimension of the table is again the list of edges of the cube
	 which are intersected by the surface.  This time however, the edges are
	 enumerated in the order of the vertices making up the triangle mesh of
	 the surface.  Each edge contains one vertex that is on the surface.
	 Each triple of edges listed in the table contains the vertices of one
	 triangle on the mesh.  The are 16 entries because it has been shown that
	 there are at most 5 triangles in a cube and each "edge triple" list is
	 terminated with the value -1.
	 For example triTable[3] contains
	 {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	 This corresponds to the case of a cube whose vertex 0 and 1 are inside
	 of the surface and the rest of the verts are outside (00000001 bitwise
	 OR'ed with 00000010 makes 00000011 == 3).  Therefore, this cube is
	 intersected by the surface roughly in the form of a plane which cuts
	 edges 8,9,1 and 3.  This quadrilateral can be constructed from two
	 triangles: one which is made of the intersection vertices found on edges
	 1,8, and 3; the other is formed from the vertices on edges 9,8, and 1.
	 Remember, each intersected edge contains only one surface vertex.  The
	 vertex triples are listed in counter clockwise order for proper facing.
	 */
	int triTable[256][16] = { { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1, -1, -1 }, { 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1, -1 }, { 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
			-1, -1 },
			{ 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
					2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 9,
					2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
					9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1 }, { 9,
					8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 4,
					1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
					2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1 }, {
					9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 }, {
					2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1 }, { 8,
					4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1 }, {
					9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 }, {
					4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1 }, {
					3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1 },
			{ 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1 }, { 4, 7,
					8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1 }, { 4, 7,
					11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1 }, { 9,
					5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 0,
					5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1 }, {
					1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 }, { 5, 2,
					10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1 }, { 2, 10,
					5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1 }, { 9, 5, 4,
					2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 11,
					2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1 }, { 0, 5,
					4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1 }, { 2, 1,
					5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1 }, { 10, 3,
					11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1 }, { 4, 9,
					5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1 }, { 5, 4,
					0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1 }, { 5, 4,
					8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1 }, { 9,
					7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1 }, {
					0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1 }, {
					1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1 }, { 10, 1,
					2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1 }, { 8, 0, 2,
					8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1 }, { 2, 10, 5, 2,
					5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1 }, { 7, 9, 5, 7,
					8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1 }, { 9, 5, 7, 9,
					7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1 }, { 2, 3, 11, 0, 1,
					8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1 }, { 11, 2, 1, 11, 1, 7,
					7, 1, 5, -1, -1, -1, -1, -1, -1, -1 }, { 9, 5, 8, 8, 5, 7,
					10, 1, 3, 10, 3, 11, -1, -1, -1, -1 }, { 5, 7, 0, 5, 0, 9,
					7, 11, 0, 1, 0, 10, 11, 10, 0, -1 }, { 11, 10, 0, 11, 0, 3,
					10, 5, 0, 8, 0, 7, 5, 7, 0, -1 }, { 11, 10, 5, 7, 11, 5, -1,
					-1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 10, 6, 5, -1, -1,
					-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 8, 3, 5,
					10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 9, 0, 1,
					5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 8,
					3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 }, { 1, 6,
					5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
					6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1 }, { 9,
					6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1 }, { 5,
					9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1 }, { 2, 3,
					11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1 }, { 5,
					10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1 }, { 6,
					3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1 }, { 0,
					8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1 }, { 3,
					11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1 }, { 6, 5,
					9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1 }, { 5,
					10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1 }, {
					1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1 }, {
					10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1 }, { 6,
					1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1 }, { 1,
					2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1 }, { 8, 4,
					7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1 }, { 7, 3, 9,
					7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1 }, { 3, 11, 2, 7, 8,
					4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1 }, { 5, 10, 6, 4, 7,
					2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1 }, { 0, 1, 9, 4, 7, 8,
					2, 3, 11, 5, 10, 6, -1, -1, -1, -1 }, { 9, 2, 1, 9, 11, 2,
					9, 4, 11, 7, 11, 4, 5, 10, 6, -1 }, { 8, 4, 7, 3, 11, 5, 3,
					5, 1, 5, 11, 6, -1, -1, -1, -1 }, { 5, 1, 11, 5, 11, 6, 1,
					0, 11, 7, 11, 4, 0, 4, 11, -1 }, { 0, 5, 9, 0, 6, 5, 0, 3,
					6, 11, 6, 3, 8, 4, 7, -1 }, { 6, 5, 9, 6, 9, 11, 4, 7, 9, 7,
					11, 9, -1, -1, -1, -1 }, { 10, 4, 9, 6, 4, 10, -1, -1, -1,
					-1, -1, -1, -1, -1, -1, -1 }, { 4, 10, 6, 4, 9, 10, 0, 8, 3,
					-1, -1, -1, -1, -1, -1, -1 }, { 10, 0, 1, 10, 6, 0, 6, 4, 0,
					-1, -1, -1, -1, -1, -1, -1 }, { 8, 3, 1, 8, 1, 6, 8, 6, 4,
					6, 1, 10, -1, -1, -1, -1 }, { 1, 4, 9, 1, 2, 4, 2, 6, 4, -1,
					-1, -1, -1, -1, -1, -1 }, { 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6,
					4, -1, -1, -1, -1 }, { 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1,
					-1, -1, -1, -1, -1 }, { 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1,
					-1, -1, -1, -1, -1 }, { 10, 4, 9, 10, 6, 4, 11, 2, 3, -1,
					-1, -1, -1, -1, -1, -1 }, { 0, 8, 2, 2, 8, 11, 4, 9, 10, 4,
					10, 6, -1, -1, -1, -1 }, { 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1,
					10, -1, -1, -1, -1 }, { 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1,
					11, 8, 11, 1, -1 }, { 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3,
					-1, -1, -1, -1 }, { 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6,
					4, 1, -1 }, { 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1,
					-1, -1, -1 }, { 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1,
					-1, -1, -1, -1 }, { 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1,
					-1, -1, -1, -1, -1 }, { 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7,
					10, -1, -1, -1, -1 }, { 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8,
					0, -1, -1, -1, -1 }, { 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1,
					-1, -1, -1, -1, -1 }, { 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7,
					-1, -1, -1, -1 }, { 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7,
					3, 9, -1 }, { 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1,
					-1, -1 }, { 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1,
					-1, -1, -1 }, { 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1,
					-1, -1, -1 }, { 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10,
					7, -1 }, { 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11,
					-1 }, { 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1,
					-1 }, { 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1 },
			{ 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 7,
					8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1 }, { 7,
					11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 8,
					1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 }, { 10,
					1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 }, {
					2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1 }, {
					6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1 }, {
					7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1 }, { 2, 7,
					6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1 }, { 1, 6,
					2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1 }, { 10, 7, 6,
					10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1 }, { 10, 7, 6,
					1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1 }, { 0, 3, 7, 0,
					7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1 }, { 7, 6, 10, 7,
					10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1 }, { 6, 8, 4,
					11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 3, 6,
					11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1 }, { 8, 6,
					11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1 }, { 9, 4,
					6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1 }, { 6, 8, 4,
					6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 2,
					10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1 }, { 4, 11,
					8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1 }, { 10, 9,
					3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1 }, { 8, 2, 3, 8,
					4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1 }, { 0, 4, 2, 4,
					6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 9, 0,
					2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1 }, { 1, 9, 4, 1,
					4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1 }, { 8, 1, 3, 8,
					6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1 }, { 10, 1, 0, 10,
					0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1 }, { 4, 6, 3, 4,
					3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1 }, { 10, 9, 4, 6, 10,
					4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 4, 9, 5, 7,
					6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 8, 3,
					4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1 }, { 5, 0, 1,
					5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 }, { 11, 7, 6,
					8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1 }, { 9, 5, 4, 10,
					1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1 }, { 6, 11, 7, 1,
					2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1 }, { 7, 6, 11, 5, 4,
					10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1 }, { 3, 4, 8, 3, 5, 4,
					3, 2, 5, 10, 5, 2, 11, 7, 6, -1 }, { 7, 2, 3, 7, 6, 2, 5, 4,
					9, -1, -1, -1, -1, -1, -1, -1 }, { 9, 5, 4, 0, 8, 6, 0, 6,
					2, 6, 8, 7, -1, -1, -1, -1 }, { 3, 6, 2, 3, 7, 6, 1, 5, 0,
					5, 4, 0, -1, -1, -1, -1 }, { 6, 2, 8, 6, 8, 7, 2, 1, 8, 4,
					8, 5, 1, 5, 8, -1 }, { 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7,
					-1, -1, -1, -1 }, { 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9,
					5, 4, -1 }, { 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7,
					10, -1 }, { 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1,
					-1, -1 }, { 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1,
					-1, -1 }, { 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1,
					-1 }, { 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1,
					-1 }, { 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1,
					-1 }, { 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1,
					-1 },
			{ 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1 }, { 11, 8, 5,
					11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1 }, { 6, 11, 3, 6,
					3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1 }, { 5, 8, 9, 5, 2,
					8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1 }, { 9, 5, 6, 9, 6, 0,
					0, 6, 2, -1, -1, -1, -1, -1, -1, -1 }, { 1, 5, 8, 1, 8, 0,
					5, 6, 8, 3, 8, 2, 6, 2, 8, -1 }, { 1, 5, 6, 2, 1, 6, -1, -1,
					-1, -1, -1, -1, -1, -1, -1, -1 }, { 1, 3, 6, 1, 6, 10, 3, 8,
					6, 5, 6, 9, 8, 9, 6, -1 }, { 10, 1, 0, 10, 0, 6, 9, 5, 0, 5,
					6, 0, -1, -1, -1, -1 }, { 0, 3, 8, 5, 6, 10, -1, -1, -1, -1,
					-1, -1, -1, -1, -1, -1 }, { 10, 5, 6, -1, -1, -1, -1, -1,
					-1, -1, -1, -1, -1, -1, -1, -1 }, { 11, 5, 10, 7, 5, 11, -1,
					-1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 11, 5, 10, 11, 7, 5,
					8, 3, 0, -1, -1, -1, -1, -1, -1, -1 }, { 5, 11, 7, 5, 10,
					11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1 }, { 10, 7, 5, 10,
					11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1 }, { 11, 1, 2, 11,
					7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1 }, { 0, 8, 3, 1,
					2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1 }, { 9, 7, 5, 9, 2,
					7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1 }, { 7, 5, 2, 7, 2, 11,
					5, 9, 2, 3, 2, 8, 9, 8, 2, -1 }, { 2, 5, 10, 2, 3, 5, 3, 7,
					5, -1, -1, -1, -1, -1, -1, -1 }, { 8, 2, 0, 8, 5, 2, 8, 7,
					5, 10, 2, 5, -1, -1, -1, -1 }, { 9, 0, 1, 5, 10, 3, 5, 3, 7,
					3, 10, 2, -1, -1, -1, -1 }, { 9, 8, 2, 9, 2, 1, 8, 7, 2, 10,
					2, 5, 7, 5, 2, -1 }, { 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1,
					-1, -1, -1, -1, -1 }, { 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1,
					-1, -1, -1, -1, -1 }, { 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1,
					-1, -1, -1, -1, -1 }, { 9, 8, 7, 5, 9, 7, -1, -1, -1, -1,
					-1, -1, -1, -1, -1, -1 }, { 5, 8, 4, 5, 10, 8, 10, 11, 8,
					-1, -1, -1, -1, -1, -1, -1 }, { 5, 0, 4, 5, 11, 0, 5, 10,
					11, 11, 3, 0, -1, -1, -1, -1 }, { 0, 1, 9, 8, 4, 10, 8, 10,
					11, 10, 4, 5, -1, -1, -1, -1 }, { 10, 11, 4, 10, 4, 5, 11,
					3, 4, 9, 4, 1, 3, 1, 4, -1 }, { 2, 5, 1, 2, 8, 5, 2, 11, 8,
					4, 5, 8, -1, -1, -1, -1 }, { 0, 4, 11, 0, 11, 3, 4, 5, 11,
					2, 11, 1, 5, 1, 11, -1 }, { 0, 2, 5, 0, 5, 9, 2, 11, 5, 4,
					5, 8, 11, 8, 5, -1 }, { 9, 4, 5, 2, 11, 3, -1, -1, -1, -1,
					-1, -1, -1, -1, -1, -1 }, { 2, 5, 10, 3, 5, 2, 3, 4, 5, 3,
					8, 4, -1, -1, -1, -1 }, { 5, 10, 2, 5, 2, 4, 4, 2, 0, -1,
					-1, -1, -1, -1, -1, -1 }, { 3, 10, 2, 3, 5, 10, 3, 8, 5, 4,
					5, 8, 0, 1, 9, -1 }, { 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2,
					-1, -1, -1, -1 }, { 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1,
					-1, -1, -1, -1 }, { 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1,
					-1, -1, -1, -1, -1 }, { 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5,
					-1, -1, -1, -1 }, { 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1,
					-1, -1, -1, -1, -1 }, { 4, 11, 7, 4, 9, 11, 9, 10, 11, -1,
					-1, -1, -1, -1, -1, -1 }, { 0, 8, 3, 4, 9, 7, 9, 11, 7, 9,
					10, 11, -1, -1, -1, -1 }, { 1, 10, 11, 1, 11, 4, 1, 4, 0, 7,
					4, 11, -1, -1, -1, -1 }, { 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4,
					11, 10, 11, 4, -1 }, { 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1,
					2, -1, -1, -1, -1 }, { 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11,
					1, 0, 8, 3, -1 }, { 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1,
					-1, -1, -1, -1 }, { 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4,
					-1, -1, -1, -1 }, { 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1,
					-1, -1, -1 }, { 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0,
					7, -1 }, { 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10,
					-1 }, { 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1,
					-1, -1 }, { 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1,
					-1, -1 }, { 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1,
					-1 }, { 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1,
					-1, -1 }, { 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
					-1, -1, -1 }, { 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1,
					-1, -1, -1, -1 }, { 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1,
					-1, -1, -1, -1, -1 }, { 0, 1, 10, 0, 10, 8, 8, 10, 11, -1,
					-1, -1, -1, -1, -1, -1 }, { 3, 1, 10, 11, 3, 10, -1, -1, -1,
					-1, -1, -1, -1, -1, -1, -1 }, { 1, 2, 11, 1, 11, 9, 9, 11,
					8, -1, -1, -1, -1, -1, -1, -1 }, { 3, 0, 9, 3, 9, 11, 1, 2,
					9, 2, 11, 9, -1, -1, -1, -1 }, { 0, 2, 11, 8, 0, 11, -1, -1,
					-1, -1, -1, -1, -1, -1, -1, -1 }, { 3, 2, 11, -1, -1, -1,
					-1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 2, 3, 8, 2, 8,
					10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1 }, { 9, 10, 2, 0,
					9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 2, 3, 8,
					2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1 }, { 1, 10, 2,
					-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, { 1,
					3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
			{ 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }, {
					-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
					-1 } };

	/*
	 Determine the index into the edge table which
	 tells us which vertices are inside of the surface
	 */
	cubeindex = 0;
	if (g.val[0] < iso)
		cubeindex |= 1;
	if (g.val[1] < iso)
		cubeindex |= 2;
	if (g.val[2] < iso)
		cubeindex |= 4;
	if (g.val[3] < iso)
		cubeindex |= 8;
	if (g.val[4] < iso)
		cubeindex |= 16;
	if (g.val[5] < iso)
		cubeindex |= 32;
	if (g.val[6] < iso)
		cubeindex |= 64;
	if (g.val[7] < iso)
		cubeindex |= 128;

	/* Cube is entirely in/out of the surface */
	if (edgeTable[cubeindex] == 0)
		return (0);

	/* Find the vertices where the surface intersects the cube */
	if (edgeTable[cubeindex] & 1) {
		vertlist[0] = VertexInterp(iso, g.p[0], g.p[1], g.val[0], g.val[1]);
	}
	if (edgeTable[cubeindex] & 2) {
		vertlist[1] = VertexInterp(iso, g.p[1], g.p[2], g.val[1], g.val[2]);
	}
	if (edgeTable[cubeindex] & 4) {
		vertlist[2] = VertexInterp(iso, g.p[2], g.p[3], g.val[2], g.val[3]);
	}
	if (edgeTable[cubeindex] & 8) {
		vertlist[3] = VertexInterp(iso, g.p[3], g.p[0], g.val[3], g.val[0]);
	}
	if (edgeTable[cubeindex] & 16) {
		vertlist[4] = VertexInterp(iso, g.p[4], g.p[5], g.val[4], g.val[5]);
	}
	if (edgeTable[cubeindex] & 32) {
		vertlist[5] = VertexInterp(iso, g.p[5], g.p[6], g.val[5], g.val[6]);
	}
	if (edgeTable[cubeindex] & 64) {
		vertlist[6] = VertexInterp(iso, g.p[6], g.p[7], g.val[6], g.val[7]);
	}
	if (edgeTable[cubeindex] & 128) {
		vertlist[7] = VertexInterp(iso, g.p[7], g.p[4], g.val[7], g.val[4]);
	}
	if (edgeTable[cubeindex] & 256) {
		vertlist[8] = VertexInterp(iso, g.p[0], g.p[4], g.val[0], g.val[4]);
	}
	if (edgeTable[cubeindex] & 512) {
		vertlist[9] = VertexInterp(iso, g.p[1], g.p[5], g.val[1], g.val[5]);
	}
	if (edgeTable[cubeindex] & 1024) {
		vertlist[10] = VertexInterp(iso, g.p[2], g.p[6], g.val[2], g.val[6]);
	}
	if (edgeTable[cubeindex] & 2048) {
		vertlist[11] = VertexInterp(iso, g.p[3], g.p[7], g.val[3], g.val[7]);
	}

	/* Create the triangles */
	for (i = 0; triTable[cubeindex][i] != -1; i += 3) {
		tri[ntri].p[0] = vertlist[triTable[cubeindex][i]];
		tri[ntri].p[1] = vertlist[triTable[cubeindex][i + 1]];
		tri[ntri].p[2] = vertlist[triTable[cubeindex][i + 2]];

		Vec3 v0 { (double) tri[ntri].p[0].x, (double) tri[ntri].p[0].y,
				(double) tri[ntri].p[0].z };
		Vec3 v1 { (double) tri[ntri].p[1].x, (double) tri[ntri].p[1].y,
				(double) tri[ntri].p[1].z };
		Vec3 v2 { (double) tri[ntri].p[2].x, (double) tri[ntri].p[2].y,
				(double) tri[ntri].p[2].z };

		mesh.vertices.push_back(v0);
		mesh.vertices.push_back(v1);
		mesh.vertices.push_back(v2);

		Vec3 V = v1 - v0;
		Vec3 W = v2 - v0;

		Vec3 normal0 { 1, 0, 0 };
		Vec3 normal1 { 0, 1, 0 };
		Vec3 normal2 { 0, 0, 1 };

		normal2.x = normal1.x = normal0.x = V.y * W.z - V.z * W.y;
		normal2.y = normal1.y = normal0.y = V.z * W.x - V.x * W.z;
		normal2.z = normal1.z = normal0.z = V.x * W.y - V.y * W.x;

		normal2.x = normal1.x = normal0.x = normal0.x/ (abs(normal0.x)+abs(normal0.y)+abs(normal0.z));
		 normal2.y = normal1.y = normal0.y = normal0.y/ (abs(normal0.x)+abs(normal0.y)+abs(normal0.z));
		 normal2.z = normal1.z = normal0.z = normal0.z/ (abs(normal0.x)+abs(normal0.y)+abs(normal0.z));

		mesh.vertexNormals.push_back(normal0);
		mesh.vertexNormals.push_back(normal1);
		mesh.vertexNormals.push_back(normal2);

		auto last = static_cast<int>(mesh.vertices.size() - 1);

		mesh.triangles.push_back( { last - 2, last - 1, last });

		ntri++;
	}

	return (ntri);
}

/*-------------------------------------------------------------------------
 Return the point between two points in the same ratio as
 isolevel is between valp1 and valp2
 */
XYZ VertexInterp(double isolevel, XYZ p1, XYZ p2, double valp1, double valp2) {
	double mu;
	XYZ p;

	if (ABS(isolevel-valp1) < 0.00001)
		return (p1);
	if (ABS(isolevel-valp2) < 0.00001)
		return (p2);
	if (ABS(valp1-valp2) < 0.00001)
		return (p1);
	mu = (isolevel - valp1) / (valp2 - valp1);
	p.x = p1.x + mu * (p2.x - p1.x);
	p.y = p1.y + mu * (p2.y - p1.y);
	p.z = p1.z + mu * (p2.z - p1.z);
	//cout<<p.x<<", "<<p.y<<", "<<p.z<<endl;

	return (p);
}

