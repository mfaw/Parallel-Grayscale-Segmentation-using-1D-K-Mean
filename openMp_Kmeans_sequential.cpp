#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <math.h>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;
#define MAXIMUM_STEPS 50
#define NUM_CLUSTERS 8
#define GRAYSCALE false
#define IMG "test.jpg"


// compute euclidean distance between 2 pixels
// dim = 1 if grey scale
// dim = 3 if RGB
float compute_distance(float* p1, float* p2, int dim = 1) {
	float result = 0.0;
	for (int i = 0; i < dim; i++) {
		result += (float)pow((*(p1 + i)) - (*(p2 + i)), 2);
	}
	return sqrt(result);
}

// struct for each cluster that will contain the centroid
// the number of points in each cluster
// the sum of points for each cluster

struct Cluster
{
	float* centroid = new float[NUM_CLUSTERS];
	int points_in_cluster = 0;
	long double* sum_points = new long double[GRAYSCALE ? 1 : 3];
}clusters[NUM_CLUSTERS];

float** old_centroid = new float* [NUM_CLUSTERS];
bool first_iter = true;

int main()
{
	auto start = high_resolution_clock::now();
	// read the image
	Mat greyMat, Input_Image = imread(IMG);
	// change image to grayscale and get grayscale matrix
	greyMat = Input_Image;
	if (GRAYSCALE)
		cvtColor(Input_Image, greyMat, COLOR_BGR2GRAY);
	Input_Image = greyMat;
	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		if (GRAYSCALE) {
			clusters[i].centroid = new float[1];
			clusters[i].centroid[0] = rand() % 256;
			clusters[i].sum_points[0] = 0;
			old_centroid[i] = new float[1];
		}
		else {
			old_centroid[i] = new float[3];
			clusters[i].centroid = new float[3];
			clusters[i].centroid[0] = rand() % 256;
			clusters[i].centroid[1] = rand() % 256;
			clusters[i].centroid[2] = rand() % 256;
			clusters[i].sum_points[0] = 0;
			clusters[i].sum_points[1] = 0;
			clusters[i].sum_points[2] = 0;
		}
	}
	//for (int i = 0; i < NUM_CLUSTERS; i++) 
		//printf("Cluster %d  : %f \n", i, clusters[i].centroid[0]);
	int step = 0;
	while (step < MAXIMUM_STEPS) {
		for (int row = 0; row < Input_Image.rows; row++) {
			for (int col = 0; col < Input_Image.cols; col++) {
				float minimum_distance = INFINITY;
				int minimum_index = -1;
				float* minimum_point = NULL;
				// this won't be done in parallel
				for (int c = 0; c < NUM_CLUSTERS; c++) {
					float* pixel;
					if (GRAYSCALE) {
						pixel = new float[1];
						pixel[0] = Input_Image.at<uchar>(row, col);
					}
					else
					{
						pixel = new float[3];
						pixel[0] = Input_Image.at<Vec3b>(row, col)[0];
						pixel[1] = Input_Image.at<Vec3b>(row, col)[1];
						pixel[2] = Input_Image.at<Vec3b>(row, col)[2];
					}
					float dist = compute_distance(clusters[c].centroid, pixel, GRAYSCALE ? 1 : 3);
					if (dist < minimum_distance) {
						minimum_point = pixel;
						minimum_distance = dist;
						minimum_index = c;
					}
				}
				clusters[minimum_index].points_in_cluster++;
				//printf("centroid : %f \n", clusters[minimum_index].sum_points[0]);
				//printf("point : %f \n", minimum_point[0]);
				if (GRAYSCALE)
					clusters[minimum_index].sum_points[0] += (long double)minimum_point[0];
				else {
					clusters[minimum_index].sum_points[0] += (long double)minimum_point[0];
					clusters[minimum_index].sum_points[1] += (long double)minimum_point[1];
					clusters[minimum_index].sum_points[2] += (long double)minimum_point[2];
				}
				//printf("centroid : %f \n", clusters[minimum_index].sum_points[0]);
				//system("pause");
			}
		}
		// compute new centroid for each cluster
		// reset the sum of points in each cluster
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			//printf("centroid : %f \n", clusters[i].sum_points[0]);
			if (clusters[i].points_in_cluster == 0)
				continue;
			if (GRAYSCALE) {
				clusters[i].centroid[0] = (long double)clusters[i].sum_points[0] / clusters[i].points_in_cluster;
				clusters[i].sum_points[0] = 0.0;
			}
			else {
				clusters[i].centroid[0] = (long double)clusters[i].sum_points[0] / clusters[i].points_in_cluster;
				clusters[i].centroid[1] = (long double)clusters[i].sum_points[1] / clusters[i].points_in_cluster;
				clusters[i].centroid[2] = (long double)clusters[i].sum_points[2] / clusters[i].points_in_cluster;
				clusters[i].sum_points[0] = 0.0;
				clusters[i].sum_points[1] = 0.0;
				clusters[i].sum_points[2] = 0.0;
			}
			//printf("centroid : %f \n", clusters[i].centroid[0]);
			//printf("Cluster %d : %d \n",i,clusters[i].points_in_cluster);
			clusters[i].points_in_cluster = 0;
		}
		if (first_iter) {
			for (int i = 0; i < NUM_CLUSTERS; i++) {
				if (GRAYSCALE) {
					old_centroid[i][0] = clusters[i].centroid[0];
				}
				else {
					old_centroid[i][0] = clusters[i].centroid[0];
					old_centroid[i][1] = clusters[i].centroid[1];
					old_centroid[i][2] = clusters[i].centroid[2];

				}
			}
			first_iter = false;
		}
		else {
			bool condition = true;
			for (int i = 0; i < NUM_CLUSTERS; i++) {
				if (compute_distance(clusters[i].centroid, old_centroid[i], GRAYSCALE ? 1 : 3) > 0.0001) {
					condition = false;
					break;
				}
			}
			if (condition) {
				break;
			}
			else {
				for (int i = 0; i < NUM_CLUSTERS; i++) {
					if (GRAYSCALE) {
						old_centroid[i][0] = clusters[i].centroid[0];
					}
					else {
						old_centroid[i][0] = clusters[i].centroid[0];
						old_centroid[i][1] = clusters[i].centroid[1];
						old_centroid[i][2] = clusters[i].centroid[2];

					}
				}
			}
		}


		//printf("\n");
	 //here will be a barrier 
		step++;
	}
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		printf("clusters %d : %f \n", i, clusters[i].centroid[0]);
	}
	for (int row = 0; row < Input_Image.rows; row++) {
		for (int col = 0; col < Input_Image.cols; col++) {
			float minimum_distance = INFINITY;
			int minimum_index = -1;
			float* minimum_point = NULL;
			for (int c = 0; c < NUM_CLUSTERS; c++) {
				float* pixel;

				if (GRAYSCALE) {
					pixel = new float[1];
					pixel[0] = Input_Image.at<uchar>(row, col);
				}
				else
				{
					pixel = new float[3];
					pixel[0] = Input_Image.at<Vec3b>(row, col)[0];
					pixel[1] = Input_Image.at<Vec3b>(row, col)[1];
					pixel[2] = Input_Image.at<Vec3b>(row, col)[2];
				}
				float dist = compute_distance(clusters[c].centroid, pixel, GRAYSCALE ? 1 : 3);
				if (dist < minimum_distance) {
					minimum_point = pixel;
					minimum_distance = dist;
					minimum_index = c;
				}
			}
			if (GRAYSCALE) {
				Input_Image.at<uchar>(row, col) = clusters[minimum_index].centroid[0];

			}
			else {
				Input_Image.at<Vec3b>(row, col)[0] = clusters[minimum_index].centroid[0];
				Input_Image.at<Vec3b>(row, col)[1] = clusters[minimum_index].centroid[1];
				Input_Image.at<Vec3b>(row, col)[2] = clusters[minimum_index].centroid[2];
			}
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time taken by function: "
		<< duration.count() * 1e-6 << " seconds " << endl;
	Mat image = imread(IMG);
	imshow("Display Window", image);
	waitKey(0);

	imshow("Display Window", Input_Image);
	waitKey(0);
	printf("print steps : %d", step);

	return 0;
}