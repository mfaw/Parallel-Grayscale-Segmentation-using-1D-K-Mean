#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <math.h>


using namespace cv;
using namespace std;

/*
   manually set the:
   1- maximum steps of iteration
   2- number of clusteers (k)
   3- if we want output to be grayscale (true), output to be RGB (false)
*/
#define MAXIMUM_STEPS 2000
#define NUM_CLUSTERS 11
#define GRAYSCALE false

/*
 compute euclidean distance between 2 pixels
 dim = 1 if grey scale
 dim = 3 if RGB
*/
float compute_distance(float* p1, float* p2, int dim = 1) {
	float result = 0.0;
	for (int i = 0;  i < dim; i++) {
		result += (float)pow((*(p1 + i)) - (*(p2 + i)), 2);
	}
	return sqrt(result);
}

/*
 struct for each cluster that will contain the centroids
 the number of points in each cluster
 the sum of points for each cluster
*/
struct Cluster
{
	float* centroid = new float [NUM_CLUSTERS];
	int points_in_cluster = 0;
	long double* sum_points = new long double [GRAYSCALE ? 1:3];
}clusters[NUM_CLUSTERS];

float** old_centroid = new float*[NUM_CLUSTERS];
bool first_iter = true;

int main()
{
	// read the image
	Mat greyMat, Input_Image = imread("RGB.jpg");
	// change image to grayscale and get grayscale matrix
	greyMat = Input_Image;
	if(GRAYSCALE)
		cvtColor(Input_Image, greyMat, COLOR_BGR2GRAY);
	Input_Image = greyMat;
	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;

	/*
	initialize the cluster centroids using random numbers and set sum of points to 0
	if it's grayscale segmentation, we have 1 array of centroids and sum of points array
	if it's RGB segmentation, we have 3 arrays of centroids for each cluster and 3 sum of points array ( 1 for each R, G, B)
	*/
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

	// while loop to iterate used to compute new centroids
	// break condition is centroids not changed
	int step = 0;
	while (step < MAXIMUM_STEPS) {
	// loop over the pixels matrix
		for (int row = 0; row < Input_Image.rows; row++) {
			for (int col = 0; col < Input_Image.cols; col++) {
				float minimum_distance = INFINITY;
				int minimum_index = -1;
				float* minimum_point = NULL;
				/*
				    in this for loop, we loop over the clusters and store the value in the pixel array
				    then, compute the distance between the centroid of the cluster and the current pixel
				    if distance is less than minimum distance:
				    we assign minimum point to pixel, minimum distance to the computed distance, and
				    minimum index to current cluster
				    by this, the point is assigned to the closest cluster
				*/
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
				// increment the points in cluster
				clusters[minimum_index].points_in_cluster++;
				// add point to the cluster's sum_points
				if(GRAYSCALE)
					clusters[minimum_index].sum_points[0] += (long double) minimum_point[0];
				else {
					clusters[minimum_index].sum_points[0] += (long double) minimum_point[0];
					clusters[minimum_index].sum_points[1] += (long double) minimum_point[1];
					clusters[minimum_index].sum_points[2] += (long double) minimum_point[2];
				}

			}
		}
		// compute new centroid for each cluster by dividing sum of points and the points in the cluster
		// reset the sum of points in each cluster and reset the points in cluster to 0 for each cluster
		for (int i = 0; i < NUM_CLUSTERS; i++) {
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
			clusters[i].points_in_cluster = 0;
		}

		// assign the centroids of each cluster to the old centroid if it's the first iteration
		// and change the first_iter flag to false
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
		// break condition check if it's not the first iteration
		// compute the distance between the old centroid and the current centorid, if it's greater than 0.0001
		// change the condition flag to false and assign the old centroid to the current centroid
		// if distance is less than 0.0001 the condition flag is true, so we break of the while loop showing that we
		// reached the final clusters
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
		step++;
	}
	/*This part of the program prints the final cluster centroids after the clustering algorithm has converged.
	Specifically, the code iterates over all clusters and prints their centroid coordinates.*/
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		printf("clusters %d : %f \n", i, clusters[i].centroid[0]);
	}
	/* This for loop, iterates over the pixels in the input image and finds the nearest cluster centroid for each pixel
	by computing the distance between the pixel and each cluster centroid, then it assigns the value of centroid of the
	nearest cluster to the pixel of the input image*/
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
	imshow("Display Window", Input_Image);
	waitKey(0);
	printf("print steps : %d", step);

	return 0;
}