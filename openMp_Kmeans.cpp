#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <math.h>
#include "omp.h"
#include<ctime>

using namespace cv;
using namespace std;

#define MAXIMUM_STEPS 50
#define NUM_CLUSTERS 20
#define GRAYSCALE true	
#define IMG "Hazem.jpeg"
#define N_THREADS 4
/*
 compute euclidean distance between 2 pixels
 dim = 1 if grey scale
 dim = 3 if RGB
*/
float compute_distance(float* p1, float* p2, int dim = 1) {
	float result = 0.0;
	for (int i = 0; i < dim; i++) {
		result += (float)pow((*(p1 + i)) - (*(p2 + i)), 2);
	}
	return sqrt(result);
}


/*
 struct for each cluster that will contain the centroid
 the number of points in each cluster
 the sum of points for each cluster
*/
struct Cluster
{
	float* centroid = new float[NUM_CLUSTERS];
	int points_in_cluster = 0;
	long double* sum_points = new long double[GRAYSCALE ? 1 : 3];
};


bool first_iter = true;

Cluster* init_clusters(bool grayscale = true, int num_cluster = 3, Cluster* reduced_cluster = NULL) {
	Cluster* clusters = new Cluster[num_cluster];

	for (int i = 0; i < num_cluster; i++) {
		if (grayscale) {
			clusters[i].centroid = new float[1];
			if (reduced_cluster == NULL) {
				clusters[i].centroid[0] = rand() % 256;
			}
			else {
				clusters[i].centroid[0] = reduced_cluster[i].centroid[0];

			}

			clusters[i].sum_points[0] = 0;
		}
		else {
			clusters[i].centroid = new float[3];
			if (reduced_cluster != NULL) {
				clusters[i].centroid[0] = reduced_cluster[i].centroid[0];
				clusters[i].centroid[1] = reduced_cluster[i].centroid[1];
				clusters[i].centroid[2] = reduced_cluster[i].centroid[2];
			}
			else {
				clusters[i].centroid[0] = rand() % 256;
				clusters[i].centroid[1] = rand() % 256;
				clusters[i].centroid[2] = rand() % 256;
			}

			clusters[i].sum_points[0] = 0;
			clusters[i].sum_points[1] = 0;
			clusters[i].sum_points[2] = 0;
		}
	}

	return clusters;
}
int main()
{


	omp_set_num_threads(N_THREADS);
	float** old_centroid = new float* [NUM_CLUSTERS];
	for (int i = 0; i < NUM_CLUSTERS; i++)
	{
		old_centroid[i] = new float[GRAYSCALE ? 1 : 3];
	}
	// read the image
	Mat greyMat, Input_Image = imread(IMG);
	// change image to grayscale and get grayscale matrix
	greyMat = Input_Image;
	if (GRAYSCALE)
		cvtColor(Input_Image, greyMat, COLOR_BGR2GRAY);
	Input_Image = greyMat;


	int* channel_1;
	int* channel_2;
	int* channel_3;
	channel_1 = new int[Input_Image.rows * Input_Image.cols];
	if (!GRAYSCALE) {
		channel_2 = new int[Input_Image.rows * Input_Image.cols];
		channel_3 = new int[Input_Image.rows * Input_Image.cols];
	}
	for (int i = 0; i < Input_Image.rows; i++) {
		for (int j = 0; j < Input_Image.cols; j++) {
			if (GRAYSCALE) {
				channel_1[(i * Input_Image.cols) + j] = Input_Image.at<uchar>(i, j);
			}
			else {
				channel_1[(i * Input_Image.cols) + j] = Input_Image.at<Vec3b>(i, j)[0];
				channel_2[(i * Input_Image.cols) + j] = Input_Image.at<Vec3b>(i, j)[1];
				channel_3[(i * Input_Image.cols) + j] = Input_Image.at<Vec3b>(i, j)[2];
			}
		}
	}

	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
	Cluster* clusters = init_clusters(GRAYSCALE, NUM_CLUSTERS);

	int time_before_loop_begins = time(NULL);
	float start = omp_get_wtime();
	int step = 0;
	while (step < MAXIMUM_STEPS) {
#pragma omp parallel shared(clusters,old_centroid, first_iter, channel_1,channel_2,channel_3, Input_Image)
		{
			/*int i = omp_get_thread_num();
			cout << "thread num : " << i<<endl;*/
			Cluster* private_cluster = init_clusters(GRAYSCALE, NUM_CLUSTERS, clusters);

#pragma omp for 
			for (int index = 0; index < Input_Image.rows * Input_Image.cols; index++) {

				float minimum_distance = INFINITY;
				int minimum_index = -1;
				float* minimum_point = NULL;
				// this won't be done in parallel
				for (int c = 0; c < NUM_CLUSTERS; c++) {
					float* pixel;
					if (GRAYSCALE) {
						pixel = new float[1];
						pixel[0] = channel_1[index];
					}
					else
					{
						pixel = new float[3];
						pixel[0] = channel_1[index];
						pixel[1] = channel_2[index];
						pixel[2] = channel_3[index];
					}
					float dist = compute_distance(clusters[c].centroid, pixel, GRAYSCALE ? 1 : 3);
					if (dist < minimum_distance) {
						minimum_point = pixel;
						minimum_distance = dist;
						minimum_index = c;
					}
				}
				private_cluster[minimum_index].points_in_cluster++;
				//printf("centroid : %f \n", clusters[minimum_index].sum_points[0]);
				//printf("point : %f \n", minimum_point[0]);
				if (GRAYSCALE)
					private_cluster[minimum_index].sum_points[0] += (long double)minimum_point[0];
				else {
					private_cluster[minimum_index].sum_points[0] += (long double)minimum_point[0];
					private_cluster[minimum_index].sum_points[1] += (long double)minimum_point[1];
					private_cluster[minimum_index].sum_points[2] += (long double)minimum_point[2];
				}
			}



			// compute new centroid for each cluster
			// reset the sum of points in each cluster


#pragma omp critical
			{
				for (int i = 0; i < NUM_CLUSTERS; i++) {

					clusters[i].points_in_cluster += private_cluster[i].points_in_cluster;
					if (GRAYSCALE) {
						clusters[i].sum_points[0] += private_cluster[i].sum_points[0];
					}
					else {
						clusters[i].sum_points[0] += private_cluster[i].sum_points[0];
						clusters[i].sum_points[1] += private_cluster[i].sum_points[1];
						clusters[i].sum_points[2] += private_cluster[i].sum_points[2];
					}
				}
			}

#pragma omp barrier
#pragma omp single 
			{

				for (int i = 0; i < NUM_CLUSTERS; i++) {
					if (clusters[i].points_in_cluster == 0)
						continue;
					//Cluster temp = clusters[i];
					if (GRAYSCALE) {
						clusters[i].centroid[0] = clusters[i].sum_points[0] / clusters[i].points_in_cluster;
					}
					else {
						clusters[i].centroid[0] = clusters[i].sum_points[0] / clusters[i].points_in_cluster;
						clusters[i].centroid[1] = clusters[i].sum_points[1] / clusters[i].points_in_cluster;
						clusters[i].centroid[2] = clusters[i].sum_points[2] / clusters[i].points_in_cluster;
					}

					clusters[i].points_in_cluster = 0;
					clusters[i].sum_points[0] = 0.0;
					clusters[i].sum_points[1] = 0.0;
					clusters[i].sum_points[2] = 0.0;

					/*clusters[i].centroid[0] = temp.centroid[0];
					clusters[i].centroid[1] = temp.centroid[1];
					clusters[i].centroid[2] = temp.centroid[2];*/
				}
			}
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
		//here will be a barrier 
		step++;
	}
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		if (GRAYSCALE) {
			printf("clusters %d : %f \n", i, clusters[i].centroid[0]);
		}
		else {
			printf("clusters %d : %f %f %f \n", i, clusters[i].centroid[0], clusters[i].centroid[1], clusters[i].centroid[2]);

		}
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
	int time_after_loop_ends = time(NULL);
	float end = omp_get_wtime();
	//printf("time taken : %d seconds taken (geenral)\n", time_after_loop_ends - time_before_loop_begins);
	printf("time taken : %f seconds taken (omp)\n", end - start);
	Mat image = imread(IMG);
	imshow("Display Window", image);
	waitKey(0);

	imshow("Display Window", Input_Image);
	waitKey(0);
	printf("print steps : %d\n", step);
	imwrite("hazem_segmented.jpg", Input_Image);

	return 0;
}