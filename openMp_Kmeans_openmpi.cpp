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

/*
   manually set the:
   1- maximum steps of iteration
   2- number of clusteers (k)
   3- if we want output to be grayscale (true), output to be RGB (false)
   4- the image to segment
   5- the number of threads used
*/
#define MAXIMUM_STEPS 100
#define NUM_CLUSTERS 8
#define GRAYSCALE true
#define IMG "test.jpg"
#define N_THREADS 8
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
 /* This function initializes the cluster centroids, Specifically, the function creates an array of num_cluster 'clusters' 
 and sets the initial centroid for each cluster to a random value if reduced_cluster is NULL, or to the centroid of a 
 reduced set of clusters if reduced_cluster is not NULL.*/
Cluster* init_clusters(bool grayscale = true, int num_cluster = 3, Cluster* reduced_cluster = NULL) {
/* -The grayscale parameter is a flag that indicates whether the input image is grayscale (true) or color (false).
   -The num_cluster parameter specifies the number of clusters to create.
   -The reduced_cluster parameter is an optional pointer to an array of reduced clusters that can be used to initialize 
	the centroids of the new clusters.*/
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
   // and Finally, we return the array of clusters
	return clusters;
}
int main()
{
	omp_set_num_threads(N_THREADS);

	/*old_centroid is a pointer to the first element of an array of pointers, 
	where each pointer points to a row of float values*/
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

	//These channels are used to store pixel values of the input image.
	int* channel_1;
	int* channel_2;
	int* channel_3;
	 /*Channel_1 creates a (1D-array) of int values with a size equal to the number of pixels in the input image. 
	 This array is used to store the grey channel pixel values, if the input image is grayscale*/
	channel_1 = new int[Input_Image.rows * Input_Image.cols];

	/*The if (!GRAYSCALE) statement checks if the input image is not grayscale.If it is not grayscale, channel_2 and
	channel_3 are allocated with new int[Input_Image.rows * Input_Image.cols], which creates two more 1D arrays 
	with the same size as channel_1.These arrays are used to store the pixel values of the green and blue channels 
	of the input image, respectively*/
	if (!GRAYSCALE) {
		channel_2 = new int[Input_Image.rows * Input_Image.cols];
		channel_3 = new int[Input_Image.rows * Input_Image.cols];
	}

	/*This for loop, loops through each pixel in the image and stores its value in the appropriate channel array*/
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

	cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels()<< endl;
	Cluster* clusters = init_clusters(GRAYSCALE, NUM_CLUSTERS);

	int time_before_loop_begins = time(NULL);
	float start = omp_get_wtime();
	int step = 0;

	/*The loop iterates until either the maximum number of steps (MAXIMUM_STEPS) is reached or the centroids of the 
	clusters stop changing significantly.*/
	while (step < MAXIMUM_STEPS) {

/* The loop is parallelized using the #pragma omp parallel directive, which creates a team of threads to execute 
the enclosed code block in parallel. The shared clause is used to specify variables that are shared among all threads,
including the clusters,old_centroid, first_iter, channel_1, channel_2, channel_3, and Input_Image variables.*/
#pragma omp parallel shared(clusters,old_centroid, first_iter, channel_1,channel_2,channel_3, Input_Image)
		{
			/*Within the parallel block, each thread initializes a private copy of the clusters using the init_clusters 
			function.*/
			Cluster* private_cluster = init_clusters(GRAYSCALE, NUM_CLUSTERS, clusters);

/* The #pragma omp for directive is used to distribute the iterations of the loop among the threads in a parallel manner
Each thread computes the closest cluster centroid for each pixel in the input image and updates its private copy of 
the clusters accordingly.*/
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

/*The #pragma omp critical directive is used to ensure that the updates to the shared clusters array are performed 
atomically.*/
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

/*The loop uses a barrier to ensure that all threads have finished computing the new centroids for their private clusters*/
#pragma omp barrier

/*After all threads have finished computing the new centroids for their private clusters, the #pragma omp single 
directive is used to ensure that only one thread updates the shared clusters array with the new centroids.*/
#pragma omp single 
			{
				for (int i = 0; i < NUM_CLUSTERS; i++) {
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
					
				}
			}
		}

        /*If we are in the first iteration, then we will assign the current centroids of the clusters, 
        to the old centroids.*/	
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

		/*If we are not in the first iteration, then we will check for convergence by computing the distance 
		between each cluster's current centroid and its old centroid, and checking if the distance is greater
		than a threshold (0.0001 in this case). If the centroids have converged, the breakCondition flag is set to true
		and the loop is exited. Otherwise, the old_centroid array is updated with the current centroids.*/
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
		/*Finally, the loop increments the step counter and repeats the process until the maximum number of steps is 
		reached or the centroids converge*/
		step++;
	}

	/*This part of the program prints the final cluster centroids after the clustering algorithm has converged.
	Specifically, the code iterates over all clusters and prints their centroid coordinates.*/
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		if (GRAYSCALE) {
			printf("clusters %d : %f \n", i, clusters[i].centroid[0]);
		}
		else {
			printf("clusters %d : %f %f %f \n", i, clusters[i].centroid[0], clusters[i].centroid[1], clusters[i].centroid[2]);

		}
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

    /* This measures the time taken by the clustering algorithm to complete and displays the elapsed time. Specifically, 
	the code records the time before and after the main loop that performs the clustering computations, computes the 
	elapsed time, and displays it*/
	int time_after_loop_ends = time(NULL);
	float end = omp_get_wtime();
	//printf("time taken : %d seconds taken (geenral)\n", time_after_loop_ends - time_before_loop_begins);
	printf("time taken : %f seconds taken (omp)\n", end - start);

	/*This displays the input image and the output image after clustering.*/
	Mat image = imread(IMG);
	imshow("Display Window", image);
	waitKey(0);
	imshow("Display Window", Input_Image);
	waitKey(0);
	printf("print steps : %d\n", step);

	return 0;
}