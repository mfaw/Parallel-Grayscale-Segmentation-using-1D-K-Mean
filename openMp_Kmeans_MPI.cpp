#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <math.h>
#include<ctime>
#include "mpi.h"

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

#define MAXIMUM_STEPS 50
#define NUM_CLUSTERS 8
#define GRAYSCALE false
#define IMG "test.jpg"


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

int main()
{
	bool first_iter;

	/*old_centroid is a pointer to the first element of an array of pointers, 
	where each pointer points to a row of float values*/
	float** old_centroid = new float* [NUM_CLUSTERS]; // private for rank 0

	//Mat is a class that represents a matrix of image pixels.
	Mat greyMat, Input_Image; // private for rank 0

	//These channels are used to store pixel values of the input image.
	int* channel_1 = NULL; // scatter
	int* channel_2 = NULL; // scatter
	int* channel_3 = NULL; // scatter

    //breakCondition is a flag used to indicate whether a loop should be broken.
	bool breakCondition; // broadcast

	//MPI_Init initializes the MPI environment with the default options.
	MPI_Init(NULL, NULL);
	
	int size;
	/*MPI_Comm_size, gets the number of processes in the MPI Communicator
	"MPI_COMM_WORLD" and store it in the variable "Size".*/
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
    /*MPI_Comm_rank, gets the rank of the current process in the MPI Communicator
    "MPI_COMM_WORLD" and store it in the variable "rank". */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	 
	//counter variable is used to measure the elapsed time
	double counter = 0.0;
	counter -= MPI_Wtime();

	double start = MPI_Wtime();
	float** local_result;
	int step;

     /*clusters is a pointer to a pointer to float, this is used to store 
	 the cluster data.*/
	float** clusters; // broadcast
	clusters = new float* [NUM_CLUSTERS];

	//allocate memory for clusters.
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		clusters[i] = new float[7];
	}
	int rows, cols; // broadcast

    //allocate memory for local_result.
	local_result = new float* [NUM_CLUSTERS];
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		local_result[i] = new float[4];
	}
    //This code is executed only by the process with rank 0.
	if (rank == 0) {
		printf("size : %d\n", size);

		//Indicating that this is the first iteration of the clustering algorithm.
		step = 0;
		
		//Indicating that this is the first iteration of the clustering algorithm.
		first_iter = true;

		//Indicating that the clustering Algorithm has not yet converged.
		breakCondition = false;
		
        //This for loop is used to initialize the clusters array with random values
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			//Initializes the number of data points in the cluster to 0.
			clusters[i][6] = 0.0;
			if (GRAYSCALE) {
			/*Set the centroid to a random value between 0 and 255, for channel 0*/
			clusters[i][0] = (rand() % 256) / 1.0;
			/*Set the sum of data points in channel 0 to 0.*/
			clusters[i][3] = 0.0;
			}
			else {
				/*Set the centroid to a random value between 0 and 255, for every channel.*/
				clusters[i][0] = (rand() % 256) / 1.0;
				clusters[i][1] = (rand() % 256) / 1.0;
				clusters[i][2] = (rand() % 256) / 1.0;
				/*Set the sum of data points in each channel to 0.*/
				clusters[i][3] = 0.0;
				clusters[i][4] = 0.0;
				clusters[i][5] = 0.0;
			}
		}

         /*Initializes the old_centroid to be (2D-array) with number of 
		 rows = Number of clusters , and number of columns = 1 or 3, based on whether the image is grayscale or color.*/
		for (int i = 0; i < NUM_CLUSTERS; i++)
		{
			/*Each row corresponds to a cluster, and the columns store the values 
			of the centroid of the cluster from the previous iteration*/
			old_centroid[i] = new float[GRAYSCALE ? 1 : 3];
		}
        //Loads an image from a file.
		Input_Image = imread(IMG);
		greyMat = Input_Image;

		//Convert the image into grayscale, if the GRAYSCALE flag = TRUE.
		if (GRAYSCALE)
			cvtColor(Input_Image, greyMat, COLOR_BGR2GRAY);
		Input_Image = greyMat;

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
		//It sets the values of rows and columns to the height and width of the image.
		rows = Input_Image.rows;
		cols = Input_Image.cols;
		//cout << "Height: " << Input_Image.rows << ", Width: " << Input_Image.cols << ", Channels: " << Input_Image.channels() << endl;
		//printf("after initialization 0.1 %d\n", rank);

	}

	//cout<<"after initialization"<<rank;
	int* channel_1_private;
	int* channel_2_private;
	int* channel_3_private;

    /*It receives the values of rows and columns from process 0, using MPI_Bcast.These values are broadcasted to all the 
	processes, so that they know the size of the image.*/
	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

     /*It allocates memory for the private channel arrays of each process.It calculates the number of pixels that each 
	 process will be responsible for, by dividing the total number of pixels in the image by the number of processes.*/
	if (GRAYSCALE) {
		channel_1_private = new int[(int)((rows * cols) / size)];
	}
	else {
		channel_1_private = new int[(int)((rows * cols) / size)];
		channel_2_private = new int[(int)((rows * cols) / size)];
		channel_3_private = new int[(int)((rows * cols) / size)];
	}

	//MPI_Barrier is used to ensure that all processes start at the same time.
	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Bcast is used to broadcast the value of breakCondition from process 0, to all the processes.
	MPI_Bcast(&breakCondition, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

	while (!breakCondition) {
		/*This for loop, loops through each cluster and broadcasts the cluster data(centroid, Sum of data points in 
		cluster and number of data points in each cluster) to all the processes*/
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			MPI_Bcast(&clusters[i][0],  7, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}

		/*MPI_Scatter is used to distribute the pixel values of the image to all processes.The channel_1 array (and the 
		channel_2 and channel_3 arrays, if the image is not grayscale) is divided into equal-sized chunks, with each chunk
		assigned to a different process. Each process receives the chunk of the array that corresponds to the pixels it is
		responsible for.*/
		MPI_Scatter(&channel_1[0], (int)((rows * cols) / size), MPI_INT, &channel_1_private[0], 
        (int)((rows * cols) / size), MPI_INT, 0, MPI_COMM_WORLD);
		if (!GRAYSCALE) {
			MPI_Scatter(&channel_2[0], (int)((rows * cols) / size), MPI_INT, &channel_2_private[0], 
            (int)((rows * cols) / size), MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Scatter(&channel_3[0], (int)((rows * cols) / size), MPI_INT, &channel_3_private[0], 
            (int)((rows * cols) / size), MPI_INT, 0, MPI_COMM_WORLD);
		}

       //This for loop, loops through each pixel in the local portion of the image, and assigns it to the nearest cluster.
		for (int index = 0; index < (int)((rows * cols) / size); index++) {
			float minimum_distance = INFINITY;
			int minimum_index = -1;
			float* minimum_point = NULL;
			// this won't be done in parallel
			for (int c = 0; c < NUM_CLUSTERS; c++) {
				float* pixel;
				float* centroid;
				if (GRAYSCALE) {
					pixel = new float[1];
					pixel[0] = channel_1_private[index];

					centroid = new float[1];
					centroid[0] = clusters[c][0];

				}
				else
				{
					pixel = new float[3];
					pixel[0] = channel_1_private[index];
					pixel[1] = channel_2_private[index];
					pixel[2] = channel_3_private[index];

					centroid = new float[3];
					centroid[0] = clusters[c][0];
					centroid[1] = clusters[c][1];
					centroid[2] = clusters[c][2];
				}
				/*For each pixel, it computes the distance to each centroid using the compute_distance() function
				and assigns the pixel to the cluster with the closest centroid*/
				float dist = compute_distance(centroid, pixel, GRAYSCALE ? 1 : 3);
				if (dist < minimum_distance) {
				/*It keeps track of the minimum distance and the index of the cluster with the closet centroid*/
					minimum_point = pixel;
					minimum_distance = dist;
					minimum_index = c;
				}
			}

			/*It updates the data of the assigned cluster, for each pixel, it adds the pixel value to the sum of data 
			points in the cluster, and increments the number of data points in the cluster. */
			clusters[minimum_index][6]+=1;

			if (GRAYSCALE)
				clusters[minimum_index][3] += minimum_point[0];
			else {
				clusters[minimum_index][3] += minimum_point[0];
				clusters[minimum_index][4] += minimum_point[1];
				clusters[minimum_index][5] += minimum_point[2];
			}
		}

        /* MPI_Reduce is used to sum up a portion of the clusters array and stores the result in the local_result array*/
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			MPI_Reduce(&clusters[i][3], &local_result[i][0], 4, MPI_FLOAT,
				MPI_SUM, 0, MPI_COMM_WORLD);
		
		}

        //This code is executed only by the process with rank 0.
		if(rank == 0){
 			/*This for loop, updates the clusters array with new centroids*/
			for (int i = 0; i < NUM_CLUSTERS; i++) {
				//printf("centroid : %f \n", clusters[i].sum_points[0]);
				float *  temp = local_result[i];
				if (local_result[i][3] == 0)
					continue;
				
				if (GRAYSCALE) {
					clusters[i][0] = local_result[i][0] / local_result[i][3];
				}
				else {
					clusters[i][0] = local_result[i][0] / local_result[i][3];
					clusters[i][1] = local_result[i][1] / local_result[i][3];
					clusters[i][2] = local_result[i][2] / local_result[i][3];

				}

                /*And for each cluster, it resets the sum of data points in each channel, and the number of data points*/
				clusters[i][6] = 0.0;
				clusters[i][3] = 0.0;
				clusters[i][4] = 0.0;
				clusters[i][5] = 0.0;
			}

     		/*If we are in the first iteration, then we will assign the current centroids of the clusters, 
            to the old centroids.*/	
            if (first_iter) {
				for (int i = 0; i < NUM_CLUSTERS; i++) {
					if (GRAYSCALE) {
						old_centroid[i][0] = clusters[i][0];
					}
					else {
						old_centroid[i][0] = clusters[i][0];
						old_centroid[i][1] = clusters[i][1];
						old_centroid[i][2] = clusters[i][2];
					}
				}
				first_iter = false;
			}
			/*If we are not in the first iteration, then we will check for convergence by computing the distance 
			between each cluster's current centroid and its old centroid, and checking if the distance is greater
			than a threshold (0.01 in this case). If the centroids have converged, the breakCondition flag is set to true
			and the loop is exited. Otherwise, the old_centroid array is updated with the current centroids.*/
			else {
				bool condition = true;
				for (int i = 0; i < NUM_CLUSTERS; i++) {
					float * centroid = new float[3];
					centroid[0] = clusters[i][0];
					centroid[1] = clusters[i][1];
					centroid[2] = clusters[i][2];

					if (compute_distance(centroid, old_centroid[i], GRAYSCALE ? 1 : 3) > 0.01) {
						condition = false;
						break;
					}
				}
				if (condition) {
					breakCondition = true;
				}
				else {
					for (int i = 0; i < NUM_CLUSTERS; i++) {
						if (GRAYSCALE) {
							old_centroid[i][0] = clusters[i][0];
						}
						else {

							old_centroid[i][0] = clusters[i][0];
							old_centroid[i][1] = clusters[i][1];
							old_centroid[i][2] = clusters[i][2];

						}
					}
				}
			}
			//printf("%f , %f, %f\n", clusters[0][0], clusters[0][1], clusters[0][2]);
			step++;
			if (step == MAXIMUM_STEPS) {
				breakCondition = true;

			}
		}
	    /*MPI_Bcast sends the value of the breakCondition variable from the root process (rank 0) to all other processes.*/
		MPI_Bcast(&breakCondition, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

		/*MPI_Barrier is used to synchronize all processes before continuing with the next iteration of the loop.*/
		MPI_Barrier(MPI_COMM_WORLD);
		
	}//End of While loop
	
	 //This code is executed only by the process with rank 0.
	if (rank == 0) {
		/*This part of the program prints the final cluster centroids after the clustering algorithm has converged.
		 Specifically, the code iterates over all clusters and prints their centroid coordinates.
		 It's worth noting that this code should only be executed in the root process (rank 0), since only the root 
		 process has access to the final clusters array. */

		for (int i = 0; i < NUM_CLUSTERS; i++) {
			if (GRAYSCALE) {
				printf("clusters %d : %f \n", i, clusters[i][0]);
			}
			else {
				printf("clusters %d : %f %f %f \n", i, clusters[i][0], clusters[i][1], clusters[i][2]);

			}
		}

		/* This for loop, iterates over the pixels in the input image and finds the nearest cluster centroid for each pixel by 
		computing the distance between the pixel and each cluster centroid, then it assigns the value of centroid of the
		nearest cluster to the pixel of the input image*/
		for (int row = 0; row < Input_Image.rows; row++) {
			for (int col = 0; col < Input_Image.cols; col++) {
				float minimum_distance = INFINITY;
				int minimum_index = -1;
				float* minimum_point = NULL;
				for (int c = 0; c < NUM_CLUSTERS; c++) {
					float* pixel;
					float* centroid;
					if (GRAYSCALE) {
						pixel = new float[1];
						pixel[0] = Input_Image.at<uchar>(row, col);

						centroid = new float[1];
						centroid[0] = clusters[c][0];
					}
					else
					{
						centroid = new float[3];
						centroid[0] = clusters[c][0];
						centroid[1] = clusters[c][1];
						centroid[2] = clusters[c][2];

						pixel = new float[3];
						pixel[0] = Input_Image.at<Vec3b>(row, col)[0];
						pixel[1] = Input_Image.at<Vec3b>(row, col)[1];
						pixel[2] = Input_Image.at<Vec3b>(row, col)[2];
					}
					float dist = compute_distance(centroid, pixel, GRAYSCALE ? 1 : 3);
					if (dist < minimum_distance) {
						minimum_point = pixel;
						minimum_distance = dist;
						minimum_index = c;
					}
				}
				if (GRAYSCALE) {
					Input_Image.at<uchar>(row, col) = clusters[minimum_index][0];

				}
				else {
					Input_Image.at<Vec3b>(row, col)[0] = clusters[minimum_index][0];
					Input_Image.at<Vec3b>(row, col)[1] = clusters[minimum_index][1];
					Input_Image.at<Vec3b>(row, col)[2] = clusters[minimum_index][2];
				}
			}
		}
		/* This measures the time taken by the clustering algorithm to complete and displays the number of steps taken 
		and the elapsed time. Specifically, the code records the time before and after the main loop that performs the 
		clustering computations, computes the elapsed time, and displays it along with the number of steps taken.*/
		int time_after_loop_ends = time(NULL);
		printf("steps taken : %d\n", step);
		counter += MPI_Wtime();
		printf("time taken : %f\n", counter);
		/*This displays the input image and the output image after clustering.*/
		Mat image = imread(IMG);
		imshow("Display Window", image);
		waitKey(0);
		//printf("time taken : %d seconds taken (geenral)\n", time_after_loop_ends - time_before_loop_begins
		imshow("Display Window", Input_Image);
		waitKey(0);
	}
    /*The code waits for all processes to synchronize using MPI_Barrier and then finalizes MPI using MPI_Finalize. 
	These steps ensure that all processes have completed their computations and released any MPI resources before 
	the program exits.*/
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}