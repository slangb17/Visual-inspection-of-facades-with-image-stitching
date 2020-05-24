#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/calib3d/calib3d_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

/* Procedure of the code:
*
* 1) First, the real world coordianes for the checkerboard is defined. As the checkerboard size is known these can be found.
*
* 2) Obtain images of the checkerboard through various angles and views.
*
* 3) The function findChessboardCorners is used to find the image coordianes of the checkerboard corners on all images.
*
* 4) The function calibrateCamera is used to calibrate the camera.
*
* 5) Undistort images with the undistort function
*
*/

// Variables used in the code:
cv::Mat img, grayImg;
std::vector<std::vector<cv::Point3f>> objectPoints; // Creating vector to store vectors of 3D points
std::vector<std::vector<cv::Point2f>> imagePoints; // Creating vector to store vectors of 2D points

int main() {

	// Defining the real world coordinates of the checkerboard corners:
	std::vector<cv::Point3f> threeDpoints; // Defining the world coordinates for 3D points
	double checkerboardsize = 36.8;
	for (int i = 0; i < 9; i++) {
		for (int j = 0; j < 6; j++) {
			threeDpoints.push_back(cv::Point3f(j * checkerboardsize, i * checkerboardsize, 0));
		}
	}

	// Loading images:
	std::string path = "..\\images\\calibration_images\\*.jpg"; 
	std::vector<cv::String> src;
	cv::glob(path, src);

	// Looping through all the files in the directory
	for (int i = 0; i < src.size(); i++) {
				
		std::cout << "Image " << i << ": " << src[i] << std::endl;

		img = cv::imread(src[i]);
		cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

		// Finding checker board corners
		std::vector<cv::Point2f> cornerPoints;
		bool success = findChessboardCorners(img, cv::Size(9, 6), cornerPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		
		if (success) {
			cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 0.0001);

			// Refining pixel coordinates for 2D points
			cv::cornerSubPix(grayImg, cornerPoints, cv::Size(11, 11), cv::Size(-1, -1), criteria); 

			objectPoints.push_back(threeDpoints);
			imagePoints.push_back(cornerPoints);
		}
	}

	std::cout << std::endl << "Computing camera matrix and distortion coefficients..." << std::endl;

	cv::Mat cameraMatrix, distortionCoefficients, rotation, translation; // Defining variables for calibrateCamera()

	calibrateCamera(objectPoints, imagePoints, cv::Size(img.rows, img.cols), cameraMatrix, distortionCoefficients, rotation, translation);

	std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
	std::cout << "distortionCoefficients : " << distortionCoefficients << std::endl;

	std::cout << std::endl << "Undistorting images..." << std::endl;

	// Loading images:
	std::string path2 = "..\\images\\distorted_images\\*.jpg";
	std::vector<cv::String> src2;
	cv::glob(path2, src2);

	cv::Mat img, undistortImg; // Defining variable for undistort()

	for (int i = 0; i < src2.size(); i++) {

		std::cout << "Image " << i << ": " << src2[i] << std::endl;

		img = cv::imread(src2[i]);

		cv::undistort(img, undistortImg, cameraMatrix, distortionCoefficients);

		cv::imwrite("..\\images\\undistorted_images\\image_" + std::to_string(i) + ".jpg", undistortImg);

		cv::namedWindow("Undistorted", cv::WindowFlags::WINDOW_NORMAL);
		cv::resizeWindow("Undistorted", 750, 750);
		imshow("Undistorted", undistortImg);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	return 0;
}