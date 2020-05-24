// Libraries used in the code:
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
#include <iostream>

// Function declarations:
std::vector<std::vector<cv::Mat>> image_loader(std::string path);
void detect_and_extract_keypoints(std::vector<std::vector<cv::Mat>> input_images, std::vector<std::vector<std::vector<cv::KeyPoint>>>& output_keypoints, std::vector<std::vector<cv::Mat>>& output_descriptors);
void neighbor_matcher(std::vector<std::vector<cv::Mat>>& input_descriptors, std::vector<std::vector<std::vector<cv::DMatch>>>& output_matches);
void compute_homography(std::vector<std::vector<std::vector<cv::KeyPoint>>> keypoints, std::vector<std::vector<std::vector<cv::DMatch>>>matches, std::vector<std::vector<cv::Mat>>& homograhpy_matrices);
void compute_homography_to_reference(std::vector<std::vector<cv::Mat>> input_homograhpy, std::vector<std::vector<cv::Mat>>& output_homograhpy);
void exposure_compensation(std::vector<std::vector<cv::Mat>>& input_images);
cv::Mat warp_align_blend(std::vector<std::vector<cv::Mat>> images, std::vector<std::vector<cv::Mat>> homograhpy_to_reference);
cv::Mat crop_img(cv::Mat result);

// Global variables:
std::vector<std::vector<cv::Mat>> images;
std::vector<std::vector<cv::Mat>> masks;
std::vector<std::vector<std::vector<cv::KeyPoint>>> keypoints;
std::vector<std::vector<std::vector<cv::DMatch>>> matches;

int main() {

	// Specifing path to folder containing images to be stitched into a single image:
	std::string load_folder_path = "..\\images\\facade_1\\vertical\\";
	std::string save_folder_path =  load_folder_path + "result.jpg";

	// Loading images into vector row-wise:
	images = image_loader(load_folder_path);

	// Detect and extract features from images:
	std::vector<std::vector<cv::Mat>> descriptors;
	detect_and_extract_keypoints(images, keypoints, descriptors);

	// Matching features between row-wise neighbors:
	neighbor_matcher(descriptors, matches);

	// Computing homography between matched neighbors:
	std::vector<std::vector<cv::Mat>> homograhpy;
	compute_homography(keypoints, matches, homograhpy);

	// Multiply homography matrices:
	std::vector<std::vector<cv::Mat>> homograhpy_to_reference;
	compute_homography_to_reference(homograhpy, homograhpy_to_reference);

	// Exposure compensation:
	exposure_compensation(images);

	// Warping, aligning and blend images:
	cv::Mat result = warp_align_blend(images, homograhpy_to_reference);

	// Cropping image:
	result = crop_img(result);

	// Saving result:
	std::cout << "Saving result to: " << save_folder_path << std::endl << std::endl;
	cv::imwrite(save_folder_path, result);

	// Showing the result:
	std::cout << "Displaying stitched image of the facade..." << std::endl;
	cv::namedWindow("Result", cv::WINDOW_KEEPRATIO);
	cv::resizeWindow("Result", result.cols / 4, result.rows / 4);
	cv::imshow("Result", result);
	cv::waitKey(0);
}

std::vector<std::vector<cv::Mat>> image_loader(std::string path) {

	std::vector<std::vector<cv::Mat>> images;
	std::vector<std::string> images_path;

	std::cout << "Loading images: " << std::endl;
	cv::glob(path, images_path);

	for (int i = 1; i < 10; i++) {
		std::vector<cv::Mat> temp_images;
		bool row_found = false;

		for (int j = 0; j < images_path.size(); j++) {
			if (i == (int)images_path[j].at(images_path[j].size() - 7) - 48) {
				std::cout << images_path[j] << std::endl;
				cv::Mat image = cv::imread(images_path[j]);
				temp_images.push_back(image);
				row_found = true;
			}
		}

		if (row_found == false) break;
		images.push_back(temp_images);
	}

	std::cout << "Done loading images... " << std::endl << std::endl;

	return images;
}

void detect_and_extract_keypoints(std::vector<std::vector<cv::Mat>> input_images, std::vector<std::vector<std::vector<cv::KeyPoint>>>& output_keypoints, std::vector<std::vector<cv::Mat>>& output_descriptors) {

	std::cout << "Starting to detect and extract features for the loaded images: " << std::endl;

	cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create(1000, 3, 0.01, 10, 1.6);

	for (int i = 0; i < input_images.size(); i++) {
		std::vector<std::vector<cv::KeyPoint>> row_keypoints;
		std::vector<cv::Mat> row_descriptors;

		for (int j = 0; j < input_images[i].size(); j++) {
			std::vector<cv::KeyPoint> keypoint;
			cv::Mat descriptor;

			std::cout << "Image: row " + std::to_string(i + 1) + ", column " + std::to_string(j + 1) << "... ";

			detector->detectAndCompute(input_images[i][j], cv::noArray(), keypoint, descriptor);

			std::cout << "features found: " << keypoint.size() << std::endl;

			row_keypoints.push_back(keypoint);
			row_descriptors.push_back(descriptor);
		}

		output_keypoints.push_back(row_keypoints);
		output_descriptors.push_back(row_descriptors);
	}

	std::cout << "Done finding features..." << std::endl << std::endl;
}

void neighbor_matcher(std::vector<std::vector<cv::Mat>>& input_descriptors, std::vector<std::vector<std::vector<cv::DMatch>>>& output_matches) {

	std::cout << "Starting to match features between neighboring images: " << std::endl;

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);

	for (int i = 0; i < input_descriptors.size(); i++) {
		std::vector<std::vector<cv::DMatch>> row_matches;

		for (int j = 0; j < input_descriptors[i].size(); j++) {

			std::cout << "Image: row " + std::to_string(i + 1) + ", column " + std::to_string(j) << " & " << std::to_string(j + 1) << "...";

			std::vector<std::vector<cv::DMatch>> knn_matches;

			if (j == 0) {
				if (i == 0) {
					std::cout << std::endl;
					std::vector<cv::DMatch> first_image;
					row_matches.push_back(first_image);
					continue;
				}

				// For the first image in a row where 'i' is not 0:
				matcher->knnMatch(input_descriptors[i - 1][j], input_descriptors[i][j], knn_matches, 2);
			}
			else {
				matcher->knnMatch(input_descriptors[i][j - 1], input_descriptors[i][j], knn_matches, 2);
			}

			//-- Filter matches using the Lowe's ratio test
			const float ratio_thresh = 0.8f;
			std::vector<cv::DMatch> filtered_matches;

			for (int k = 0; k < knn_matches.size(); k++) {
				if (knn_matches[k][0].distance < ratio_thresh * knn_matches[k][1].distance) {
					filtered_matches.push_back(knn_matches[k][0]);
				}
			}

			std::cout << " remaining matches: " << filtered_matches.size() << std::endl;

			row_matches.push_back(filtered_matches);
		}

		output_matches.push_back(row_matches);
	}

	std::cout << "Done matching features..." << std::endl << std::endl;
}

void compute_homography(std::vector<std::vector<std::vector<cv::KeyPoint>>> input_keypoints, std::vector<std::vector<std::vector<cv::DMatch>>> input_matches, std::vector<std::vector<cv::Mat>>& output_homography) {

	std::cout << "Starting to compute homography between neighbors: " << std::endl;

	for (int i = 0; i < input_matches.size(); i++) {
		std::vector<cv::Mat> row_homography;

		for (int j = 0; j < input_matches[i].size(); j++) {
			std::cout << "Image: row " + std::to_string(i + 1) + ", column " + std::to_string(j) << " & " << std::to_string(j + 1) << "...";

			// Extract location of good matches
			std::vector<cv::Point2f> points_in_image_1, points_in_image_2;

			if (j == 0) {
				if (i == 0) {
					cv::Mat identity_matrix = cv::Mat::eye(cv::Size(3, 3), CV_64F);
					std::cout << std::endl;
					row_homography.push_back(identity_matrix);
					continue;
				}

				// If j = 0 && i != 0:
				for (int k = 0; k < input_matches[i][j].size(); k++) {
					points_in_image_1.push_back(input_keypoints[i - 1][j][input_matches[i][j][k].queryIdx].pt);
					points_in_image_2.push_back(input_keypoints[i][j][input_matches[i][j][k].trainIdx].pt);
				}

			}
			else {
				for (int k = 0; k < input_matches[i][j].size(); k++) {
					points_in_image_1.push_back(input_keypoints[i][j - 1][input_matches[i][j][k].queryIdx].pt);
					points_in_image_2.push_back(input_keypoints[i][j][input_matches[i][j][k].trainIdx].pt);
				}

			}

			// Finding homography matrix:
			cv::Mat ransac_mat;
			cv::Mat homography = cv::findHomography(points_in_image_2, points_in_image_1, cv::RANSAC, 1.0, ransac_mat, 2000, 0.999);

			float inlier = 0, outlier = 0;
			for (int k = 0; k < ransac_mat.rows; k++) {

				// We have an inlier:
				if ((int)ransac_mat.at<uchar>(k, 0) == 1) {
					inlier = inlier + 1;
				}

				// We have an outlier:
				else outlier = outlier + 1;
			}

			std::cout << " RANSAC inliers: " << inlier << std::endl;

			row_homography.push_back(homography);
		}

		output_homography.push_back(row_homography);
	}

	std::cout << "Done computing homography..." << std::endl << std::endl;
}

void compute_homography_to_reference(std::vector<std::vector<cv::Mat>> input_homograhpy, std::vector<std::vector<cv::Mat>>& output_homograhpy) {

	std::cout << "Starting to compute homography to reference: " << std::endl;

	for (int i = 0; i < input_homograhpy.size(); i++) {
		std::vector<cv::Mat> row_homography;

		for (int j = 0; j < input_homograhpy[i].size(); j++) {

			std::cout << "Image: row " + std::to_string(i + 1) + ", column " + std::to_string(j) << " & " << std::to_string(j + 1) << std::endl;

			// In case of the first image, no additional homography is needed:
			if (i == 0 && j == 0) {
				row_homography.push_back(input_homograhpy[i][j]);
				continue;
			}

			// In of the first row:
			if (i == 0) {
				cv::Mat homograhpy = input_homograhpy[0][1].clone();

				for (int k = 1; k < j; k++) {
					homograhpy *= input_homograhpy[0][k + 1];
				}

				row_homography.push_back(homograhpy);
			}

			// In case of any other row:
			else {

				cv::Mat homograhpy = input_homograhpy[1][0].clone();
				if (i > 1) homograhpy *= input_homograhpy[2][0];

				for (int k = 0; k < j; k++) {
					homograhpy *= input_homograhpy[i][k + 1];
				}

				row_homography.push_back(homograhpy);
			}
		}

		output_homograhpy.push_back(row_homography);
	}

	std::cout << "Done computing homography to reference..." << std::endl << std::endl;
}

void exposure_compensation(std::vector<std::vector<cv::Mat>>& input_images) {
	std::cout << "Starting to exposure compensate images..." << std::endl;

	cv::Ptr<cv::detail::ExposureCompensator> exposure = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN);

	std::vector<cv::Point> corners;
	std::vector<cv::UMat> temp_images;
	std::vector<cv::UMat> temp_masks;

	for (int i = 0; i < input_images.size(); i++) {
		for (int j = 0; j < input_images[i].size(); j++) {

			std::cout << "Preparing image: row " << i + 1 << ", column: " << j + 1 << std::endl;

			// Removing black windows:
			cv::Mat mask = cv::Mat::zeros(cv::Size(input_images[i][j].cols, input_images[i][j].rows), CV_8U);
			mask = 255;

			for (int x = 0; x < input_images[i][j].cols; x++) {
				for (int y = 0; y < input_images[i][j].rows; y++) {

					if (input_images[i][j].at<cv::Vec3b>(cv::Point(x, y))[0] < 5 &&
						input_images[i][j].at<cv::Vec3b>(cv::Point(x, y))[1] < 5 &&
						input_images[i][j].at<cv::Vec3b>(cv::Point(x, y))[2] < 5) mask.at<uchar>(cv::Point(x, y)) = 0;
				}
			}

			corners.push_back(cv::Point(0, 0));
			temp_images.push_back(input_images[i][j].getUMat(cv::ACCESS_READ).clone());
			temp_masks.push_back(mask.getUMat(cv::ACCESS_READ).clone());
		}
	}

	std::cout << "Exposure compensating..." << std::endl;
	exposure->feed(corners, temp_images, temp_masks);

	for (int i = 0; i < temp_images.size(); i++) {
		exposure->apply(i, corners[i], temp_images[i], temp_masks[i]);

		if (i < input_images[0].size()) {
			input_images[0][i] = temp_images[i].getMat(cv::ACCESS_READ).clone();
		}
		else if (i < input_images[0].size() + input_images[1].size()) {
			input_images[1][i - input_images[0].size()] = temp_images[i].getMat(cv::ACCESS_READ).clone();
		}
		else if (i < input_images[0].size() + input_images[1].size() + input_images[2].size()) {
			input_images[2][i - (input_images[0].size() + input_images[1].size())] = temp_images[i].getMat(cv::ACCESS_READ).clone();
		}
	}

	std::cout << "Done exposure compensating images..." << std::endl << std::endl;
}

cv::Mat warp_align_blend(std::vector<std::vector<cv::Mat>> images, std::vector<std::vector<cv::Mat>> homograhpy_to_reference) {

	std::cout << "Starting to warp images... " << std::endl;

	std::vector<cv::UMat> images_warped, masks_warped;
	std::vector<cv::Point> corners;
	std::vector<cv::Size> image_sizes;

	cv::Mat total_mask = cv::Mat::zeros(cv::Size(5000, 3000), CV_8U);

	for (int i = 0; i < images.size(); i++) {
		for (int j = 0; j < images[i].size(); j++) {

			std::cout << "Warping images: row " + std::to_string(i + 1) + ", column " + std::to_string(j + 1) + "..." << std::endl;

			// Creating image and mask:
			cv::Mat image, mask;

			cv::Mat unwarped_mask = cv::Mat::zeros(cv::Size(images[i][j].cols, images[i][j].rows), CV_8U);
			unwarped_mask(cv::Rect(1, 1, unwarped_mask.cols - 2, unwarped_mask.rows - 2)) = 255;

			// Warping image and mask:
			cv::warpPerspective(images[i][j], image, homograhpy_to_reference[i][j], cv::Size(5000, 3000), cv::INTER_LINEAR, cv::BORDER_REFLECT);
			cv::warpPerspective(unwarped_mask, mask, homograhpy_to_reference[i][j], cv::Size(5000, 3000));

			cv::Mat temp_mask = cv::Mat::zeros(cv::Size(5000, 3000), CV_8U);

			for (int x = 0; x < total_mask.cols; x++) {
				for (int y = 0; y < total_mask.rows; y++) {
					
					if (total_mask.at<uchar>(cv::Point(x, y)) == 0 && mask.at<uchar>(cv::Point(x, y)) == 255) {
						temp_mask.at<uchar>(cv::Point(x, y)) = 255;
						total_mask.at<uchar>(cv::Point(x, y)) = 255;
					}
				}
			}

			// Pushing images, masks, and top-left corners to matrices:
			images_warped.push_back(image.getUMat(cv::ACCESS_READ).clone());
			masks_warped.push_back(temp_mask.getUMat(cv::ACCESS_READ).clone());
			corners.push_back(cv::Point(0, 0));
			image_sizes.push_back(cv::Size(5000, 3000));
		}
	}

	std::cout << std::endl << "Starting to blend... " << std::endl;
	// Blending images based on the found seams:
	cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(cv::detail::Blender::MULTI_BAND, true);

	blender->prepare(corners, image_sizes);

	std::cout << "Feeding blender... " << std::endl;
	
	// Feeding warped images and masks into the blender:
	for (int i = 0; i < images_warped.size(); i++) {
		blender->feed(images_warped[i], masks_warped[i], corners[i]);
	}

	// Getting blended result:
	std::cout << "Blending... " << std::endl;
	cv::UMat result_image, result_mask;
	blender->blend(result_image, result_mask);
	result_image.convertTo(result_image, CV_8UC3);

	cv::Mat return_image = result_image.getMat(cv::ACCESS_READ).clone();

	std::cout << "Done blending images..." << std::endl << std::endl;

	// Returning exposure compensated and blended image:
	return return_image;
}

cv::Mat crop_img(cv::Mat result) {

	std::cout << "Starting to crop image..." << std::endl;

	// vector with all non-black point positions
	std::vector<cv::Point> nonBlackList;

	// add all non-black points to the vector
	for (int j = 0; j < result.rows; ++j)
		for (int i = 0; i < result.cols; ++i)
		{
			// if not black: add to the list
			if (result.at<cv::Vec3b>(j, i) != cv::Vec3b(0, 0, 0)) {
				nonBlackList.push_back(cv::Point(i, j));
			}
		}

	// create bounding rect around points
	cv::Rect croppingrect = cv::boundingRect(nonBlackList);

	result = result(croppingrect);

	std::cout << "Done cropping image..." << std::endl << std::endl;
	return result;
}