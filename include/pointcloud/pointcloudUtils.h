#pragma once

#include <geometry_msgs/Pose.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

namespace pointcloud
{

void segIm2PointCloud(const pcl::PointCloud<pcl::PointXYZRGB> &inputCloud, 
                      cv::Mat &segmentedImage,
                      geometry_msgs::Point &spherePosition,
                      float sphereRadius,
                      float maxRange,
                      pcl::PointCloud<pcl::PointXYZRGB> &outFullCloud,
                      pcl::PointCloud<pcl::PointXYZRGB> &outSegCloud,
                      cv::Mat &segmentedImageFiltered
                      );

void segIm2PointCloud(const pcl::PointCloud<pcl::PointXYZRGB> &inputCloud, 
                      cv::Mat &segmentedImage,
                      geometry_msgs::Point &spherePosition,
                      float sphereRadius,
                      float maxRange,
                      pcl::PointCloud<pcl::PointXYZRGBA> &outFullCloud,
                      pcl::PointCloud<pcl::PointXYZRGB> &outSegCloud,
                      cv::Mat &segmentedImageFiltered
                      );

} //end namespace pointcloud