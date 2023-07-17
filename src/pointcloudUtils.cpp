#include "pointcloud/pointcloudUtils.h"
#include <ros/ros.h>

void pointcloud::segIm2PointCloud(const pcl::PointCloud<pcl::PointXYZRGB> &inputCloud, 
                                  cv::Mat &segmentedImage,
                                  geometry_msgs::Point &spherePosCamFrame,
                                  float sphereRadius,
                                  float maxRange,
                                  pcl::PointCloud<pcl::PointXYZRGB> &outFullCloud,
                                  pcl::PointCloud<pcl::PointXYZRGB> &outSegCloud,
                                  cv::Mat &segmentedImageFiltered
)
{
    outFullCloud.header = inputCloud.header;
    outSegCloud.header = inputCloud.header;

    uint32_t height = inputCloud.height;
    uint32_t width = inputCloud.width;

    pcl::PointXYZRGB point;

    uint8_t bad_val = 255;
    uint8_t bg_val = 254;
    uint8_t seg_id;

    float radiusSquared = (sphereRadius + 1e-6)*(sphereRadius + 1e-6);
    float maxRangeSquared = maxRange*maxRange;
    float sphereDist;
    float rangeDist;

    if (height != segmentedImage.rows)
        ROS_WARN("cloud and image height do not match");

    if (width != segmentedImage.cols)
        ROS_WARN("cloud and image width do not match");

    for (uint32_t row = 0; row < height; row++)
    {
        for (uint32_t col = 0; col < width; col++)
        {
            segmentedImageFiltered.at<uint8_t>(row, col) = 255;

            point = inputCloud.at(col, row);

            seg_id = segmentedImage.at<uint8_t>(row, col);
            if (seg_id == bg_val)
                continue;

            //skip if not finite
            //this is because we set bad points to nan and isfinute checks for nan
            if (!(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)))
                continue;

            rangeDist = point.x*point.x + point.y*point.y + point.z*point.z;

            //outFullCloud.push_back(point);
            
            if (rangeDist > maxRangeSquared)
                continue;

            outFullCloud.push_back(point);

            if (seg_id == bad_val)
                continue;

            sphereDist = (point.x - spherePosCamFrame.x)*(point.x - spherePosCamFrame.x) +
                               (point.y - spherePosCamFrame.y)*(point.y - spherePosCamFrame.y) +
                               (point.z - spherePosCamFrame.z)*(point.z - spherePosCamFrame.z);
            
            if (sphereDist > radiusSquared)
                continue;

            outSegCloud.push_back(point);
            segmentedImageFiltered.at<uint8_t>(row, col) = seg_id;
        }
    }
}

void pointcloud::segIm2PointCloud(const pcl::PointCloud<pcl::PointXYZRGB> &inputCloud, 
                                  cv::Mat &segmentedImage,
                                  geometry_msgs::Point &spherePosCamFrame,
                                  float sphereRadius,
                                  float maxRange,
                                  pcl::PointCloud<pcl::PointXYZRGBA> &outFullCloud,
                                  pcl::PointCloud<pcl::PointXYZRGB> &outSegCloud,
                                  cv::Mat &segmentedImageFiltered
)
{
    outFullCloud.header = inputCloud.header;
    outSegCloud.header = inputCloud.header;

    uint32_t height = inputCloud.height;
    uint32_t width = inputCloud.width;

    pcl::PointXYZRGB point;
    pcl::PointXYZRGBA alphaPoint;

    uint8_t bad_val = 255;
    uint8_t bg_val = 254;
    uint8_t seg_id;

    float radiusSquared = (sphereRadius + 1e-6)*(sphereRadius + 1e-6);
    float maxRangeSquared = maxRange*maxRange;
    float sphereDist;
    float rangeDist;

    if (height != segmentedImage.rows)
        ROS_WARN("cloud and image height do not match");

    if (width != segmentedImage.cols)
        ROS_WARN("cloud and image width do not match");

    for (uint32_t row = 0; row < height; row++)
    {
        for (uint32_t col = 0; col < width; col++)
        {
            segmentedImageFiltered.at<uint8_t>(row, col) = 255;

            point = inputCloud.at(col, row);
            alphaPoint.x = point.x;
            alphaPoint.y = point.y;
            alphaPoint.z = point.z;
            alphaPoint.r = point.r;
            alphaPoint.g = point.g;
            alphaPoint.b = point.b;
            alphaPoint.a = 0;

            seg_id = segmentedImage.at<uint8_t>(row, col);
            if (seg_id == bg_val)
                continue;

            //skip if not finite
            //this is because we set bad points to nan and isfinute checks for nan
            if (!(std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)))
                continue;

            rangeDist = point.x*point.x + point.y*point.y + point.z*point.z;
            
            if (rangeDist > maxRangeSquared)
            {
                //outFullCloud.push_back(alphaPoint);
                continue;
            }

            //outFullCloud.push_back(point);

            if (seg_id == bad_val)
            {
                outFullCloud.push_back(alphaPoint);
                continue;
            }

            sphereDist = (point.x - spherePosCamFrame.x)*(point.x - spherePosCamFrame.x) +
                               (point.y - spherePosCamFrame.y)*(point.y - spherePosCamFrame.y) +
                               (point.z - spherePosCamFrame.z)*(point.z - spherePosCamFrame.z);
            
            if (sphereDist > radiusSquared)
            {
                outFullCloud.push_back(alphaPoint);
                continue;
            }

            outSegCloud.push_back(point);
            segmentedImageFiltered.at<uint8_t>(row, col) = seg_id;

            alphaPoint.a = 255;
            outFullCloud.push_back(alphaPoint);
        }
    }
}