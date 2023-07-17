
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>

//TODO reconcile intrinsic functions in this and disparity_sgbm.cpp into common file
struct Intrinsics
{
    float baseline;
    float f_norm;
    float cx;
    float cy;
};

std::shared_ptr<Intrinsics> getCameraInfo(const sensor_msgs::CameraInfoConstPtr &rightCameraInfo)
{
    //TODO is this right combining left and right?
    //is left f_norm and right f_norm the same?
    //which should be used when calculating baseline?

    float baseline = rightCameraInfo->P[3] / rightCameraInfo->P[0];
    float f_norm = rightCameraInfo->P[0];
    float cx = rightCameraInfo->P[2];
    float cy = rightCameraInfo->P[6];

    std::shared_ptr<Intrinsics> intrinsics = std::make_shared<Intrinsics>();
    intrinsics->baseline = baseline;
    intrinsics->f_norm = f_norm;
    intrinsics->cx = cx;
    intrinsics->cy = cy;

    return intrinsics;
}

class PointCloudHandler 
{
public:
PointCloudHandler(ros::NodeHandle &nh_, std::string pointCloudTopic, int pointCloudQueuesize,
                  int minDisparity, float maxRange
):minDisparity(minDisparity), maxRange(maxRange)
{
    pointCloudPub = nh_.advertise<sensor_msgs::PointCloud2>(pointCloudTopic, pointCloudQueuesize);
}

void buildPointCloud(const sensor_msgs::ImageConstPtr &leftImage, 
                     const sensor_msgs::CameraInfoConstPtr &rightCameraInfo,
                     const sensor_msgs::ImageConstPtr &disparityImage
)
{
    //Get camera info
    if (cameraInfo == nullptr)
        cameraInfo = getCameraInfo(rightCameraInfo);

    //rgb for point cloud
    cv::Mat left = cv_bridge::toCvCopy(*leftImage, "rgb8")->image;
    cv::Mat disparity = cv_bridge::toCvCopy(*disparityImage, "")->image;

    sensor_msgs::PointCloud2 pointCloudMsg;
    pointCloudMsg.header.stamp = disparityImage->header.stamp;
    pointCloudMsg.header.frame_id = disparityImage->header.frame_id;
    pointCloudMsg.height = disparity.rows;
    pointCloudMsg.width  = disparity.cols;
    pointCloudMsg.is_bigendian = false;
    pointCloudMsg.is_dense = false; // there may be invalid points

    sensor_msgs::PointCloud2Modifier pointCloudModifier(pointCloudMsg);
    pointCloudModifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    sensor_msgs::PointCloud2Iterator<float> iter_x(pointCloudMsg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(pointCloudMsg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(pointCloudMsg, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(pointCloudMsg, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(pointCloudMsg, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(pointCloudMsg, "b");

    //there shouldn't be but just in case there are NaNs
    cv::patchNaNs(disparity, minDisparity-1);

    float x;
    float y;
    float z;

    float r;
    float g;
    float b;

    float stub;
    float disp;

    float bad_point = std::numeric_limits<float>::quiet_NaN();
    
    for (int v = 0; v < disparity.rows; ++v)
    {
        for (int u = 0; u < disparity.cols; ++u, ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b)
        {

            disp = disparity.at<float>(v, u);

            //if it is an invaliid disparity, we want to set it to bad point
            //so that it will be skipped.  Do not want to set it to far away as it 
            //just might be a not found disparity that is close.
            //-1
            if (disp < minDisparity)
            {
                *iter_x = *iter_y = *iter_z = bad_point;
                *iter_r = *iter_g = *iter_b = 0;
                continue;
            }

            if (disp==0)
                stub = maxRange / cameraInfo->f_norm;
            else
                stub = (-cameraInfo->baseline / disp);

            *iter_x = (u - cameraInfo->cx) * stub;
            *iter_y = (v - cameraInfo->cy) * stub;
            *iter_z = (cameraInfo->f_norm * stub);

            if (stub == 0)
            {
                ROS_WARN_STREAM("Stub 0. disp is: " << disp);
            }

            const cv::Vec3b& rgb = left.at<cv::Vec3b>(v, u);

            *iter_r = rgb[0];
            *iter_g = rgb[1];
            *iter_b = rgb[2];

            if (cameraInfo->f_norm * stub < 1e-6)
                ROS_WARN_STREAM("Bad pcl val. disp is: " << disp << ". Stub is " << stub << ". fnorm is " << cameraInfo->f_norm << 
                                ". max range is " << maxRange);
        }
    }

    pointCloudPub.publish(pointCloudMsg);


    ROS_INFO("Point Cloud succesfully processed from disparity ...............................");
}

~PointCloudHandler() {}

protected:
int minDisparity;
float maxRange;
std::shared_ptr<Intrinsics> cameraInfo;
ros::Publisher pointCloudPub;
};

int main(int argc, char** argv){

    //init ros node
    ros::init(argc, argv, "pointcloud");

    //create async spinner
    ros::AsyncSpinner spinner(4);
    spinner.start();

    //create node handles
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");

    //ros params
    int minDisparity;
    float maxRange;

    bool success = true;
    success &= nhp.getParam("min_disparity", minDisparity);
    success &= nhp.getParam("max_range", maxRange);

    if (!success)
    {
        ROS_WARN("Failed to read parameters");
        return 0;
    }

    //subscriptions
    std::string leftImageTopic = "/theia/left/image_rect_color";
    std::string rightInfoTopic = "/theia/right/camera_info";
    std::string disparityTopic = "/disparity_image";

    //made all queue sizes 5 or else messsages weren't received, not sure why
    message_filters::Subscriber<sensor_msgs::Image> leftImageSub(nh, leftImageTopic, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo> rightInfoSub(nh, rightInfoTopic, 1);
    message_filters::Subscriber<sensor_msgs::Image> disparitySub(nh, disparityTopic, 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image> MySyncPolicy;

    MySyncPolicy mySyncPolicy(9);
    mySyncPolicy.setAgePenalty(1.0);
    mySyncPolicy.setInterMessageLowerBound(ros::Duration(0.2));
    mySyncPolicy.setMaxIntervalDuration(ros::Duration(0.1));

    const MySyncPolicy myConstSyncPolicy = mySyncPolicy;

    message_filters::Synchronizer<MySyncPolicy> sync(myConstSyncPolicy, leftImageSub, rightInfoSub, disparitySub);

    PointCloudHandler pointCloudHandler(nh, "/pointcloud", 1,
                                        minDisparity, maxRange
    );

    sync.registerCallback(boost::bind(&PointCloudHandler::buildPointCloud, &pointCloudHandler, _1, _2, _3));

    ROS_INFO("Point Cloud Node Started");

    ros::waitForShutdown();

    return 0;
}