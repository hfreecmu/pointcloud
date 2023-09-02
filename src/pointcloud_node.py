#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo, PointField
from sensor_msgs import point_cloud2
import message_filters
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct
import ctypes

def get_intrinsics(camera_info):
    P = list(camera_info.P)
    f_norm = P[0]
    baseline = P[3] / P[0]
    cx = P[2]
    cy = P[6]
    intrinsics = (baseline, f_norm, cx, cy)

    return intrinsics

#bilateral filter
def bilateral_filter(disparity, intrinsics, bilateral_d,
                     bilateral_sc, bilateral_ss):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm
    z_new = cv2.bilateralFilter(z, bilateral_d, bilateral_sc, bilateral_ss)

    stub_new = z_new / f_norm
    disparity_new = -baseline / stub_new

    return disparity_new

#extract depth discontinuities
def extract_depth_discontuinities(disparity, intrinsics, disc_use_rat,
                                  disc_rat_thresh, disc_dist_thresh):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(z, element)
    erosion = cv2.erode(z, element)

    dilation -= z
    erosion = z - erosion

    max_image = np.max((dilation, erosion), axis=0)

    if disc_use_rat:
        ratio_image = max_image / z
        _, discontinuity_map = cv2.threshold(ratio_image, disc_rat_thresh, 1.0, cv2.THRESH_BINARY)
    else:
        _, discontinuity_map = cv2.threshold(max_image, disc_dist_thresh, 1.0, cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    discontinuity_map = cv2.morphologyEx(discontinuity_map, cv2.MORPH_CLOSE, element)

    return discontinuity_map

def compute_points(disparity, intrinsics):
    baseline, f_norm, cx, cy = intrinsics
    stub = -baseline / disparity #*0.965

    x_pts, y_pts = np.meshgrid(np.arange(disparity.shape[1]), np.arange(disparity.shape[0]))

    x = stub * (x_pts - cx)
    y = stub * (y_pts - cy)
    z = stub*f_norm

    points = np.stack((x, y, z), axis=2)

    return points

#TODO not super fast (1.5 seconds to iterate through cluod and create message)
#but should be good enough for now?

#TODO convert this into c++
class PointCloudHandler:
    def __init__(self, pub, bilateral_filter, bilateral_d, bilateral_sc,
                 bilateral_ss, disc_use_rat, disc_rat_thresh, disc_dist_thresh,
                 z_only, max_dist, min_disparity):
        self.pub = pub
        self.bilateral_filter = bilateral_filter
        self.bilateral_d = bilateral_d
        self.bilateral_sc = bilateral_sc
        self.bilateral_ss = bilateral_ss
        self.disc_use_rat = disc_use_rat
        self.disc_rat_thresh = disc_rat_thresh
        self.disc_dist_thresh = disc_dist_thresh
        self.z_only = z_only
        self.max_dist = max_dist
        self.min_disparity = min_disparity
        self.bridge = CvBridge()

    def buildPointCloud(self, ros_left_image, ros_right_info, ros_disp_image):
        colors = np.copy(self.bridge.imgmsg_to_cv2(ros_left_image, desired_encoding='rgb8'))
        disparity = np.copy(self.bridge.imgmsg_to_cv2(ros_disp_image))

        intrinsics = get_intrinsics(ros_right_info)

        inf_inds = np.where(disparity <= 0)
        if np.min(disparity) <= self.min_disparity:
            disparity[disparity <= self.min_disparity] = self.min_disparity + 1e-6

        if self.bilateral_filter:
            disparity = bilateral_filter(disparity, intrinsics, self.bilateral_d,
                                         self.bilateral_sc, self.bilateral_ss)
            
        discontinuity_map = extract_depth_discontuinities(disparity, intrinsics, 
                                                          self.disc_use_rat,
                                                          self.disc_rat_thresh,
                                                          self.disc_dist_thresh)
        
        points = compute_points(disparity, intrinsics)

        is_nan = np.zeros(points.shape[0:-1])
        nan_inds = np.where(discontinuity_map > 0) 
        is_nan[nan_inds] = 1
        is_nan[inf_inds] = 1

        ros_points = []
        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                if is_nan[i, j]:
                    x, y, z = float(np.nan), float(np.nan), float(np.nan)
                    r, g, b = int(0), int(0), int(0)
                else:
                    x, y, z = points[i, j]
                    r, g, b = colors[i, j]

                a = 255
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]  
                pt = [x, y, z, rgb]
                ros_points.append(pt) 
                                

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgba", 12, PointField.UINT32, 1)
        ]

        #pc2 = point_cloud2.create_cloud(ros_left_image.header, fields, ros_points)

        cloud_struct = struct.Struct(point_cloud2._get_struct_fmt(False, fields))
        buff = ctypes.create_string_buffer(cloud_struct.size * len(ros_points))
        point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
        offset = 0
        for p in ros_points:
            pack_into(buff, offset, *p)
            offset += point_step

        pc2 = PointCloud2(header=ros_left_image.header,
                          height=points.shape[0],
                          width=points.shape[1],
                          is_dense=False,
                          is_bigendian=False,
                          fields=fields,
                          point_step=cloud_struct.size,
                          row_step=cloud_struct.size * points.shape[1],
                          data=buff.raw)


        self.pub.publish(pc2)

        rospy.loginfo("Point Cloud succesfully processed from disparity ...............................")

def run_segmentor_service():
    rospy.init_node("pointcloud")

    bilateral_filter = rospy.get_param("~bilateral_filter")
    bilateral_d = rospy.get_param("~bilateral_d")
    bilateral_sc = rospy.get_param("~bilateral_sc")
    bilateral_ss = rospy.get_param("~bilateral_ss")
    
    disc_use_rat = rospy.get_param("~disc_use_rat")
    disc_rat_thresh = rospy.get_param("~disc_rat_thresh")
    disc_dist_thresh = rospy.get_param("~disc_dist_thresh")

    z_only = rospy.get_param("~z_only")
    max_dist = rospy.get_param("~max_range")

    min_disparity = rospy.get_param("~min_disparity")

    pub = rospy.Publisher('/pointcloud', PointCloud2, queue_size=5)

    pointcloud_handler = PointCloudHandler(pub, bilateral_filter, bilateral_d, bilateral_sc,
                                           bilateral_ss, disc_use_rat, disc_rat_thresh, disc_dist_thresh,
                                           z_only, max_dist, min_disparity)

    left_info_sub = message_filters.Subscriber('/theia/left/image_rect_color', Image, queue_size=2)
    right_info_sub = message_filters.Subscriber('/theia/right/camera_info', CameraInfo, queue_size=2)
    disparity_sub = message_filters.Subscriber('/disparity_image', Image, queue_size=2)

    ts = message_filters.ApproximateTimeSynchronizer([left_info_sub, right_info_sub, disparity_sub], 9, 0.1)
    ts.registerCallback(pointcloud_handler.buildPointCloud)
    rospy.loginfo("Point Cloud Node Started")

    rospy.spin()

if __name__ == "__main__":
    run_segmentor_service()