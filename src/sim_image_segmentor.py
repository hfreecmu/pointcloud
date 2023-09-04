#!/usr/bin/env python

import numpy as np
import matplotlib.colors
from sklearn.cluster import DBSCAN

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import message_filters
import ros_numpy

#TODO ultiamtely do this in c++, just easier here for now

#TODO these are just a hack so we can properly detect and segment
#segment colours will be different
purple = np.array([255, 100, 255])
blue = np.array([40, 40, 255])
orange = np.array([255, 140, 67])
red = np.array([253, 39, 39])
yellow = np.array([255, 255, 41])
teal = np.array([42, 254, 255])

purple_hsv = matplotlib.colors.rgb_to_hsv(purple.astype(float)/255)
blue_hsv = matplotlib.colors.rgb_to_hsv(blue.astype(float)/255)
orange_hsv = matplotlib.colors.rgb_to_hsv(orange.astype(float)/255)
red_hsv = matplotlib.colors.rgb_to_hsv(red.astype(float)/255)
yellow_hsv = matplotlib.colors.rgb_to_hsv(yellow.astype(float)/255)
teal_hsv = matplotlib.colors.rgb_to_hsv(teal.astype(float)/255)

hsv_colors = np.vstack((purple_hsv, blue_hsv, orange_hsv, red_hsv, yellow_hsv, teal_hsv))

bg_color = np.array([0.64, 0.86, 0.91])
bg_hsv = matplotlib.colors.rgb_to_hsv(bg_color.astype(float))

class ImageSegmentor:
    def __init__(self, pub, hsv_thresh, bg_thresh, min_area, shuffle_seg,
                 db_eps, db_min_samples):
        self.pub = pub
        self.hsv_thresh = hsv_thresh
        self.bg_thresh = bg_thresh
        self.min_area = min_area
        self.bridge = CvBridge()
        self.shuffle_seg = shuffle_seg
        self.hsv_colors = hsv_colors
        self.db_eps = db_eps
        self.db_min_samples = db_min_samples

    def segment_image(self, ros_image, ros_point_cloud):
        rospy.loginfo('segmenting image')

        image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='rgb8')
        n_rows, n_cols, _ = image.shape

        cam_points = ros_numpy.numpify(ros_point_cloud)
        cam_points = np.stack((cam_points['x'], cam_points['y'], cam_points['z']), axis=2)
        cam_points = cam_points.reshape((image.shape[0], image.shape[1], 3))

        hsv_image = matplotlib.colors.rgb_to_hsv(image.astype(float)/255)
        hsv_image_reshape = hsv_image.reshape(-1, 3)

        if self.shuffle_seg:
            hsv_inds = np.random.permutation(self.hsv_colors.shape[0])
            hsv_colors = self.hsv_colors[hsv_inds]
        else:
            hsv_colors = self.hsv_colors

        #start with fruitlet segmentation
        dists = np.linalg.norm((hsv_image_reshape - hsv_colors[:, None]), axis=2) #hsv_colors.shape[0] x hsv_image_reshape.shape[0]
        min_inds = np.argmin(dists, axis=0) #hsv_image_reshape.shape[0]
        min_dists = np.choose(min_inds, dists) #hsv_image_reshape.shape[0]
        seg_inds = np.argwhere(min_dists < self.hsv_thresh)[:, 0] #hsv_image_reshape.shape[0]
        
        #now do bg segmentation
        #commenting out for now
        #dists = np.linalg.norm(hsv_image_reshape - bg_hsv, axis=1) #hsv_image_reshape.shape[0]
        #bg_inds = np.argwhere(dists < self.bg_thresh)[:, 0]


        seg_ids = np.zeros((hsv_image_reshape.shape[0]), dtype=np.uint8) - 1
        #seg_ids[bg_inds] = 254
        seg_ids[seg_inds] = min_inds[seg_inds]

        seg_ids = seg_ids.reshape(n_rows, n_cols)

        #TODO not sure if min thresh will help
        unique_ids = np.unique(seg_ids)
        clustered_seg_ids = np.zeros(seg_ids.shape, dtype=np.uint8) - 1
        curr_id = 0
        for id in unique_ids:
            if id == 255:
                continue

            seg_inds = np.argwhere(seg_ids == id)
            seg_points = cam_points[seg_inds[:, 0], seg_inds[:, 1]]

            non_nan_inds = np.argwhere(~np.isnan(seg_points).any(axis=1))[:, 0]
            non_nan_seg_points = seg_points[non_nan_inds]

            if non_nan_seg_points.shape[0] == 0:
                continue

            db = DBSCAN(eps=self.db_eps, min_samples=self.db_min_samples).fit(non_nan_seg_points)
            labels = db.labels_

            for label_id in np.unique(labels):
                if label_id == -1:
                    continue

                label_inds = np.argwhere(labels == label_id)[:, 0]
                non_nan_label_inds = non_nan_inds[label_inds]
                seg_label_inds = seg_inds[non_nan_label_inds]

                if seg_label_inds.shape[0] < self.min_area:
                    continue

                clustered_seg_ids[seg_label_inds[:, 0], seg_label_inds[:, 1]] = curr_id
                curr_id += 1

        seg_id_msg = self.bridge.cv2_to_imgmsg(clustered_seg_ids)
        seg_id_msg.header = ros_image.header

        self.pub.publish(seg_id_msg)

        rospy.loginfo('image segmented')
        

def run_segmentor_service():
    rospy.init_node('image_segmentor')

    hsv_thresh = rospy.get_param("~hsv_thresh")
    bg_thresh = rospy.get_param("~bg_thresh")
    min_area = rospy.get_param("~min_area")
    shuffle_seg = rospy.get_param("~shuffle_seg")
    db_eps = rospy.get_param("~db_eps")
    db_min_samples = rospy.get_param("~db_min_samples")

    pub = rospy.Publisher('/segmented_image', Image, queue_size=5)

    segmentor = ImageSegmentor(pub, hsv_thresh, bg_thresh, min_area, shuffle_seg,
                               db_eps, db_min_samples)

    #sub = rospy.Subscriber('/theia/left/image_rect_color', Image, segmentor.segment_image, queue_size=5)
    image_sub = message_filters.Subscriber('/theia/left/image_rect_color', Image, queue_size=2)
    pointcloud_sub = message_filters.Subscriber('/pointcloud', PointCloud2, queue_size=2)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, pointcloud_sub], 6, 0.1)
    ts.registerCallback(segmentor.segment_image)

    rospy.spin()

if __name__ == "__main__":
    run_segmentor_service()
