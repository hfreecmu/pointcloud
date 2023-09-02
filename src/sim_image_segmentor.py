#!/usr/bin/env python

import numpy as np
import matplotlib.colors

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

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
    def __init__(self, pub, hsv_thresh, bg_thresh, min_area, shuffle_seg):
        self.pub = pub
        self.hsv_thresh = hsv_thresh
        self.bg_thresh = bg_thresh
        self.min_area = min_area
        self.bridge = CvBridge()
        self.shuffle_seg = shuffle_seg
        self.hsv_colors = hsv_colors

    def segment_image(self, ros_image):
        rospy.loginfo('segmenting image')

        image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='rgb8')
        n_rows, n_cols, _ = image.shape

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

        #TODO not usre if this will help
        unique_ids = np.unique(seg_ids)
        for id in unique_ids:
            if id == 255:
                continue

            seg_inds = np.argwhere(seg_ids == id)
            if seg_inds.shape[0] < self.min_area:
                seg_ids[seg_inds[:, 0], seg_inds[:, 1]] = 255
        #

        seg_id_msg = self.bridge.cv2_to_imgmsg(seg_ids)
        seg_id_msg.header = ros_image.header

        self.pub.publish(seg_id_msg)

        rospy.loginfo('image segmented')
        

def run_segmentor_service():
    rospy.init_node('image_segmentor')

    hsv_thresh = rospy.get_param("~hsv_thresh")
    bg_thresh = rospy.get_param("~bg_thresh")
    min_area = rospy.get_param("~min_area")
    shuffle_seg = rospy.get_param("~shuffle_seg")

    pub = rospy.Publisher('/segmented_image', Image, queue_size=5)

    segmentor = ImageSegmentor(pub, hsv_thresh, bg_thresh, min_area, shuffle_seg)

    sub = rospy.Subscriber('/theia/left/image_rect_color', Image, segmentor.segment_image, queue_size=5)

    rospy.spin()

if __name__ == "__main__":
    run_segmentor_service()
