/**
 * @file attention_viewer_ros.h
 * @author Parker Lusk <parkerclusk@gmail.com>
 */

#pragma once

#include <memory>
#include <string>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud.h>

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/Camera.h>

class AttentionViewerROS
{
public:
  AttentionViewerROS(ros::NodeHandle nh);
  ~AttentionViewerROS() = default;

private:
  ros::NodeHandle nh_;

  // message filter sync
  using SyncPolicy = message_filters::sync_policies::ExactTime<
        sensor_msgs::Image, sensor_msgs::PointCloud, sensor_msgs::PointCloud>;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  // ROS sub/pub
  image_transport::Publisher img_pub_;
  image_transport::SubscriberFilter img_sf_;
  message_filters::Subscriber<sensor_msgs::PointCloud> features_f_;
  message_filters::Subscriber<sensor_msgs::PointCloud> selinfo_f_;

  camodocal::CameraPtr m_camera_; ///< geometric camera model

  int frame_ = 0; ///< what is the current frame number to process

  // subscriber callbacks
  void callback(const sensor_msgs::ImageConstPtr& _img,
                const sensor_msgs::PointCloudConstPtr& _features,
                const sensor_msgs::PointCloudConstPtr& _selinfo);
};
