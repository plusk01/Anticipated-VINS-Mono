/**
 * @file feature_tracker_ros.h
 * @author Parker Lusk <parkerclusk@gmail.com>
 */

#pragma once

#include <memory>
#include <string>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <anticipation/feature_tracker.h>

class FeatureTrackerROS
{
public:
  FeatureTrackerROS(ros::NodeHandle nh);
  ~FeatureTrackerROS() = default;

private:
  ros::NodeHandle nh_;

  // parameters
  std::string config_file_;
  int window_length_ = 10;
  anticipation::FeatureTracker::Parameters params_;

  // feature tracker
  std::unique_ptr<anticipation::FeatureTracker> tracker_;

  // ROS sub/pub
  image_transport::Subscriber img_sub_;
  image_transport::Publisher img_pub_;
  ros::Publisher features_pub_;

  // callbacks
  void imageCb(const sensor_msgs::ImageConstPtr& msg);

  // other methods
  void loadParameters();

  /**
   * @brief      Convenience method for safely reading required ROS parameters
   *
   * @param[in]  name  name of parameter on ROS parameter server
   *
   * @tparam     T     data type of the parameter
   *
   * @return     the parameter
   */
  template <typename T>
  T readROSParam(const std::string& name) const;
};

// ----------------------------------------------------------------------------

template <typename T>
T FeatureTrackerROS::readROSParam(const std::string& name) const
{
  T param;
  if (nh_.getParam(name, param)) {
    ROS_INFO_STREAM("[readROSParam] " << name << ": " << param);
  } else {
    ROS_ERROR_STREAM("[readROSParam] Could not read " << name);
    // nh_.shutdown();
  }
  return param;
}
