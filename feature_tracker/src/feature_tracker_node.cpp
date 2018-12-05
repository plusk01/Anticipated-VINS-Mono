/**
 * @file feature_tracker_node.cpp
 * @author Parker Lusk <parkerclusk@gmail.com>
 */

#include <ros/ros.h>

#include "anticipation/feature_tracker_ros.h"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "feature_tracker");
  ros::NodeHandle nh("~");
  FeatureTrackerROS obj(nh);
  ros::spin();
  return 0;
}