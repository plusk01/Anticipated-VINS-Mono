/**
 * @file feature_tracker_node.cpp
 * @brief Simple node for visualization of selected features
 * @author Parker Lusk <parkerclusk@gmail.com>
 */

#include <ros/ros.h>

#include "anticipation/attention_viewer_ros.h"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "attention_viewer");
  ros::NodeHandle nh("~");
  AttentionViewerROS obj(nh);
  ros::spin();
  return 0;
}