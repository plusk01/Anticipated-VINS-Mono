#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>

#include <Eigen/Dense>

#include "state_defs.h"

class HorizonGenerator
{
public:
  HorizonGenerator(ros::NodeHandle nh);
  ~HorizonGenerator() = default;

  xhVector imu(const xVector& xk, int nrImuMeasurements, double deltaImu);

  xhVector groundTruth();

  void visualize(const std_msgs::Header& header, const xhVector& x_kkH);

private:
  // ROS stuff
  ros::NodeHandle nh_;
  ros::Publisher pub_horizon_;
};