#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Dense>

#include "state_defs.h"
#include "csviterator.h"

class HorizonGenerator
{
public:
  HorizonGenerator(ros::NodeHandle nh);
  ~HorizonGenerator() = default;

  state_horizon_t imu(const state_t& state_0, const state_t& state_1, int nrImuMeasurements, double deltaImu);

  state_horizon_t groundTruth(const state_t& state_0, const state_t& state_1, double timestamp, double deltaFrame);

  void visualize(const std_msgs::Header& header, const state_horizon_t& x_kkH);

private:
  // ROS stuff
  ros::NodeHandle nh_;
  ros::Publisher pub_horizon_;

  // struct and container for ground truth data read from CSV
  struct truth {
    double timestamp;
    Eigen::Vector3d p, v, w, a;
    Eigen::Quaterniond q;
  };
  std::vector<truth> truth_;
  int seek_idx_ = 0; ///< index variable that helps us "seek the truth"

  void loadGroundTruth(std::string data_csv);
};