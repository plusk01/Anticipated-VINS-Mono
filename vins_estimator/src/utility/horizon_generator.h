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

  state_horizon_t groundTruth(const state_t& state_0, const state_t& state_1, double deltaFrame);
  void visualize(const std_msgs::Header& header, const state_horizon_t& x_kkH);

private:
  // ROS stuff
  ros::NodeHandle nh_;
  ros::Publisher pub_horizon_;

  // struct and container for ground truth data read from CSV
  typedef struct {
    double timestamp;
    Eigen::Vector3d p, v, w, a;
    Eigen::Quaterniond q;
  } truth_t;
  std::vector<truth_t> truth_;
  int seek_idx_ = 0; ///< index variable that helps us "seek the truth"

  /**
   * @brief      Loads a EuRoC-style ground truth CSV into memory
   *
   * @param[in]  data_csv  The path of the data CSV file
   */
  void loadGroundTruth(std::string data_csv);

  /**
   * @brief      Given the current ground truth position, find what
   *             the pose of the camera should be at the next frame
   *
   * @param[in]  idx         Index of the current pose in the truth
   * @param[in]  timestamp   The timestamp of the current image frame
   * @param[in]  deltaFrame  The image sampling period (secs)
   *
   * @return     The next frame pose (ground truth).
   */
  truth_t getNextFrameTruth(int& idx, double timestamp, double deltaFrame);
};