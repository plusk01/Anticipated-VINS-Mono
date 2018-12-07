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
#include "utility.h"

class HorizonGenerator
{
public:
  HorizonGenerator(ros::NodeHandle nh);
  ~HorizonGenerator() = default;

  /**
   * @brief      Generate horizon using constant acceleration IMU model
   *
   * @param[in]  state_0            The state of the prev frame (k-1)
   * @param[in]  state_1            The state of the current (yet-to-be-corrected) frame (k)
   * @param[in]  a                  Latest IMU acceleration measurement
   * @param[in]  w                  Latest IMU angular vel measurement
   * @param[in]  nrImuMeasurements  Number of IMU measurements from prev frame to now
   * @param[in]  deltaImu           Sampling period of IMU measurements
   *
   * @return     a state horizon
   */
  state_horizon_t imu(const state_t& state_0, const state_t& state_1,
                      const Eigen::Vector3d& a, const Eigen::Vector3d& w,
                      int nrImuMeasurements, double deltaImu);

  /**
   * @brief      Generate horizon from ground truth data
   *
   * @param[in]  state_0     The state of the previous frame (k-1)
   * @param[in]  state_1     The state of the current (yet-to-be-corrected) frame (k)
   * @param[in]  deltaFrame  time between previous two img frames (secs)
   *
   * @return     a state horizon
   */
  state_horizon_t groundTruth(const state_t& state_0, const state_t& state_1, double deltaFrame);

  /**
   * @brief      Visualize a state propagated over a horizon
   *
   * @param[in]  header  The header message from the frame k (start)
   * @param[in]  x_kkH   The state horizon to visualize
   */
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
   * @param[in]  deltaFrame  The image sampling period (secs)
   *
   * @return     The next frame pose (ground truth).
   */
  truth_t getNextFrameTruth(int& idx, double deltaFrame);
};
