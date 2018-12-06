#pragma once

#include <map>
#include <utility>
#include <iostream>
#include <vector>

#include <ros/ros.h>
#include <std_msgs/Header.h>

#include <Eigen/Dense>

#include "estimator.h"

#define HORIZON 10 ///< number of frames to look into the future

static const Eigen::Vector3d gravity = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, -9.80665;
  return tmp;
}();

class FeatureSelector
{
public:
  // VINS-Mono calls this an 'image', but note that it is simply a collection of features
  using image_t = std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;

  FeatureSelector(ros::NodeHandle nh, Estimator& estimator);
  ~FeatureSelector() = default;

  /**
   * @brief      Process features to be selected
   *
   * @param[in]  image   A set of calibrated pixels, indexed by feature id
   * @param[in]  header  The header (with timestamp) of the corresponding image
   * @param[in]  nrImus  The number of IMU measurements between the prev frame and now
   */
  void processImage(const image_t& image, const std_msgs::Header& header, int nrImuMeasurements);

  /**
   * @brief      Provides the (yet-to-be-corrected) pose estimate at time k,
   *             calculated via IMU propagation.
   *
   * @param[in]  P     Position at time k        (P_WB)
   * @param[in]  Q     Orientation at time k     (Q_WB)
   * @param[in]  V     Linear velocity at time k (V_WB)
   */
  void setCurrentStateFromImuPropagation(const Eigen::Vector3d& P,
                                         const Eigen::Quaterniond& Q,
                                         const Eigen::Vector3d& V,
                                         const Eigen::Vector3d& a,
                                         const Eigen::Vector3d& Ba);

private:
  // ROS stuff
  ros::NodeHandle nh_;
  ros::Publisher pub_horizon_;

  Estimator& estimator_; ///< Reference to vins estimator object

  bool visualize_ = true;

  typedef enum { IMU, GT } horizon_generation_t;
  horizon_generation_t horizonGeneration_ = GT;

  // state vector type definitions
  enum : int { xPOS = 0, xVEL = 3, xB_A = 6, xSIZE = 9 };
  using xVector = Eigen::Matrix<double, xSIZE, 1>;
  using xhVector = Eigen::Matrix<double, xSIZE*(HORIZON + 1), 1>;

  // state vector of current frame estimated from IMU propagation (yet-to-be-corrected)
  xVector xk_;
  Eigen::Quaterniond Qk_[HORIZON+1];
  Eigen::Vector3d ak_;

  xhVector generateFutureHorizon(int nrImuMeasurements, double deltaImu);
  xhVector horizonImu(int nrImuMeasurements, double deltaImu);
  xhVector horizonGroundTruth();
  void visualizeFutureHorizon(const std_msgs::Header& header, const xhVector& x_kkH);
  std::vector<Eigen::MatrixXd> calcInfoFromFeatures(const image_t& image);
  Eigen::MatrixXd calcInfoFromRobotMotion();
};
