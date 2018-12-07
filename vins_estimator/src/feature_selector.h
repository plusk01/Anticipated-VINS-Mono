#pragma once

#include <map>
#include <utility>
#include <iostream>
#include <vector>
#include <memory>

#include <ros/ros.h>
#include <std_msgs/Header.h>

#include <Eigen/Dense>

#include "utility/state_defs.h"
#include "utility/horizon_generator.h"

#include "estimator.h"

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
   * @param[in]  imuT  Pose time, stamped with IMU clock
   * @param[in]  imgT  Pose time, stamped with image clock
   * @param[in]  P     Position at time k        (P_WB)
   * @param[in]  Q     Orientation at time k     (Q_WB)
   * @param[in]  V     Linear velocity at time k (V_WB)
   * @param[in]  a     Linear accel at time k    (a_WB)
   * @param[in]  Ba    Accel bias at time k      (in sensor frame)
   */
  void setCurrentStateFromImuPropagation(
          double imuTimestamp, double imageTimestamp,
          const Eigen::Vector3d& P, const Eigen::Quaterniond& Q,
          const Eigen::Vector3d& V, const Eigen::Vector3d& a,
          const Eigen::Vector3d& w, const Eigen::Vector3d& Ba);

private:
  // ROS stuff
  ros::NodeHandle nh_;

  Estimator& estimator_; ///< Reference to vins estimator object

  bool visualize_ = true;

  // state generator over the future horizon
  typedef enum { IMU, GT } horizon_generation_t;
  horizon_generation_t horizonGeneration_ = IMU;
  std::unique_ptr<HorizonGenerator> hgen_;


  state_t state_0_; ///< state of last frame (from backend)
  state_t state_k_; ///< state of current frame (from IMU prop)
  Eigen::Vector3d ak_; ///< latest accel measurement (from IMU)
  Eigen::Vector3d wk_; ///< latest ang. vel. measurement (from IMU)

  state_horizon_t generateFutureHorizon(const std_msgs::Header& header, int nrImuMeasurements,
                                                    double deltaImu, double deltaFrame);
  std::vector<Eigen::MatrixXd> calcInfoFromFeatures(const image_t& image);
  Eigen::MatrixXd calcInfoFromRobotMotion();
};
