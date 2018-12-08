#pragma once

#include <map>
#include <utility>
#include <iostream>
#include <vector>
#include <memory>

#include <ros/ros.h>
#include <std_msgs/Header.h>

#include <Eigen/Dense>
#include <Eigen/SVD>

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


  void setParameters(double accVar, double accBiasVar);

  /**
   * @brief      Process features to be selected
   *
   * @param[in]  image   A set of calibrated pixels, indexed by feature id
   * @param[in]  header  The header (with timestamp) of the corresponding image
   * @param[in]  nrImus  The number of IMU measurements between the prev frame and now
   */
  void processImage(const image_t& image, const std_msgs::Header& header, int nrImuMeasurements);

  /**
   * @brief      Provides the (yet-to-be-corrected) pose estimate
   *             for frame k+1, calculated via IMU propagation.
   *
   * @param[in]  imgT  Image clock stamp of frame k+1
   * @param[in]  P     Position at time k+1        (P_WB)
   * @param[in]  Q     Orientation at time k+1     (Q_WB)
   * @param[in]  V     Linear velocity at time k+1 (V_WB)
   * @param[in]  a     Linear accel at time k+1    (a_WB)
   * @param[in]  w     Angular vel at time k+1     (w_WB)
   * @param[in]  Ba    Accel bias at time k+1      (in sensor frame)
   */
  void setNextStateFromImuPropagation(
          double imageTimestamp,
          const Eigen::Vector3d& P, const Eigen::Quaterniond& Q,
          const Eigen::Vector3d& V, const Eigen::Vector3d& a,
          const Eigen::Vector3d& w, const Eigen::Vector3d& Ba);

private:
  // ROS stuff
  ros::NodeHandle nh_;

  Estimator& estimator_; ///< Reference to vins estimator object

  bool visualize_ = true;

  // IMU parameters. TODO: Check if these are/should be discrete
  double accVarDTime_ = 0.01;
  double accBiasVarDTime_ = 0.0001;

  // state generator over the future horizon
  typedef enum { IMU, GT } horizon_generation_t;
  horizon_generation_t horizonGeneration_ = GT;
  std::unique_ptr<HorizonGenerator> hgen_;

  // state
  state_t state_k_;     ///< state of last frame, k (from backend)
  state_t state_k1_;    ///< state of current frame, k+1 (from IMU prop)
  Eigen::Vector3d ak1_; ///< latest accel measurement, k+1 (from IMU)
  Eigen::Vector3d wk1_; ///< latest ang. vel. measurement, k+1 (from IMU)

  /**
   * @brief      Generate a future state horizon from k+1 to k+H
   *
   * @param[in]  header             The header of the current image frame (k+1)
   * @param[in]  nrImuMeasurements  Num IMU measurements from prev to current frame
   * @param[in]  deltaImu           Sampling period of IMU
   * @param[in]  deltaFrame         Sampling period of image frames
   *
   * @return     a state horizon that includes the current state, [xk xk+1:k+H]
   */
  state_horizon_t generateFutureHorizon(const std_msgs::Header& header, int nrImuMeasurements,
                                                    double deltaImu, double deltaFrame);
  
  /**
   * @brief      Calculate the expected info gain from robot motion
   *
   * @param[in]  x_kkH              The horizon (rotations are needed)
   * @param[in]  nrImuMeasurements  Num IMU measurements between frames
   * @param[in]  deltaImu           Sampling period of IMU
   *
   * @return     Returns OmegaIMU (bar) from equation (15)
   */
  omega_horizon_t calcInfoFromRobotMotion(const state_horizon_t& x_kkH,
                  double nrImuMeasurements, double deltaImu);

  /**
   * @brief      Create Ablk and OmegaIMU (no bar, eq 15) using pairs
   *             of consecutive frames in the horizon.
   *
   * @param[in]  Qi                 Orientation at frame i
   * @param[in]  Qj                 Orientation at frame j
   * @param[in]  nrImuMeasurements  Num IMU measurements between frames
   * @param[in]  deltaImu           Sampling period of IMU
   *
   * @return     OmegaIMU (no bar) and Ablk for given frame pair
   */
  std::pair<omega_t, ablk_t> createLinearImuMatrices(
      const Eigen::Quaterniond& Qi, const Eigen::Quaterniond& Qj,
      double nrImuMeasurements, double deltaImu);

  omega_horizon_t addOmegaPrior(const omega_horizon_t& OmegaIMU);
  
  std::vector<Eigen::MatrixXd> calcInfoFromFeatures(const image_t& image);

};
