#pragma once

#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include <ros/ros.h>
#include <std_msgs/Header.h>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <camodocal/camera_models/Camera.h>
#include <camodocal/camera_models/CameraFactory.h>

#include "nanoflann.hpp"

#include "utility/utility.h"
#include "utility/state_defs.h"
#include "utility/horizon_generator.h"

#include "feature_manager.h"
#include "estimator.h"

class FeatureSelector
{
public:
  FeatureSelector(ros::NodeHandle nh, Estimator& estimator, const std::string& calib_file);
  ~FeatureSelector() = default;


  void setParameters(double accVar, double accBiasVar, bool enable, int maxFeatures);

  /**
   * @brief         Select the most informative subset of features to track
   *
   * @param[inout]  image   A set of calibrated pixels, indexed by feature id
   * @param[in]     header  The header (with timestamp) of the corresponding image
   * @param[in]     nrImus  The number of IMU measurements between the prev frame and now
   * 
   * @return        <historic_ids, new_ids> of features
   */
  std::pair<std::vector<int>, std::vector<int>>
  select(image_t& image, const std_msgs::Header& header, int nrImuMeasurements);

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

  camodocal::CameraPtr m_camera_; ///< geometric camera model

  Estimator& estimator_; ///< Reference to vins estimator object

  /**
   * @brief This is the largest feature_id from the previous frame.
   *        In other words, every feature_id that is larger than
   *        this id is considered a new feature to be selected.
   */
  int lastFeatureId_ = 0;

  // feature ids that have been selected and passed to the backend
  std::vector<int> trackedFeatures_;

  bool firstImage_ = true; ///< All image features are added on first image

  bool enable_ = true; ///< should the feature selection algo be used?
  int maxFeatures_; ///< how many features should be maintained?

  // extrinsic parameters: camera frame w.r.t imu frame
  Eigen::Quaterniond q_IC_;
  Eigen::Vector3d t_IC_;

  // IMU parameters. TODO: Check if these are/should be discrete
  double accVarDTime_ = 0.01;
  double accBiasVarDTime_ = 0.0001;

  // which metric to use
  typedef enum { LOGDET, MINEIG } metric_t;
  metric_t metric_ = LOGDET;

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
   * @brief Dataset adapter for nanoflann
   */
  struct PointCloud
  {
    PointCloud(const std::vector<std::pair<double,double>>& dataset) : pts(dataset) {};
    const std::vector<std::pair<double,double>>& pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return pts[idx].first;
        else /*if (dim == 1)*/ return pts[idx].second;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

  };
  // nanoflann kdtree for guessing depth from existing landmarks
  typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 2/*dim*/> my_kd_tree_t;
  std::unique_ptr<my_kd_tree_t> kdtree_;

  /**
   * @brief      Split features on the provided key
   *
   * @param[in]  k          feature_id to split on (k will not be in image_new)
   * @param      image      Feature set to split
   * @param      image_new  Any features with id after k
   */
  void splitOnFeatureId(int k, image_t& image, image_t& image_new);

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
   * @brief      Calculate the expected info gain from selection of the l'th feature
   *
   * @param[in]  image              Feature data in this image
   * @param[in]  state_kkH          States over horizon
   *
   * @return     delta_ells (information matrix for each feature in image)
   */

  std::map<int, omega_horizon_t> calcInfoFromFeatures(
                              const image_t& image, const state_horizon_t& state_kkH);

  /**
   * @brief      Check if the pixel location is within the field of view
   *
   * @param[in]  p     The 2-vector with pixels (u,v)
   *
   * @return     True if the pixels are within the FOV
   */
  bool inFOV(const Eigen::Vector2d& p);

  /**
   * @brief      Initialize the nanoflann kd tree used to guess
   *             depths of new features based on their neighbors.
   *
   * @return     A vector of depths (of PGO features) to be used
   *             return the depth once a neighbor has been found
   *             (see findNNDepth).
   */
  std::vector<double> initKDTree();

  /**
   * @brief      Guess the depth of a new feature bearing vector by
   *             finding the nearest neighbor in the point cloud
   *             currently maintained by VINS-Mono and using its depth.
   *
   * @param[in]  depths  Depths of VINS-Mono features returned by iniKDTree
   * @param[in]  x       the normalized image plane x direction (calib pixel)
   * @param[in]  y       the normalized image plane y direction (calib pixel)
   *
   * @return     the guessed depth of the new feature
   */
  double findNNDepth(const std::vector<double>& depths, double x, double y);

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

  /**
   * @brief      Run lazy and greedy selection of features
   *
   * @param[inout] subset           Subset of features to add our selection to
   * @param[in]    image            All of the features from the current image
   * @param[in]    kappa            The number of features to try and select
   * @param[in]    Omega_kkH        Information from robot motion
   * @param[in]    Delta_ells       New features and their corresponding info
   * @param[in]    Delta_used_ells  Currently tracked features and their info
   * 
   * @return       Feature ids of new features that were selected
   */
  std::vector<int> selectInformativeFeatures(image_t& subset,
        const image_t& image, int kappa, const omega_horizon_t& Omega_kkH,
        const std::map<int, omega_horizon_t>& Delta_ells,
        const std::map<int, omega_horizon_t>& Delta_used_ells);

  /**
   * @brief      Calculate and sort upper bounds of logDet metric when adding
   *             each feature to the current subset independently of the other
   *
   * @param[in]  Omega       Information from robot motion and current tracks
   * @param[in]  OmegaS      The information content of the current subset
   * @param[in]  Delta_ells  New features and their corresponding information
   * @param[in]  blacklist   Feature ids that have already been selected
   * @param[in]  image       All of the features from the current image
   *
   * @return     A (desc) sorted map of <UB, feature id>
   */
  std::map<double, int, std::greater<double>> sortedlogDetUB(
      const omega_horizon_t& Omega, const omega_horizon_t& OmegaS,
      const std::map<int, omega_horizon_t>& Delta_ells,
      const std::vector<int>& blacklist, const image_t& image);
};
