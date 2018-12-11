#include "feature_selector.h"

FeatureSelector::FeatureSelector(ros::NodeHandle nh, Estimator& estimator, 
                                 const std::string& calib_file)
: nh_(nh), estimator_(estimator)
{

  // create future state horizon generator / manager
  hgen_ = std::unique_ptr<HorizonGenerator>(new HorizonGenerator(nh_));

  // set parameters for the horizon generator (extrinsics -- for visualization)
  hgen_->setParameters(estimator_.ric[0], estimator_.tic[0]);

  // save extrinsics (for feature info step)
  q_IC_ = estimator_.ric[0];
  t_IC_ = estimator_.tic[0];

  // create camera model from calibration YAML file
  m_camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

// ----------------------------------------------------------------------------

void FeatureSelector::setParameters(double accVar, double accBiasVar)
{
  accVarDTime_ = accVar;
  accBiasVarDTime_ = accBiasVar;
}

// ----------------------------------------------------------------------------

void FeatureSelector::setNextStateFromImuPropagation(
    double imageTimestamp,
    const Eigen::Vector3d& P, const Eigen::Quaterniond& Q,
    const Eigen::Vector3d& V, const Eigen::Vector3d& a,
    const Eigen::Vector3d& w, const Eigen::Vector3d& Ba)
{
  //
  // State of previous frame (at the end of the fixed-lag window)
  //

  // TODO: Why is this not WINDOW_SIZE-1?
  state_k_.first.coeffRef(xTIMESTAMP) = state_k1_.first.coeff(xTIMESTAMP);
  state_k_.first.segment<3>(xPOS) = estimator_.Ps[WINDOW_SIZE];
  state_k_.first.segment<3>(xVEL) = estimator_.Vs[WINDOW_SIZE];
  state_k_.first.segment<3>(xB_A) = estimator_.Bas[WINDOW_SIZE];
  state_k_.second = estimator_.Rs[WINDOW_SIZE];


  //
  // (yet-to-be-corrected) state of current frame
  //

  // set the propagated-forward state of the current frame
  state_k1_.first.coeffRef(xTIMESTAMP) = imageTimestamp;
  state_k1_.first.segment<3>(xPOS) = P;
  state_k1_.first.segment<3>(xVEL) = V;
  state_k1_.first.segment<3>(xB_A) = Ba;
  state_k1_.second = Q;

  // the latest IMU measurement
  ak1_ = a;
  wk1_ = w;
}

// ----------------------------------------------------------------------------

void FeatureSelector::select(image_t& image, int kappa,
        const std_msgs::Header& header, int nrImuMeasurements)
{
  //
  // Timing information
  //

  // frame time of previous image
  static double frameTime_k = header.stamp.toSec();

  // time difference between last frame and current frame
  double deltaF = header.stamp.toSec() - frameTime_k;

  // calculate the IMU sampling rate of the last frame-to-frame meas set
  double deltaImu = deltaF / nrImuMeasurements;


  //
  // Future State Generation
  //

  // We will need to know the state at each frame in the horizon, k:k+H.
  // Note that this includes the current optimized state, xk
  auto state_kkH = generateFutureHorizon(header, nrImuMeasurements, deltaImu, deltaF);
  hgen_->visualize(header, state_kkH);


  //
  // Anticipation: Compute the Expected Information over the Horizon
  //

  // Calculate the information content from motion over the horizon (eq 15)
  auto OmegaIMU_kkH = calcInfoFromRobotMotion(state_kkH, nrImuMeasurements, deltaImu);

  // Add in prior information (eq 16)
  auto Omega_kkH = addOmegaPrior(OmegaIMU_kkH);

  // Calculate the information content of each of the new features
  auto Delta_ells = calcInfoFromFeatures(image, state_kkH);

  // Calculate the information content of each of the currently used features
  std::map<int, omega_horizon_t> Delta_used_ells;
  // auto Delta_used_ells = calcInfoFromFeatures(image, state_kkH);


  //
  // Attention: Select a subset of features that maximizes expected information
  //

  // Only change the feature information if VINS-Mono is initialized
  if (estimator_.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
    // removes poor feature choices from image
    keepInformativeFeatures(image, kappa, Omega_kkH, Delta_ells, Delta_used_ells);
  }

  ROS_WARN_STREAM("Feature selector chose " << image.size() << " features");

  // for next iteration
  frameTime_k = header.stamp.toSec();
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

state_horizon_t FeatureSelector::generateFutureHorizon(
                                        const std_msgs::Header& header,
                                        int nrImuMeasurements,
                                        double deltaImu, double deltaFrame)
{

  // generate the horizon based on the requested scheme
  if (horizonGeneration_ == IMU) {
    return hgen_->imu(state_k_, state_k1_, ak1_, wk1_, nrImuMeasurements, deltaImu);
  } else { //if (horizonGeneration_ == GT) {
    return hgen_->groundTruth(state_k_, state_k1_, deltaFrame);
  }

}

// ----------------------------------------------------------------------------
std::map<int, omega_horizon_t> FeatureSelector::calcInfoFromFeatures(
                                            const image_t& image,
                                            const state_horizon_t& state_kkH)
{
  //initialize the return map, Delta_ells
  std::map<int, omega_horizon_t> Delta_ells;

  // convenience: (yet-to-be-corrected) transformation
  // of camera frame w.r.t world frame at time k+1
  const auto& t_WC_k1 = state_k1_.first.segment<3>(xPOS) + state_k1_.second * t_IC_;
  const auto& q_WC_k1 = state_k1_.second * q_IC_;

  auto depthsByIdx = initKDTree();

  for (const auto& fpair : image) {

    // there is only one camera, so we expect only one vector per feature
    constexpr int c = 0;

    // extract feature id and nip vector from obnoxious data structure
    int feature_id = fpair.first;
    Eigen::Vector3d feature = fpair.second[c].second.head<3>(); // calibrated [u v 1]

    // scale bearing vector by depth
    double d = findNNDepth(depthsByIdx, feature.coeff(0), feature.coeff(1));
    feature = feature.normalized() * d;

    // Estimated position of the landmark w.r.t the world frame
    auto pell = t_WC_k1 + q_WC_k1 * feature;

    //
    // Forward-simulate the feature bearing vector over the horizon
    //
    
    // keep count of how many camera poses could see this landmark.
    // We know it's visible in at least the current (k+1) frame.
    int numVisible = 1;

    // for storing the necessary blocks for the Delta_ell information matrix
    // NOTE: We delay computation for h=k+1 until absolutely necessary.
    Eigen::Matrix<double, 3, 3*HORIZON> Ch; // Ch == BtB_h in report
    Ch.setZero();

    // Also sum up the Ch blocks for EtE;
    Eigen::Matrix3d EtE = Eigen::Matrix3d::Zero();

    // NOTE: start forward-simulating the landmark projection from k+2
    // (we have the frame k+1 projection, since that's where it came from)
    for (int h=2; h<=HORIZON; ++h) { 

      // convenience: future camera frame (k+h) w.r.t world frame
      const auto& t_WC_h = state_kkH[h].first.segment<3>(xPOS) + state_kkH[h].second * t_IC_;
      const auto& q_WC_h = state_kkH[h].second * q_IC_;

      // create bearing vector of feature w.r.t camera pose h
      Eigen::Vector3d uell = (q_WC_h.inverse() * (pell - t_WC_h)).normalized();

      // TODO: Maybe flip the problem so we don't have to do this every looped-loop
      // project to pixels so we can perform visibility check
      Eigen::Vector2d pixels;
      m_camera_->spaceToPlane(uell, pixels);

      // If not visible from this pose, skip
      if (!inFOV(pixels)) continue;

      // Calculate sub-block of Delta_ell (zero-indexing)
      Eigen::Matrix3d Bh = Utility::skewSymmetric(uell)*((q_WC_h*q_IC_).inverse()).toRotationMatrix();
      Ch.block<3, 3>(0, 3*(h-1)) = Bh.transpose()*Bh;

      // Sum up block for EtE
      EtE += Ch.block<3, 3>(0, 3*(h-1));

      ++numVisible;
    }

    // If we don't expect to be able to triangulate a point
    // then it is not useful. By not putting this feature in
    // the output map, we are effectively getting rid of it now.
    if (numVisible == 1) continue;

    // Since the feature can be triangulated, we now do the computation that
    // we put off before forward-simulating the landmark projection:
    // we calculate Ch for h=k+1 (the frame where the feature was detected)
    Eigen::Matrix3d Bh = Utility::skewSymmetric(feature.normalized())
                                                *((q_WC_k1*q_IC_).inverse()).toRotationMatrix();
    Ch.block<3, 3>(0, 0) = Bh.transpose()*Bh;

    // add information to EtE
    EtE += Ch.block<3, 3>(0, 0);

    // Compute landmark covariance (should be invertible)
    Eigen::Matrix3d W = EtE.inverse();

    //
    // Build Delta_ell for this Feature (see support_files/report)
    //

    omega_horizon_t Delta_ell = omega_horizon_t::Zero();

    // col-wise for efficiency
    for (int j=1; j<=HORIZON; ++j) {
      // for convenience
      Eigen::Ref<Eigen::Matrix3d> Cj = Ch.block<3, 3>(0, 3*(j-1));

      for (int i=j; i<=HORIZON; ++i) { // NOTE: i=j for lower triangle
        // for convenience
        Eigen::Ref<Eigen::Matrix3d> Ci = Ch.block<3, 3>(0, 3*(i-1));
        Eigen::Matrix3d Dij = Ci*W*Cj.transpose();

        if (i == j) {
          // diagonal
          Delta_ell.block<3, 3>(9*i, 9*j) = Ci - Dij;
        } else {
          // lower triangle
          Delta_ell.block<3, 3>(9*i, 9*j) = -Dij;

          // upper triangle
          Delta_ell.block<3, 3>(9*j, 9*i) = -Dij.transpose();
        }

      }
    }

    // Store this information matrix with its associated feature ID
    Delta_ells[feature_id] = Delta_ell;
  }
  return Delta_ells;
}

// ----------------------------------------------------------------------------

bool FeatureSelector::inFOV(const Eigen::Vector2d& p)
{
  constexpr int border = 0; // TODO: Could be good to have a border here
  int u = std::round(p.coeff(0));
  int v = std::round(p.coeff(1));
  return (border <= u && u < m_camera_->imageWidth() - border) && 
         (border <= v && v < m_camera_->imageHeight() - border);
}

// ----------------------------------------------------------------------------

std::vector<double> FeatureSelector::initKDTree()
{
  // setup dataset
  static std::vector<std::pair<double, double>> dataset;
  dataset.clear(); dataset.reserve(estimator_.f_manager.feature.size());

  //
  // Build the point cloud of bearing vectors w.r.t camera frame k+1
  //
  
  // we want a vector of depths that match the ordering of the dataset
  // for lookup after the knn have been found
  std::vector<double> depths;
  depths.reserve(estimator_.f_manager.feature.size());

  // copied from visualization.cpp, pubPointCloud
  for (const auto& it_per_id : estimator_.f_manager.feature) {

    // ignore features if they haven't been around for a while or they're not stable
    int used_num = it_per_id.feature_per_frame.size();
    if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) continue;
    if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1) continue;

    // TODO: Why 0th frame?
    int imu_i = it_per_id.start_frame;
    Eigen::Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
    Eigen::Vector3d w_pts_i = estimator_.Rs[imu_i] * (estimator_.ric[0] * pts_i + estimator_.tic[0]) + estimator_.Ps[imu_i];

    // w_pts_i is the position of the landmark w.r.t. the world
    // transform it so that it is the pos of the landmark w.r.t camera frame at x_k+1
    Eigen::Vector3d p_IL_k1 = state_k1_.second.inverse() * (w_pts_i - state_k1_.first.segment<3>(xPOS));
    Eigen::Vector3d p_CL_k1 = q_IC_.inverse() * (p_IL_k1 - t_IC_);
    
    // project back to nip of the camera at time k+1
    w_pts_i = p_CL_k1 / p_CL_k1.coeff(2);
    double x = w_pts_i.coeff(0), y = w_pts_i.coeff(1);

    dataset.push_back(std::make_pair(x, y));
    depths.push_back(it_per_id.estimated_depth);
  }

  // point cloud adapter for currently tracked landmarks in PGO
  // keep as static because kdtree uses a reference to the cloud.
  // Note that this works because dataset is also static
  static PointCloud cloud(dataset);

  // create the kd-tree and index the data
  kdtree_.reset(new my_kd_tree_t(2/* dim */, cloud,
                nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
  kdtree_->buildIndex();

  // these are depths of PGO features by index of the dataset
  return depths;
}

// ----------------------------------------------------------------------------

double FeatureSelector::findNNDepth(const std::vector<double>& depths, 
                                    double x, double y)
{
  // The point cloud and the query are expected to be in the normalized image
  // plane (nip) of the camera at time k+1 (the frame the feature was detected in)

  // If this happens, then the back end is initializing
  if (depths.size() == 0) return 1.0;

  // build query
  double query_pt[2] = { x, y };

  // do a knn search
  // TODO: Considering avg multiple neighbors?
  const size_t num_results = 1;
  size_t ret_index = 0;
  double out_dist_sqr;
  nanoflann::KNNResultSet<double> resultSet(num_results);
  resultSet.init(&ret_index, &out_dist_sqr);
  kdtree_->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

  return depths[ret_index];
}

// ----------------------------------------------------------------------------

omega_horizon_t FeatureSelector::calcInfoFromRobotMotion(
                                    const state_horizon_t& state_kkH,
                                    double nrImuMeasurements, double deltaImu)
{
  // ** Build the large information matrix over the horizon (eq 15).
  //
  // There is a sparse structure to the information matrix that we can exploit.
  // We can calculate the horizon info. matrix in blocks. Notice that each
  // pair of consecutive frames in the horizon create four 9x9 sub-blocks.
  // For example, for a horizon of H=3, the first pair of images (h=1) creates
  // a large information matrix like the following (each block is 9x9):
  //
  //         |------------------------------------
  //         | At*Ω*A |  At*Ω  |    0   |    0   |
  //         |------------------------------------
  //         |   Ω*A  |    Ω   |    0   |    0   |
  //         |------------------------------------
  //         |    0   |    0   |    0   |    0   |
  //         |------------------------------------
  //         |    0   |    0   |    0   |    0   |
  //         |------------------------------------
  //
  // The four non-zero sub-blocks shift along the diagonal as we loop through
  // the horizon (i.e., for h=2 there are zeros on all the edges and for h=3
  // the Ω is in the bottom-right corner). Note that the Ai matrix must be
  // recomputed for each h. The Ω matrix is calculated as the inverse of
  // the covariance in equation (52) and characterizes the noise in a
  // preintegrated set of IMU measurements using the linear IMU model.

  // NOTE: We are even more clever and only calculate the upper-triangular
  // and then transpose since this is a symmetric PSD matrix

  omega_horizon_t Omega_kkH = omega_horizon_t::Zero();

  for (int h=1; h<=HORIZON; ++h) { // for consecutive frames in horizon

    // convenience: frames (i, j) are a consecutive pair in horizon
    const auto& Qi = state_kkH[h-1].second;
    const auto& Qj = state_kkH[h].second;

    // Create Ablk and Ω as explained in the appendix
    auto mats = createLinearImuMatrices(Qi, Qj, nrImuMeasurements, deltaImu);

    // convenience: select sub-blocks to add to, based on h
    Eigen::Ref<Eigen::MatrixXd> block1 = Omega_kkH.block<STATE_SIZE, STATE_SIZE>((h-1)*STATE_SIZE, (h-1)*STATE_SIZE);
    Eigen::Ref<Eigen::MatrixXd> block2 = Omega_kkH.block<STATE_SIZE, STATE_SIZE>((h-1)*STATE_SIZE, h*STATE_SIZE);
    Eigen::Ref<Eigen::MatrixXd> block3 = Omega_kkH.block<STATE_SIZE, STATE_SIZE>(h*STATE_SIZE, (h-1)*STATE_SIZE);
    Eigen::Ref<Eigen::MatrixXd> block4 = Omega_kkH.block<STATE_SIZE, STATE_SIZE>(h*STATE_SIZE, h*STATE_SIZE);

    // At*Ω*A (top-left sub-block)
    block1 += mats.second.transpose()*mats.first*mats.second;

    // At*Ω (top-right sub-block)
    auto tmp = mats.second.transpose()*mats.first;
    block2 += tmp;

    // Ω*A (bottom-left sub-block)
    block3 += tmp.transpose();

    // Ω (bottom-right sub-block)
    block4 += mats.first;
  }

  return Omega_kkH;
}

// ----------------------------------------------------------------------------

std::pair<omega_t, ablk_t> FeatureSelector::createLinearImuMatrices(
      const Eigen::Quaterniond& Qi, const Eigen::Quaterniond& Qj,
      double nrImuMeasurements, double deltaImu)
{
  //
  // "Pre-integrate" future IMU measurements over horizon
  //

  // helper matrices, equations (47) and (48)
  Eigen::Matrix3d Nij = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d Mij = Eigen::Matrix3d::Zero();

  // initialize block coefficients
  double CCt_11 = 0;
  double CCt_12 = 0;

  // This is an IMU-rate for loop
  for (int i=0; i<nrImuMeasurements; ++i) {

    // slerp from Qi toward Qj by where we are in between the frames
    // (this will never slerp all the way to Qj)
    auto q = Qi.slerp(i/static_cast<double>(nrImuMeasurements), Qj);

    // so many indices...
    double jkh = (nrImuMeasurements - i - 0.5);
    Nij += jkh * q.toRotationMatrix();
    Mij += q.toRotationMatrix();

    // entries of CCt
    CCt_11 += jkh*jkh;
    CCt_12 += jkh;
  }

  // powers of IMU sampling period
  const double deltaImu_2 = deltaImu*deltaImu;
  const double deltaImu_3 = deltaImu_2*deltaImu;
  const double deltaImu_4 = deltaImu_3*deltaImu;

  //
  // Build cov(eta^imu_ij) -- see equation (52)
  //

  // NOTE: In paper, bottom right entry of CCt should have (j-k), not (j-k-1).
  omega_t covImu = omega_t::Zero();
  covImu.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity()
          * nrImuMeasurements * CCt_11 * deltaImu_4 * accVarDTime_;
  covImu.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity()
          * CCt_12 * deltaImu_3 * accVarDTime_;
  covImu.block<3, 3>(3, 0) = covImu.block<3, 3>(0, 3).transpose();
  covImu.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity()
          * nrImuMeasurements * deltaImu_2 * accVarDTime_;
  covImu.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity()
          * nrImuMeasurements * accBiasVarDTime_;

  //
  // Build Ablk -- see equation (50)
  //

  Nij *= deltaImu_2;
  Mij *= deltaImu;

  ablk_t Ablk = -ablk_t::Identity();
  Ablk.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity() * nrImuMeasurements*deltaImu;
  Ablk.block<3, 3>(0, 6) = Nij;
  Ablk.block<3, 3>(3, 6) = Mij;

  return std::make_pair(covImu.inverse(), Ablk);
}

// ----------------------------------------------------------------------------

omega_horizon_t FeatureSelector::addOmegaPrior(const omega_horizon_t& OmegaIMU)
{
  return OmegaIMU;
}

// ----------------------------------------------------------------------------

void FeatureSelector::keepInformativeFeatures(image_t& image, int kappa,
                        const omega_horizon_t& Omega_kkH,
                        const std::map<int, omega_horizon_t>& Delta_ells,
                        const std::map<int, omega_horizon_t>& Delta_used_ells)
{
  // Combine motion information with information from features that are already
  // being used in the VINS-Mono optimization backend
  omega_horizon_t Omega = Omega_kkH;
  // for (const auto& Delta : Delta_used_ells) {
  //   Omega += Delta.second;
  // }

  // subset of most informative features
  image_t subset;

  // blacklist of already selected features (by id)
  std::vector<int> blacklist;
  blacklist.reserve(kappa);

  // combined information of subset
  omega_horizon_t OmegaS = omega_horizon_t::Zero();

  // select the indices of the best features
  for (int i=0; i<kappa; ++i) {

    // compute upper bounds in form of <UB, featureId> descending by UB
    auto upperBounds = sortedlogDetUB(Omega, OmegaS, Delta_ells, blacklist, image);

    // initialize the best cost function value and feature ID to worst case
    double fMax = -1.0;
    int lMax = -1;

    // iterating through upperBounds in descending order, check each feature
    for (const auto& fpair : upperBounds) {
      int feature_id = fpair.second;
      double ub = fpair.first;

      // lazy evaluation: break if UB is less than the current best cost
      if (ub < fMax) break;
 
      // convenience: the information matrix corresponding to this feature
      const auto& Delta_ell = Delta_ells.at(feature_id);

      // find probability of this feature being tracked
      double p = 1.0; // image.at(feature_id)[0].second.coeff(??);

      // calculate logdet efficiently (with regulizer to ensure PSD)
      omega_horizon_t reg = 1.0*omega_horizon_t::Identity();
      double fValue = Utility::logdet(Omega + OmegaS + p*Delta_ell + reg, true);

      // nan check
      if (std::isnan(fValue)) ROS_ERROR_STREAM("nan! increase strength of regularizer.");

      // store this feature/reward if better than before
      if (fValue > fMax) {
        fMax = fValue;
        lMax = feature_id;
      }
    }

    // if lMax == -1 there was likely a nan (probably because roundoff error
    // caused det(M) < 0). I guess there just won't be a feature this iter.
    if (lMax > -1) {
      // Accumulate combined feature information in subset
      double p = 1.0; // image.at(lMax)[0].second.coeff(??);
      OmegaS += p*Delta_ells.at(lMax);

      // add feature that returns the most information to the subset
      subset[lMax] = image.at(lMax);
    }
  }

  // return the subset to estimator_node.cpp
  subset.swap(image);
}

// ----------------------------------------------------------------------------

std::map<double, int, std::greater<double>> FeatureSelector::sortedlogDetUB(
  const omega_horizon_t& Omega, const omega_horizon_t& OmegaS,
  const std::map<int, omega_horizon_t>& Delta_ells,
  const std::vector<int>& blacklist, const image_t& image)
{
  // returns a descending sorted map with upper bound as the first key,
  // and feature id as the value for all features in image
  std::map<double, int, std::greater<double>> UBs;

  // Partially create argument to UB function (see eq 9). The only thing
  // missing from this is the additive and expected information from the
  // l-th feature. Each is added one at a time (independently) to calc UB
  const omega_horizon_t M = Omega + OmegaS;

  // Find the upper bound of adding each Delta_ell to M independently
  for (const auto& fpair : Delta_ells) {
    int feature_id = fpair.first;

    // if a feature was already selected, do not calc UB. Not including it
    // in the UBs prevents it from being selected again.
    bool in_blacklist = std::find(blacklist.begin(), blacklist.end(),
                                  feature_id) != blacklist.end();
    if (in_blacklist) continue;

    // find probability of this feature being tracked
    double p = 1.0; // image.at(feature_id)[0].second.coeff(??);

    // construct the argument to the logdetUB function
    omega_horizon_t A = M + p*Delta_ells.at(feature_id);

    // calculate upper bound (eq 29)
    double ub = A.diagonal().array().log().sum();

    // store in map for automatic sorting (desc) and easy lookup
    UBs[ub] = feature_id;
  }

  return UBs;
}

// ----------------------------------------------------------------------------
