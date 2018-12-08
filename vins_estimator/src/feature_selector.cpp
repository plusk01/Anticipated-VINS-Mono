#include "feature_selector.h"

FeatureSelector::FeatureSelector(ros::NodeHandle nh, Estimator& estimator)
: nh_(nh), estimator_(estimator)
{

  // create future state horizon generator / manager
  hgen_ = std::unique_ptr<HorizonGenerator>(new HorizonGenerator(nh_));

  // set parameters for the horizon generator (extrinsics -- for visualization)
  hgen_->setParameters(estimator_.ric[0], estimator_.tic[0]);
}

// ----------------------------------------------------------------------------

void FeatureSelector::setParameters(double accVar, double accBiasVar)
{
  accVarDTime_ = accVar;
  accBiasVarDTime_ = accBiasVar;
}

// ----------------------------------------------------------------------------

void FeatureSelector::setCurrentStateFromImuPropagation(
    double imuTimestamp, double imageTimestamp,
    const Eigen::Vector3d& P, const Eigen::Quaterniond& Q,
    const Eigen::Vector3d& V, const Eigen::Vector3d& a,
    const Eigen::Vector3d& w, const Eigen::Vector3d& Ba)
{
  //
  // State of previous frame
  //

  state_0_.first.coeffRef(xTIMESTAMP) = state_k_.first.coeff(xTIMESTAMP);
  state_0_.first.segment<3>(xPOS) = estimator_.Ps[WINDOW_SIZE];
  state_0_.first.segment<3>(xVEL) = estimator_.Vs[WINDOW_SIZE];
  state_0_.first.segment<3>(xB_A) = estimator_.Bas[WINDOW_SIZE];
  state_0_.second = estimator_.Rs[WINDOW_SIZE];

  //
  // (yet-to-be-corrected) state of current frame
  //

  // set the propagated-forward state of the current frame
  state_k_.first.coeffRef(xTIMESTAMP) = imageTimestamp;
  state_k_.first.segment<3>(xPOS) = P;
  state_k_.first.segment<3>(xVEL) = V;
  state_k_.first.segment<3>(xB_A) = Ba;
  state_k_.second = Q;

  // the last IMU measurement
  ak_ = a;
  wk_ = w;
}

// ----------------------------------------------------------------------------

void FeatureSelector::processImage(const image_t& image,
                                   const std_msgs::Header& header,
                                   int nrImuMeasurements)
{
  //
  // Timing information
  //

  // frame time of previous image
  static double lastFrameTime_ = header.stamp.toSec();

  // time difference between last frame and current frame
  double deltaF = header.stamp.toSec() - lastFrameTime_;

  // calculate the IMU sampling rate of the last frame-to-frame meas set
  double deltaImu = deltaF / nrImuMeasurements;


  //
  // Future State Generation
  //

  // We will need to know the state at each frame in the horizon, k:k+H
  auto state_kkH = generateFutureHorizon(header, nrImuMeasurements, deltaImu, deltaF);

  if (visualize_) {
    hgen_->visualize(header, state_kkH);
  }


  //
  // Anticipation: Compute the Expected Information over the Horizon
  //

  // Calculate the information content from motion over the horizon (eq 15)
  auto OmegaIMU = calcInfoFromRobotMotion(state_kkH, nrImuMeasurements, deltaImu);

  // Add in prior information (eq 16)
  auto Omega = addOmegaPrior(OmegaIMU);

  // Calculate the information content of each of the new features
  Eigen::Vector2i imageDimensions;
  imageDimensions << 640,480;
  Eigen::Matrix3d cameraCalibration;
  cameraCalibration << 1, 1, 1, // placeholder
                       0, 1, 1,
                       0, 0, 1;
  Eigen::Matrix3d RcamIMU = Eigen::Matrix3d::Identity();
  auto Delta_ells = calcInfoFromFeatures(image, state_kkH, imageDimensions, cameraCalibration, RcamIMU);

  // Calculate the information content of each of the currently used features
  auto Delta_used_ells = calcInfoFromFeatures(image, state_kkH, imageDimensions, cameraCalibration, RcamIMU);


  //
  // Attention: Select a subset of features that maximizes expected information
  //

  estimator_.processImage(image, header);

  lastFrameTime_ = header.stamp.toSec();
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
    return hgen_->imu(state_0_, state_k_, ak_, wk_, nrImuMeasurements, deltaImu);
  } else { //if (horizonGeneration_ == GT) {
    return hgen_->groundTruth(state_0_, state_k_, deltaFrame);
  }

}

// ----------------------------------------------------------------------------

std::map<int, Eigen::Matrix<double, 9*(HORIZON+1), 9*(HORIZON+1)>> FeatureSelector::calcInfoFromFeatures(const image_t& image, const state_horizon_t& state_kkH, Eigen::Vector2i imageDimensions, Eigen::Matrix3d cameraCalibration, Eigen::Matrix3d RcamIMU)
{
  // need to estimate depth better
  // need to fix it such that iterating through feature-id's not feature numbers
  std::map<int, Eigen::Matrix<double, 9*(HORIZON+1), 9*(HORIZON+1)>> delta_ells;
  // project the u,v pixel projection of the feature on the frames over the horizon
  // k to to k+H, and check if still visibile in frame. The returned matrix will
  // be formatted to have u in the first column, v in the second, and a visibility
  // check (1 if visible, 0 if not) in the third column
  //        /--------------
  //        | u1 | v1 | 0 |
  //        | u2 | v2 | 1 |
  //        | u3 | v3 | 1 |
  //        |...| ...| ...| (continue for all features in curent frame)
  // Eigen::Ref<Eigen::MatrixXd> projectedPixels = projectFeaturePixelOverHorizon(x_kkH);
  int maxU = imageDimensions[0]; // max u pixel to stay in frame
  int maxV = imageDimensions[1]; // max v pixel to stay in frame
  // Eigen::MatrixXd featureBearingVectors(numFeatures,3);
  std::map<int,Eigen::Vector3d> featureBearingVectors;
  std::map<int,Eigen::Vector3d> worldFeatureBearingVectors;
  // Eigen::MatrixXd worldFeatureBearingVectors(numFeatures,3);
  // for (int f = 0; f<numFeatures; ++f)
  int camera_id = 0; // PLACEHOLDER, pipe in from...?
  for (std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>::const_iterator feature=image.begin();
   feature!=image.end(); ++feature)
  {
    int feature_id = feature->first;
    featureBearingVectors[feature_id] << feature->second[camera_id].second(0,0),
                                   feature->second[camera_id].second(1,0),
                                   feature->second[camera_id].second(2,0);
    // featureBearingVectors[feature_id] << image.at(feature_id)[camera_id].second(0,0),
    //                                image.at(feature_id)[camera_id].second(1,0),
    //                                image.at(feature_id)[camera_id].second(2,0);
    // TODO : check if already normalized or not; should we be normalizing?
    featureBearingVectors[feature_id].normalized();
    // TODO: Make a better depth guess
  }
  for (std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>::const_iterator feature=image.begin();
   feature!=image.end(); ++feature)
  // for (int f = 0; f<numFeatures; ++f)
  {
    int feature_id = feature->first;
    // for each feature, reinitialize pixelProjection variable
    Eigen::MatrixXd pixelProjection(3,HORIZON);
    worldFeatureBearingVectors[feature_id] = featureBearingVectors[feature_id] + state_kkH[0].first.segment<3>(xPOS);
    Eigen::MatrixXd p_lc(3,HORIZON);
    for (int h=0; h<HORIZON; ++h)
    {
      p_lc.col(h) = worldFeatureBearingVectors[feature_id] - state_kkH[h].first.segment<3>(xPOS);
      pixelProjection(0,h) = cameraCalibration(0,0)*(p_lc(0,h)/p_lc(2,h)) + cameraCalibration(0,2);// u pixel projection
      pixelProjection(1,h) = cameraCalibration(1,1)*(p_lc(1,h)/p_lc(2,h)) + cameraCalibration(1,2);// v pixel projection
      // check if projected pixel location is within field of view of the camera
      if (pixelProjection(0,h) > 0 && pixelProjection(1,h) > 0 && pixelProjection(0,h) < maxU && pixelProjection(1,h) < maxV)
      {
        pixelProjection(2,h) = 1;
      }
      else
      {
        pixelProjection(2,h) = 0;
      }
    }
    // initialize F to have as many rows as visibile pixels
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(3*pixelProjection.row(2).nonZeros(),9*(HORIZON+1));
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(3*pixelProjection.row(2).nonZeros(),3);
    int visible_counter = 0;
    for (int h=1; h<=HORIZON; ++h)
    {
      if (pixelProjection(2,h) == 1)
      {
        Eigen::Vector3d u_kl;
        u_kl << p_lc.col(h)[0]/p_lc.col(h)[2], p_lc.col(h)[1]/p_lc.col(h)[2], 1;
        u_kl.normalize();
        Eigen::Matrix3d u_klSkew;
        u_klSkew << 0,        -u_kl[2], u_kl[1],
                    u_kl[2],  0,        u_kl[0],
                    -u_kl[1], u_kl[0],  0;
        Eigen::Quaterniond qh = state_kkH[h].second;
        qh.normalize();
        Eigen::Matrix3d Rh = qh.toRotationMatrix();
        F.block<3,3>(3*visible_counter,9*h) = u_klSkew*((Rh*RcamIMU).transpose());
        E.block<3,3>(3*visible_counter,0)= -u_klSkew*((Rh*RcamIMU).transpose());
        visible_counter++;
      }
    }
    delta_ells[feature_id] = F.transpose()*F-F.transpose()*E*(E.transpose()*E).inverse()*E.transpose()*F;
  }
  return delta_ells;
}
// ----------------------------------------------------------------------------
// Eigen::Ref<Eigen::MatrixXd> FeatureSelector::projectFeaturePixelOverHorizon(const image_t image&)
// {
//   std::cout << numFeatures <<std::endl;
//   Eigen::MatrixXd p_lo = Eigen::MatrixXd::Identity(numFeatures, 3); // TODO: we need to estimate the real feature location for each feature. Currently a placeholder...
//
//   return p_lo;
// }
// ----------------------------------------------------------------------------

omega_horizon_t FeatureSelector::calcInfoFromRobotMotion(
        const state_horizon_t& x_kkH, double nrImuMeasurements, double deltaImu)
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
    const auto& Qi = x_kkH[h-1].second;
    const auto& Qj = x_kkH[h].second;

    // Create Ablk and Ω as explained in the appendix
    // ROS_WARN_STREAM("Horizon " << h << " (nr IMU: " << nrImuMeasurements << ")");
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
    block4 += mats.second;
  }

  // ROS_INFO_STREAM("Omega_kkH:\n" << Omega_kkH);

  // Eigen::EigenSolver<omega_horizon_t> es(Omega_kkH);
  // ROS_WARN_STREAM("eig: " << es.eigenvalues().real().transpose());

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
          * CCt_11 * deltaImu_3 * accVarDTime_;
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

  // // https://stackoverflow.com/a/33577450/2392520
  // Eigen::JacobiSVD<omega_t> svd(covImu);
  // double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

  // ROS_INFO_STREAM("covImu ("<<cond<<"):\n" << covImu);
  // ROS_INFO_STREAM("covImu^-1 ("<<cond<<"):\n" << covImu.inverse());
  // ROS_INFO_STREAM("\nAblk:\n" << Ablk);

 return std::make_pair(covImu.inverse(), Ablk);
}

// ----------------------------------------------------------------------------

omega_horizon_t FeatureSelector::addOmegaPrior(const omega_horizon_t& OmegaIMU)
{
  return OmegaIMU;
}

// ----------------------------------------------------------------------------
