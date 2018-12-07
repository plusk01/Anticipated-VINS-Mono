#include "feature_selector.h"

FeatureSelector::FeatureSelector(ros::NodeHandle nh, Estimator& estimator)
: nh_(nh), estimator_(estimator)
{

  // create future state horizon generator / manager
  hgen_ = std::unique_ptr<HorizonGenerator>(new HorizonGenerator(nh_));
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
  auto OmegaIMU = calcInfoFromRobotMotion();

  // Add in prior information (eq 16)
  auto Omega = addOmegaPrior(OmegaIMU);

  // Calculate the information content of each of the new features
  auto Delta_ells = calcInfoFromFeatures(image);

  // Calculate the information content of each of the currently used features
  auto Delta_used_ells = calcInfoFromFeatures(image);


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

std::vector<Eigen::MatrixXd> FeatureSelector::calcInfoFromFeatures(const image_t& image)
{
  return {};
}

// ----------------------------------------------------------------------------

omega_horizon_t FeatureSelector::calcInfoFromRobotMotion()
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

  // TODO: Be more clever and only calculate the upper-triangular
  //       and then transpose since this is a symmetric PSD matrix

  omega_horizon_t Omega_kkH;

  for (int h=1; h<=HORIZON; ++h) { // for consecutive frames in horizon
    // Create Ablk and Ω as explained in the appendix
    auto mats = createLinearImuMatrices();

    // convenience: select sub-blocks to add to, based on h
    Eigen::Ref<Eigen::MatrixXd> block1 = Omega_kkH.block<STATE_SIZE, STATE_SIZE>((h-1)*STATE_SIZE, (h-1)*STATE_SIZE);
    Eigen::Ref<Eigen::MatrixXd> block2 = Omega_kkH.block<STATE_SIZE, STATE_SIZE>((h-1)*STATE_SIZE, h*STATE_SIZE);
    Eigen::Ref<Eigen::MatrixXd> block3 = Omega_kkH.block<STATE_SIZE, STATE_SIZE>(h*STATE_SIZE, (h-1)*STATE_SIZE);
    Eigen::Ref<Eigen::MatrixXd> block4 = Omega_kkH.block<STATE_SIZE, STATE_SIZE>(h*STATE_SIZE, h*STATE_SIZE);

    // At*Ω*A (top-left sub-block)
    block1 += mats.second.transpose()*mats.first*mats.second;

    // At*Ω (top-right sub-block)
    block2 += mats.second.transpose()*mats.first;
    
    // Ω*A (bottom-left sub-block)
    block3 += mats.first*mats.second;

    // Ω (bottom-right sub-block)
    block4 += mats.second;
  }

  return Omega_kkH;
}

// ----------------------------------------------------------------------------

std::pair<omega_t, ablk_t> FeatureSelector::createLinearImuMatrices()
{
 return std::make_pair<omega_t, ablk_t>({}, {});
}

// ----------------------------------------------------------------------------

omega_horizon_t FeatureSelector::addOmegaPrior(const omega_horizon_t& OmegaIMU)
{
  return OmegaIMU;
}

// ----------------------------------------------------------------------------
