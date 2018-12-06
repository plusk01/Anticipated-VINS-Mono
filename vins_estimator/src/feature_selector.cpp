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
    const Eigen::Vector3d& Ba)
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

  // the last accelerometer measurement
  ak_ = ((Eigen::Vector3d() << 9.17739, 0.0735499,  -2.61511).finished()); //a;

  // ROS_INFO_STREAM("accel: " << a.transpose());
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




  // Calculate the information content of each of the new features
  auto Delta_ells = calcInfoFromFeatures(image);

  // Calculate the information content of each of the currently used features
  auto Delta_used_ells = calcInfoFromFeatures(image);

  // Use the IMU model to propagate forward the current noise
  auto OmegaIMU = calcInfoFromRobotMotion();

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
    return hgen_->imu(state_0_, state_k_, nrImuMeasurements, deltaImu);
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

Eigen::MatrixXd FeatureSelector::calcInfoFromRobotMotion()
{
  return {};
}

// ----------------------------------------------------------------------------
