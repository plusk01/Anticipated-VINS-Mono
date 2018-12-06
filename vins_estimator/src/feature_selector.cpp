#include "feature_selector.h"

FeatureSelector::FeatureSelector(ros::NodeHandle nh, Estimator& estimator)
: nh_(nh), estimator_(estimator)
{

  // create future state horizon generator / manager
  hgen_ = std::unique_ptr<HorizonGenerator>(new HorizonGenerator(nh));
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
  auto x_kkH = generateFutureHorizon(nrImuMeasurements, deltaImu);

  if (visualize_) {
    hgen_->visualize(header, x_kkH);
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

void FeatureSelector::setCurrentStateFromImuPropagation(const Eigen::Vector3d& P,
                                                        const Eigen::Quaterniond& Q,
                                                        const Eigen::Vector3d& V,
                                                        const Eigen::Vector3d& a,
                                                        const Eigen::Vector3d& Ba)
{
  // set the propagated-forward state of the current frame
  xk_.segment<3>(xPOS) = P;
  xk_.segment<3>(xVEL) = V;
  xk_.segment<3>(xB_A) = Ba;

  // the last accelerometer measurement
  ak_ = ((Eigen::Vector3d() << 9.17739, 0.0735499,  -2.61511).finished()); //a;

  ROS_INFO_STREAM("accel: " << a.transpose());

  // rotation of the body w.r.t. the world at the kth frame
  Qk_[0] = Q;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

xhVector FeatureSelector::generateFutureHorizon(int nrImuMeasurements,
                                                                 double deltaImu)
{

  // generate the horizon based on the requested scheme
  if (horizonGeneration_ == IMU) {
    return hgen_->imu(xk_, nrImuMeasurements, deltaImu);
  } else { //if (horizonGeneration_ == GT) {
    return hgen_->groundTruth();
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
