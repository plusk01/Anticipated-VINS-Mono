#include "feature_selector.h"

#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

FeatureSelector::FeatureSelector(ros::NodeHandle nh, Estimator& estimator)
: nh_(nh), estimator_(estimator)
{

  pub_horizon_ = nh_.advertise<nav_msgs::Path>("horizon", 10);

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
    visualizeFutureHorizon(header, x_kkH);
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

FeatureSelector::xhVector FeatureSelector::generateFutureHorizon(int nrImuMeasurements,
                                                                 double deltaImu)
{

  // generate the horizon based on the requested scheme
  if (horizonGeneration_ == IMU) {
    return horizonImu(nrImuMeasurements, deltaImu);
  } else if (horizonGeneration_ == GT) {
    return horizonGroundTruth();
  }

}

// ----------------------------------------------------------------------------

FeatureSelector::xhVector FeatureSelector::horizonGroundTruth()
{
  return {};
}

// ----------------------------------------------------------------------------

FeatureSelector::xhVector FeatureSelector::horizonImu(int nrImuMeasurements,
                                                      double deltaImu)
{
  xhVector x_kkH;
  x_kkH.segment<xSIZE>(0*xSIZE) = xk_;

  // ROS_INFO_STREAM("xk_: " << xk_.transpose());

  // let's just assume constant bias over the horizon
  auto Ba = x_kkH.segment<3>(xB_A);

  for (int h=1; h<=HORIZON; ++h) {

    // use the prev frame state to initialize the current k+h frame state
    x_kkH.segment<xSIZE>(h*xSIZE) = x_kkH.segment<xSIZE>((h-1)*xSIZE);

    // we assume constant angular acceleration between image frames
    // Qk_[h] = 

    // constant acceleration IMU propagation
    for (int i=0; i<nrImuMeasurements; ++i) {

      // vi, eq (11)
      // x_kkH.segment<3>(h*xSIZE + xVEL) += (gravity + Qk_*(ak_ - Ba))*deltaImu;
      
      // ti, second equality in eq (12)
      x_kkH.segment<3>(h*xSIZE + xPOS) += x_kkH.segment<3>(h*xSIZE + xVEL)*deltaImu;// + 0.5*gravity*deltaImu*deltaImu + 0.5*(Qk_*(ak_ - Ba))*deltaImu*deltaImu;
    }

  }

  return x_kkH;
}

// ----------------------------------------------------------------------------

void FeatureSelector::visualizeFutureHorizon(const std_msgs::Header& header,
                                             const xhVector& x_kkH)
{
  nav_msgs::Path path;
  path.header = header;
  path.header.frame_id = "world";

  xhVector tmp = xhVector::Zero();

  tmp.segment<xSIZE>(0) = xk_;

  for (int h=0; h<=HORIZON; ++h) {

    // use the prev frame state to initialize the current k+h frame state
    if (h>0) {
      tmp.segment<xSIZE>(h*xSIZE) = tmp.segment<xSIZE>((h-1)*xSIZE);
      tmp.segment<3>(h*xSIZE + xPOS) += Eigen::Vector3d(1, 0, 0);
    }

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = x_kkH(h*xSIZE + xPOS+0);
    pose.pose.position.y = x_kkH(h*xSIZE + xPOS+1);
    pose.pose.position.z = x_kkH(h*xSIZE + xPOS+2);
    pose.pose.orientation.w = 1;
    pose.pose.orientation.x = 0;
    pose.pose.orientation.y = 0;
    pose.pose.orientation.z = 0;

    path.poses.push_back(pose);
  }


  pub_horizon_.publish(path);
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
