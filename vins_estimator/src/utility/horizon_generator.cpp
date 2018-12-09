#include "horizon_generator.h"

HorizonGenerator::HorizonGenerator(ros::NodeHandle nh)
: nh_(nh)
{
  pub_horizon_ = nh_.advertise<nav_msgs::Path>("horizon", 10);

  // load ground truth data if available
  std::string data_csv;
  if (nh_.getParam("gt_data_csv", data_csv)) {
    loadGroundTruth(data_csv);
  }
}

// ----------------------------------------------------------------------------

void HorizonGenerator::setParameters(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
{
  q_IC_ = R;
  t_IC_ = t;
}

// ----------------------------------------------------------------------------

state_horizon_t HorizonGenerator::imu(
                      const state_t& state_0, const state_t& state_1,
                      const Eigen::Vector3d& a, const Eigen::Vector3d& w,
                      int nrImuMeasurements, double deltaImu)
{
  state_horizon_t state_kkH;

  // initialize with the the last frame's pose (tail of optimizer window)
  state_kkH[0] = state_0;

  // Additionally, since we already have an IMU propagated estimate
  // (yet-to-be-corrected) of the current frame, we will start there.
  state_kkH[1] = state_1;

  // let's just assume constant bias over the horizon
  auto Ba = state_kkH[0].first.segment<3>(xB_A);

  // we also assume a constant angular velocity during the horizon
  auto Qimu = Utility::deltaQ(w * deltaImu);

  for (int h=2; h<=HORIZON; ++h) { // NOTE: we already have k+1.

    // use the prev frame state to initialize the k+h frame state
    state_kkH[h] = state_kkH[h-1];

    // constant acceleration IMU propagation
    for (int i=0; i<nrImuMeasurements; ++i) {

      // propagate attitude with incremental IMU update
      state_kkH[h].second = state_kkH[h].second * Qimu;

      // Convenience: quat from world to current IMU-rate body pose
      const auto& q_hi = state_kkH[h].second;

      // vi, eq (11)
      state_kkH[h].first.segment<3>(xVEL) += (gravity + q_hi*(a - Ba))*deltaImu;
      
      // ti, second equality in eq (12)
      state_kkH[h].first.segment<3>(xPOS) += state_kkH[h].first.segment<3>(xVEL)*deltaImu + 0.5*gravity*deltaImu*deltaImu + 0.5*(q_hi*(a - Ba))*deltaImu*deltaImu;
    }

  }

  return state_kkH;
}

// ----------------------------------------------------------------------------

state_horizon_t HorizonGenerator::groundTruth(const state_t& state_0,
                                    const state_t& state_1, double deltaFrame)
{
  state_horizon_t state_kkH;

  // get the timestamp of the previous frame
  double timestamp = state_0.first.coeff(xTIMESTAMP);

  // if this condition is true, then it is likely the first state_0 (which may have random values)
  if (timestamp > truth_.back().timestamp) timestamp = truth_.front().timestamp;

  // naive time synchronization with the previous image frame and ground truth
  while (seek_idx_ < static_cast<int>(truth_.size()) && 
                          truth_[seek_idx_++].timestamp <= timestamp);
  int idx = seek_idx_-1;

  // initialize state horizon structure with [xk]. The rest are future states.
  state_kkH[0].first = state_0.first;
  state_kkH[0].second = state_0.second;

  // ground-truth pose of previous frame
  Eigen::Vector3d prevP = truth_[idx].p;
  Eigen::Quaterniond prevQ = truth_[idx].q;

  // predict pose of camera for frames k to k+H
  for (int h=1; h<=HORIZON; ++h) {

    // while the inertial frame of ground truth and vins will be different,
    // it is not a problem because we only care about _relative_ transformations.
    auto gtPose = getNextFrameTruth(idx, deltaFrame);

    // Ground truth orientation of frame k+h w.r.t. orientation of frame k+h-1
    auto relQ = prevQ.inverse() * gtPose.q;

    // Ground truth position of frame k+h w.r.t. position of frame k+h-1
    auto relP = gtPose.q.inverse() * (gtPose.p - prevP);


    // "predict" where the current frame in the horizon (k+h)
    // will be by applying this relative rotation to
    // the previous frame (k+h-1)
    state_kkH[h].first.segment<3>(xPOS) = state_kkH[h-1].first.segment<3>(xPOS) + state_kkH[h-1].second * relP;
    state_kkH[h].second = state_kkH[h-1].second * relQ;

    // for next iteration
    prevP = gtPose.p;
    prevQ = gtPose.q;
  }
  
  return state_kkH;
}

// ----------------------------------------------------------------------------

void HorizonGenerator::visualize(const std_msgs::Header& header,
                                 const state_horizon_t& state_kkH)
{
  // Don't waste cycles unnecessarily
  if (pub_horizon_.getNumPublishers() == 0) return;

  nav_msgs::Path path;
  path.header = header;
  path.header.frame_id = "world";

  // include the current state, xk (i.e., h=0)
  for (int h=0; h<=HORIZON; ++h) {

    // for convenience
    const auto& x_h = state_kkH[h].first;
    const auto& q_h = state_kkH[h].second;

    // Compose world-to-imu estimate with imu-to-cam extrinsic transform
    Eigen::Vector3d P = x_h.segment<3>(xPOS) + q_h * t_IC_;
    Eigen::Quaterniond R = q_h * q_IC_;

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = P.x();
    pose.pose.position.y = P.y();
    pose.pose.position.z = P.z();
    pose.pose.orientation.w = R.w();
    pose.pose.orientation.x = R.x();
    pose.pose.orientation.y = R.y();
    pose.pose.orientation.z = R.z();

    path.poses.push_back(pose);
  }


  pub_horizon_.publish(path);
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void HorizonGenerator::loadGroundTruth(std::string data_csv)
{
  // open the CSV file and create an iterator
  std::ifstream file(data_csv);
  CSVIterator it(file);

  // throw away the headers
  ++it;

  truth_.clear();

  for (; it != CSVIterator(); ++it) {
    truth_t data;

    data.timestamp = std::stod((*it)[0])*1e-9; // convert ns to s
    data.p << std::stod((*it)[1]), std::stod((*it)[2]), std::stod((*it)[3]);
    data.q = Eigen::Quaterniond(std::stod((*it)[4]), std::stod((*it)[5]), std::stod((*it)[6]), std::stod((*it)[7]));
    data.v << std::stod((*it)[8]), std::stod((*it)[9]), std::stod((*it)[10]);
    data.w << std::stod((*it)[11]), std::stod((*it)[12]), std::stod((*it)[13]);
    data.a << std::stod((*it)[14]), std::stod((*it)[15]), std::stod((*it)[16]);

    truth_.push_back(data);
  }

  // reset seek
  seek_idx_ = 0;

}

// ----------------------------------------------------------------------------

HorizonGenerator::truth_t HorizonGenerator::getNextFrameTruth(int& idx,
                                                            double deltaFrame)
{
  double nextTimestep = truth_[idx].timestamp + deltaFrame;

  // naive time synchronization with the previous image frame and ground truth
  while (idx < static_cast<int>(truth_.size()) && 
                          truth_[idx++].timestamp <= nextTimestep);

  return truth_[idx];
}
