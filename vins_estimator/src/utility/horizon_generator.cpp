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

state_horizon_t HorizonGenerator::imu(const state_t& state_0, const state_t& state_1,
                                        int nrImuMeasurements, double deltaImu)
{
  state_horizon_t state_kkH;

  // x_kkH.segment<xSIZE>(0*xSIZE) = xk;

  // // ROS_INFO_STREAM("xk_: " << xk_.transpose());

  // // let's just assume constant bias over the horizon
  // auto Ba = x_kkH.segment<3>(xB_A);

  // for (int h=1; h<=HORIZON; ++h) {

  //   // use the prev frame state to initialize the current k+h frame state
  //   x_kkH.segment<xSIZE>(h*xSIZE) = x_kkH.segment<xSIZE>((h-1)*xSIZE);

  //   // we assume constant angular acceleration between image frames
  //   // Qk_[h] = 

  //   // constant acceleration IMU propagation
  //   for (int i=0; i<nrImuMeasurements; ++i) {

  //     // vi, eq (11)
  //     // x_kkH.segment<3>(h*xSIZE + xVEL) += (gravity + Qk_*(ak_ - Ba))*deltaImu;
      
  //     // ti, second equality in eq (12)
  //     x_kkH.segment<3>(h*xSIZE + xPOS) += x_kkH.segment<3>(h*xSIZE + xVEL)*deltaImu;// + 0.5*gravity*deltaImu*deltaImu + 0.5*(Qk_*(ak_ - Ba))*deltaImu*deltaImu;
  //   }

  // }

  return state_kkH;
}

// ----------------------------------------------------------------------------

state_horizon_t HorizonGenerator::groundTruth(const state_t& state_0, const state_t& state_1,
                                          double timestamp, double deltaFrame)
{
  state_horizon_t state_kkH;

  // naive time synchronization with the current image frame and ground truth
  while (seek_idx_ < static_cast<int>(truth_.size()) && 
                          truth_[seek_idx_++].timestamp <= timestamp);
  int idx = seek_idx_-1;


  state_kkH[0].first.segment<3>(xPOS) = truth_[idx].p;
  state_kkH[0].first.segment<3>(xVEL) = truth_[idx].v;
  state_kkH[0].first.segment<3>(xB_A) = Eigen::Vector3d::Zero();

  for (int h=1; h<=HORIZON; ++h) {

    int i = idx + 20*h;

    state_kkH[h].first.segment<3>(xPOS) = truth_[i].p;
    state_kkH[h].first.segment<3>(xVEL) = truth_[i].v;
    state_kkH[h].first.segment<3>(xB_A) = Eigen::Vector3d::Zero();
  }

  return state_kkH;
}

// ----------------------------------------------------------------------------

void HorizonGenerator::visualize(const std_msgs::Header& header,
                                 const state_horizon_t& state_kkH)
{
  nav_msgs::Path path;
  path.header = header;
  path.header.frame_id = "world";

  // include the (the to-be-corrected) current state, xk (i.e., h=0)
  for (int h=0; h<=HORIZON; ++h) {

    // for convenience
    const auto& x_h = state_kkH[h].first;
    const auto& q_h = state_kkH[h].second;

    geometry_msgs::PoseStamped pose;
    pose.header = path.header;
    pose.pose.position.x = x_h.x();
    pose.pose.position.y = x_h.y();
    pose.pose.position.z = x_h.z();
    pose.pose.orientation.w = q_h.w();
    pose.pose.orientation.x = q_h.x();
    pose.pose.orientation.y = q_h.y();
    pose.pose.orientation.z = q_h.z();

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
    truth data;

    data.timestamp = std::stod((*it)[0])*1e-9; // convert ns to s
    data.p << std::stod((*it)[1]), std::stod((*it)[2]), std::stod((*it)[3]);
    data.q = Eigen::Quaterniond(std::stod((*it)[4]), std::stod((*it)[5]), std::stod((*it)[6]), std::stod((*it)[7]));
    data.v << std::stod((*it)[8]), std::stod((*it)[9]), std::stod((*it)[10]);
    data.w << std::stod((*it)[11]), std::stod((*it)[12]), std::stod((*it)[13]);
    data.a << std::stod((*it)[14]), std::stod((*it)[15]), std::stod((*it)[16]);

    truth_.push_back(data);
  }

}
