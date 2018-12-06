#include "horizon_generator.h"

#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

HorizonGenerator::HorizonGenerator(ros::NodeHandle nh)
: nh_(nh)
{
  pub_horizon_ = nh_.advertise<nav_msgs::Path>("horizon", 10);
}

// ----------------------------------------------------------------------------

xhVector HorizonGenerator::imu(const xVector& xk, int nrImuMeasurements, double deltaImu)
{
  xhVector x_kkH;
  x_kkH.segment<xSIZE>(0*xSIZE) = xk;

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

xhVector HorizonGenerator::groundTruth()
{
  return {};
}

// ----------------------------------------------------------------------------

void HorizonGenerator::visualize(const std_msgs::Header& header, const xhVector& x_kkH)
{
  nav_msgs::Path path;
  path.header = header;
  path.header.frame_id = "world";

  // include the (the to-be-corrected) current state, xk (i.e., h=0)
  for (int h=0; h<=HORIZON; ++h) {

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