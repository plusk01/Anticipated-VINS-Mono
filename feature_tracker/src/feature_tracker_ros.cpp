/**
 * @file feature_tracker_ros.cpp
 * @author Parker Lusk <parkerclusk@gmail.com>
 */

#include "anticipation/feature_tracker_ros.h"

#include <sensor_msgs/PointCloud.h>

#include "anticipation/tic_toc.h"

FeatureTrackerROS::FeatureTrackerROS(ros::NodeHandle nh)
: nh_(nh)
{
  // load ROS parameters from server
  loadParameters();

  // setup the feature tracker
  tracker_.reset(new anticipation::FeatureTracker(config_file_, params_));

  // subscribe to the input image
  image_transport::ImageTransport it(nh_);
  img_sub_ = it.subscribe("image", 1, &FeatureTrackerROS::imageCb, this);

  // publisher for features (as point cloud)
  features_pub_ = nh_.advertise<sensor_msgs::PointCloud>("feature", 1000);

  // publisher for output image annotated with features
  img_pub_ = it.advertise("output", 1);
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void FeatureTrackerROS::imageCb(const sensor_msgs::ImageConstPtr& msg)
{

  //
  // Determine if we should process this frame or not
  //

  if (++frame_ % stride_ != 0) {
    return;
  }

  cv::Mat img;

  try {
    img = cv_bridge::toCvShare(msg, "bgr8")->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    return;
  }

  cv::Mat grey;
  cv::cvtColor(img, grey, cv::COLOR_RGB2GRAY);

  //
  // Main image processing done by FeatureTracker object
  //

  static TicToc t_process("afs_cost");
  t_process.tic();
  tracker_->process(grey, msg->header.stamp.toSec());
  ROS_INFO("entire feature tracker processing costs: %f ms", t_process.toc());

  auto measurements = tracker_->measurements();
  ROS_WARN_STREAM(measurements.size() << " features");

  //
  // Publish ROS measurements
  //

  sensor_msgs::PointCloudPtr cloud(new sensor_msgs::PointCloud);
  sensor_msgs::ChannelFloat32 ids;
  sensor_msgs::ChannelFloat32 pts_u;
  sensor_msgs::ChannelFloat32 pts_v;
  sensor_msgs::ChannelFloat32 vel_x;
  sensor_msgs::ChannelFloat32 vel_y;

  cloud->header = msg->header;
  // cloud->header.frame_id = "world";

  for (size_t i=0; i<measurements.size(); ++i) {
    auto id = std::get<0>(measurements[i]);
    auto pt = std::get<1>(measurements[i]);
    auto nip = std::get<2>(measurements[i]);
    auto ell = std::get<3>(measurements[i]);
    auto vel = std::get<4>(measurements[i]);

    geometry_msgs::Point32 cloud_pt;
    cloud_pt.x = nip.x;
    cloud_pt.y = nip.y;
    cloud_pt.z = 1.0f;

    cloud->points.push_back(cloud_pt);
    ids.values.push_back(id);
    pts_u.values.push_back(pt.x);
    pts_v.values.push_back(pt.y);
    vel_x.values.push_back(vel.x);
    vel_y.values.push_back(vel.y);
  }

  cloud->channels.push_back(ids);
  cloud->channels.push_back(pts_u);
  cloud->channels.push_back(pts_v);
  cloud->channels.push_back(vel_x);
  cloud->channels.push_back(vel_y);

  features_pub_.publish(cloud);

  // optional feature visualization publish
  if (img_pub_.getNumSubscribers()) {
    for (size_t i=0; i<measurements.size(); ++i) {
      auto id = std::get<0>(measurements[i]);
      auto pt = std::get<1>(measurements[i]);
      auto nip = std::get<2>(measurements[i]);
      auto ell = std::get<3>(measurements[i]);
      auto vel = std::get<4>(measurements[i]);

      // color blend: new-blue, old-red
      double alpha = std::min(1.0, static_cast<double>(ell) / window_length_);
      auto color = cv::Scalar(255 * (1 - alpha), 0, 255 * alpha);

      // draw feature
      cv::circle(img, pt, 2, color, 2);

      // draw velocity
      constexpr double dt = 0.10;
      cv::Point2f nip0 = nip - dt*vel;
      Eigen::Vector3d b0(nip0.x, nip0.y, 1.0);
      Eigen::Vector2d a0;
      tracker_->camera()->spaceToPlane(b0, a0);
      cv::line(img, pt, cv::Point2f(a0.x(), a0.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);

      // print id next to feature
      char strID[10];
      sprintf(strID, "%d", id);
      cv::putText(img, strID, pt, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    // pack up and publish
    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding= sensor_msgs::image_encodings::BGR8;
    out.image = img;
    img_pub_.publish(out.toImageMsg());
  }
}

// ----------------------------------------------------------------------------

void FeatureTrackerROS::loadParameters()
{
  config_file_ = readROSParam<std::string>("config_file");

  // read the config file so we can populate feature tracker parameters
  cv::FileStorage config(config_file_, cv::FileStorage::READ);

  // frame processing
  stride_ = config["stride"];

  // GFTT Parameters
  params_.equalize = static_cast<int>(config["equalize"]) != 0;
  params_.borderMargin = config["border_margin"];
  params_.minDistance = config["min_distance"];
  params_.maxFeatures = config["max_features"];
  params_.reprojErrorF = config["F_reproj_error"];

  // get smoother window length, just for visualization
  window_length_ = 20;

  // release IO
  config.release();
}

// ----------------------------------------------------------------------------