/**
 * @file feature_tracker_ros.cpp
 * @author Parker Lusk <parkerclusk@gmail.com>
 */

#include "anticipation/attention_viewer_ros.h"

AttentionViewerROS::AttentionViewerROS(ros::NodeHandle nh)
: nh_(nh)
{
  // get calib file
  std::string calib_file;
  if (!nh_.getParam("config_file", calib_file)) {
    ROS_ERROR_STREAM("[AttentionViewerROS] Could not read rosParam (config_file)");
    nh_.shutdown();
    return;
  }

  // create camera model from calibration YAML file
  m_camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);

  // subscribe to the input image
  image_transport::ImageTransport it(nh_);
  img_sf_.subscribe(it, "image", 1);

  // subscribe to original features from feature tracker
  // and selected features from attention algorithm
  features_f_.subscribe(nh_, "feature", 1);
  subset_f_.subscribe(nh_, "subset", 1);

  // tie them all together in a synchronizer
  sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10),
                                            img_sf_, features_f_/*, subset_f_*/));
  sync_->registerCallback(&AttentionViewerROS::callback, this);


  // publisher for output image with analysis annotations
  img_pub_ = it.advertise("output", 1);
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void AttentionViewerROS::callback(const sensor_msgs::ImageConstPtr& _img,
                                  const sensor_msgs::PointCloudConstPtr& _features/*,
                                  const sensor_msgs::PointCloudConstPtr& _subset*/)
{
  cv::Mat img;

  try {
    img = cv_bridge::toCvShare(_img, "bgr8")->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", _img->encoding.c_str());
    return;
  }

  //
  // Publish ROS measurements
  //

  // sensor_msgs::PointCloudPtr cloud(new sensor_msgs::PointCloud);
  // sensor_msgs::ChannelFloat32 ids;
  // sensor_msgs::ChannelFloat32 pts_u;
  // sensor_msgs::ChannelFloat32 pts_v;
  // sensor_msgs::ChannelFloat32 vel_x;
  // sensor_msgs::ChannelFloat32 vel_y;

  // cloud->header = msg->header;
  // // cloud->header.frame_id = "world";

  // for (size_t i=0; i<measurements.size(); ++i) {
  //   auto id = std::get<0>(measurements[i]);
  //   auto pt = std::get<1>(measurements[i]);
  //   auto nip = std::get<2>(measurements[i]);
  //   auto ell = std::get<3>(measurements[i]);
  //   auto vel = std::get<4>(measurements[i]);

  //   geometry_msgs::Point32 cloud_pt;
  //   cloud_pt.x = nip.x;
  //   cloud_pt.y = nip.y;
  //   cloud_pt.z = 1.0f;

  //   cloud->points.push_back(cloud_pt);
  //   ids.values.push_back(id);
  //   pts_u.values.push_back(pt.x);
  //   pts_v.values.push_back(pt.y);
  //   vel_x.values.push_back(vel.x);
  //   vel_y.values.push_back(vel.y);
  // }

  // cloud->channels.push_back(ids);
  // cloud->channels.push_back(pts_u);
  // cloud->channels.push_back(pts_v);
  // cloud->channels.push_back(vel_x);
  // cloud->channels.push_back(vel_y);

  // features_pub_.publish(cloud);

  // optional feature visualization publish
  if (img_pub_.getNumSubscribers()) {
    for (size_t i=0; i<_features->points.size(); ++i) {
      // auto nip = cv::Point2f{_features->points[i].x, _features->points[i].y};
      int id = static_cast<int>(_features->channels[0].values[i]);
      auto pix = cv::Point2f(_features->channels[1].values[i], _features->channels[2].values[i]);
      auto vel = cv::Point2f(_features->channels[3].values[i], _features->channels[4].values[i]);

      auto color = cv::Scalar(255, 0, 255);

      // draw feature
      cv::circle(img, pix, 2, color, 2);

      // // draw velocity
      // constexpr double dt = 0.10;
      // cv::Point2f nip0 = nip - dt*vel;
      // Eigen::Vector3d b0(nip0.x, nip0.y, 1.0);
      // Eigen::Vector2d a0;
      // tracker_->camera()->spaceToPlane(b0, a0);
      // cv::line(img, pix, cv::Point2f(a0.x(), a0.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);

      // print id next to feature
      char strID[10];
      sprintf(strID, "%d", id);
      cv::putText(img, strID, pix, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    // pack up and publish
    cv_bridge::CvImage out;
    out.header = _img->header;
    out.encoding = sensor_msgs::image_encodings::BGR8;
    out.image = img;
    img_pub_.publish(out.toImageMsg());
  }
}
