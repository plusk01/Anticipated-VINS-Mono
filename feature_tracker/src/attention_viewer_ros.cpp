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
  selinfo_f_.subscribe(nh_, "selection_info", 1);

  // tie them all together in a synchronizer
  sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10),
                                            img_sf_, features_f_, selinfo_f_));
  sync_->registerCallback(&AttentionViewerROS::callback, this);


  // publisher for output image with analysis annotations
  img_pub_ = it.advertise("output", 1);
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

void AttentionViewerROS::callback(const sensor_msgs::ImageConstPtr& _img,
                                  const sensor_msgs::PointCloudConstPtr& _features,
                                  const sensor_msgs::PointCloudConstPtr& _selinfo)
{
  // bail if no one wants to listen to us
  if (img_pub_.getNumSubscribers() == 0) return;

  cv::Mat img;

  try {
    img = cv_bridge::toCvShare(_img, "bgr8")->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", _img->encoding.c_str());
    return;
  }

  // for convenience
  const auto& oldIds = _selinfo->channels[0].values;
  const auto& newIds = _selinfo->channels[1].values;

  for (size_t i=0; i<_features->points.size(); ++i) {
    // auto nip = cv::Point2f{_features->points[i].x, _features->points[i].y};
    int id = static_cast<int>(_features->channels[0].values[i]);
    auto pix = cv::Point2f(_features->channels[1].values[i], _features->channels[2].values[i]);
    auto vel = cv::Point2f(_features->channels[3].values[i], _features->channels[4].values[i]);

    bool isOld = std::find(oldIds.begin(), oldIds.end(), id) != oldIds.end();
    bool isNew = std::find(newIds.begin(), newIds.end(), id) != newIds.end();

    if (isNew || isOld) {

      // update how long each feature has been alive
      if (isNew) {
        // initialize feature lifetime
        featureLifetime_[id] = 0;
      } else {
        featureLifetime_[id]++;
      }

      // color blend: new-blue, old-red
      constexpr int horizon = 10;
      double alpha = std::min(1.0, static_cast<double>(featureLifetime_[id]) / horizon);
      auto color = cv::Scalar(255 * (1 - alpha), 0, 255 * alpha);

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

    } else { // this feature was never chosen
      auto color = cv::Scalar(0, 255, 0);
      cv::circle(img, pix, 1, color, 1);
    }
  }

  // pack up and publish
  cv_bridge::CvImage out;
  out.header = _img->header;
  out.encoding = sensor_msgs::image_encodings::BGR8;
  out.image = img;
  img_pub_.publish(out.toImageMsg());
}
