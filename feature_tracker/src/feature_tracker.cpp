/**
 * @file feature_tracker.cpp
 * @author Parker Lusk <parkerclusk@gmail.com>
 */

#include "anticipation/feature_tracker.h"

namespace anticipation
{

FeatureTracker::FeatureTracker(const std::string& calib_file,
                               const Parameters& params)
: params_(params)
{
  // create camera model from calibration YAML file
  m_camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);

  // initialize Lucas-Kanade Optical Flow
  flow_ = initOpticalFlow();

  // Create the adaptive histogram equalization object
  clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
}

// ----------------------------------------------------------------------------

void FeatureTracker::process(const cv::Mat& img, double timestamp)
{
  static std::vector<cv::Point2f> features0;
  static double timestamp0 = 0;

  // notation: <var>0 is from time k-1, while <var>1 is from time k (current)
  cv::Mat img1;

  // equalize the image histogram to minimize effects of lighting variation
  if (params_.equalize) {
    clahe_->apply(img, img1);
  } else {
    img1 = img;
  }

  // on first iteration, initialize last image to the current image
  static cv::Mat img0 = img1;

  // update dt
  dt_ = timestamp - timestamp0;

  //
  // Track previous features into this frame using optical flow
  //

  features1_.clear();

  if (features0.size()) {
    std::vector<unsigned char> matches;
    calculateFlow(img0, img1, features0, features1_, matches);

    // features and metadata should all be the same size
    assert(features0.size() == features1_.size());
    assert(features0.size() == ids1_.size());
    assert(features0.size() == lifetimes1_.size());
    assert(features0.size() == scores1_.size());

    // Only keep features that were matched in both frames
    // and that are within the border of the image
    size_t j = 0;
    for (size_t i=0; i<matches.size(); ++i) {
      // make sure the propagated point is within the border of the image
      if (matches[i] && withinBorder(features1_[i])) {
        features0[j] = features0[i];
        features1_[j] = features1_[i];

        // update feature metadata at the same time
        ids1_[j] = (ids1_[i] == 0) ? nextId_++ : ids1_[i];  // set ID if unset
        lifetimes1_[j] = lifetimes1_[i] + 1;                // inc lifetime
        scores1_[j] = scores1_[i];

        j++;
      }
    }
    features0.resize(j);
    features1_.resize(j);
    ids1_.resize(j);
    lifetimes1_.resize(j);
    scores1_.resize(j);
  }

  //
  // Geometric verification and outlier rejection
  //
  
  // use fundamental matrix and RANSAC to reject points that
  // don't make sense geometrically.
  rejectWithF(features0);

  // enforce minimum distance of features, and build a mask
  // that indicates where new features should be found
  cv::Mat mask = enforceMinDist(features0);


  //
  // Create normalized image plane points and velocities
  //

  createMeasurements(features0);


  //
  // Detect new features to maintain the user-defined number
  //

  int numFeaturesToDetect = params_.maxFeatures - static_cast<int>(features1_.size());
  if (numFeaturesToDetect > 0) {

    // look for more features using mask to detect in sparse regions
    std::vector<cv::Point2f> newFeatures1;
    std::vector<float> newScores1;
    detectFeatures(img1, newFeatures1, newScores1, numFeaturesToDetect, mask);

    // add new features for next time
    size_t newSize = features1_.size() + newFeatures1.size();
    features1_.reserve(newSize);
    ids1_.reserve(newSize);
    lifetimes1_.reserve(newSize);
    scores1_.reserve(newSize);
    for (size_t i=0; i<newFeatures1.size(); ++i) {
      features1_.push_back(newFeatures1[i]);
      ids1_.push_back(0);
      lifetimes1_.push_back(0);
      scores1_.push_back(newScores1[i]);
    }
  }

  // save for next iteration
  img0 = img1;
  features0 = features1_;
  timestamp0 = timestamp;
}

// ----------------------------------------------------------------------------
// Private Methods
// ----------------------------------------------------------------------------

cv::Ptr<cv::SparsePyrLKOpticalFlow> FeatureTracker::initOpticalFlow()
{
  return cv::SparsePyrLKOpticalFlow::create(); 
}

// ----------------------------------------------------------------------------

void FeatureTracker::calculateFlow(const cv::Mat& grey0, const cv::Mat& grey1,
                                   const std::vector<cv::Point2f>& features0,
                                   std::vector<cv::Point2f>& features1,
                                   std::vector<unsigned char>& matches)
{
  flow_->calc(grey0, grey1, features0, features1, matches);
}

// ----------------------------------------------------------------------------

void FeatureTracker::detectFeatures(const cv::Mat& grey,
                                    std::vector<cv::Point2f>& features,
                                    std::vector<float>& scores,
                                    int maxCorners,
                                    const cv::Mat& mask)
{
  // default parameters for GFTT
  constexpr double cornerQuality = 0.01;
  constexpr int blockSize = 3;
  constexpr bool useHarrisDetector = false;
  constexpr double k = 0.04;

  cvmodified::goodFeaturesToTrack(grey, features, scores, maxCorners,
                                  cornerQuality, params_.minDistance, mask,
                                  blockSize, useHarrisDetector, k);
}

// ----------------------------------------------------------------------------

bool FeatureTracker::withinBorder(const cv::Point2f& pt)
{
  int border = params_.borderMargin;
  int u = std::round(pt.x);
  int v = std::round(pt.y);
  return (border <= u && u < m_camera_->imageWidth() - border) && 
         (border <= v && v < m_camera_->imageHeight() - border);
}

// ----------------------------------------------------------------------------

cv::Mat FeatureTracker::enforceMinDist(std::vector<cv::Point2f>& features0)
{
  // create a new mask
  cv::Mat mask(m_camera_->imageHeight(), m_camera_->imageWidth(), CV_8UC1, cv::Scalar(255));

  // 
  // Prefer to keep points that have been tracked the longest
  // 

  std::vector<measurement_t> sorted;
  sorted.reserve(features1_.size());
  for (size_t i=0; i<features1_.size(); ++i) {
    auto pt0 = features0[i];
    auto pt = features1_[i];
    auto id = ids1_[i];
    auto ell = lifetimes1_[i];
    auto score = scores1_[i];

    // note that pt0 is in the place of 'nip' for typical measurements
    // and that the 'vel' field is redundant
    sorted.push_back(std::make_tuple(id, pt, score, pt0, ell, pt0));
  }

  auto comp = [](const auto& a, const auto& b) -> bool {
                    return std::get<mLIFE>(a) > std::get<mLIFE>(b);
              };
  std::sort(sorted.begin(), sorted.end(), comp);

  //
  // Use mask to enforce a minimum distance between features
  //

  std::vector<cv::Point2f> kept_pts, kept_pts0;
  std::vector<unsigned int> kept_ids, kept_lifetimes;
  std::vector<double> kept_scores;
  kept_pts0.reserve(features1_.size());
  kept_pts.reserve(features1_.size());
  kept_ids.reserve(features1_.size());
  kept_lifetimes.reserve(features1_.size());
  kept_scores.reserve(features1_.size());

  for (size_t i=0; i<sorted.size(); ++i) {
    auto pt = std::get<mPT>(sorted[i]);
    if (mask.at<uchar>(pt) == 255) {
      auto id = std::get<mID>(sorted[i]);
      auto score = std::get<mSCORE>(sorted[i]);
      auto pt0 = std::get<mNIP>(sorted[i]);
      auto ell = std::get<mLIFE>(sorted[i]);

      // keep good points and their metadata
      kept_pts0.push_back(pt0);
      kept_pts.push_back(pt);
      kept_ids.push_back(id);
      kept_lifetimes.push_back(ell);
      kept_scores.push_back(score);

      // update mask
      cv::circle(mask, pt, params_.minDistance, cv::Scalar(0), -1);
    }
  }

  kept_pts0.swap(features0);
  kept_pts.swap(features1_);
  kept_ids.swap(ids1_);
  kept_lifetimes.swap(lifetimes1_);
  kept_scores.swap(scores1_);

  return mask;
}

// ----------------------------------------------------------------------------

bool FeatureTracker::rejectWithF(std::vector<cv::Point2f>& features0)
{
  // features and metadata should all be the same size
  assert(features0.size() == features1_.size());
  assert(features0.size() == ids1_.size());
  assert(features0.size() == lifetimes1_.size());

  // fundamental matrix estimation requires 8 points
  if (features1_.size() < 8) return false;

  std::vector<unsigned char> status;
  cv::findFundamentalMat(features0, features1_, cv::FM_RANSAC,
                                  params_.reprojErrorF, 0.99, status);

  // reject outliers
  size_t j = 0;
  for (size_t i=0; i<status.size(); ++i) {
    if (status[i]) {
      features0[j] = features0[i];
      features1_[j] = features1_[i];
      ids1_[j] = ids1_[i];
      lifetimes1_[j] = lifetimes1_[i];
      scores1_[j] = scores1_[i];
      j++;
    }
  }
  features0.resize(j);
  features1_.resize(j);
  ids1_.resize(j);
  lifetimes1_.resize(j);
  scores1_.resize(j);

  return true;
}

// ----------------------------------------------------------------------------

void FeatureTracker::createMeasurements(const std::vector<cv::Point2f>& features0)
{
  // features and metadata should all be the same size
  assert(features0.size() == features1_.size());
  assert(features0.size() == ids1_.size());
  assert(features0.size() == lifetimes1_.size());
  assert(features0.size() == scores1_.size());
  assert(std::find(lifetimes1_.begin(), lifetimes1_.end(), 0) == lifetimes1_.end());

  // allocate memory for new measurements
  std::vector<measurement_t> measurements;
  measurements.reserve(features1_.size());

  // find the largest score to use as a normalizer to convert to "probability"
  auto it = std::max_element(scores1_.begin(), scores1_.end());
  float eta = (it != scores1_.end()) ? *it : 1.0;

  for (size_t i=0; i<features1_.size(); ++i) {
    auto pt0 = features0[i];
    auto pt = features1_[i];
    auto id = ids1_[i];
    auto ell = lifetimes1_[i];
    auto prob = scores1_[i] / eta;

    // undistort feature to normalized image plane
    Eigen::Vector2d a(pt.x, pt.y);
    Eigen::Vector3d b;
    m_camera_->liftProjective(a, b);
    auto nip = cv::Point2f(b.x() / b.z(), b.y() / b.z());

    // undistort previous feature to nip
    a << pt0.x, pt0.y;
    m_camera_->liftProjective(a, b);
    auto nip0 = cv::Point2f(b.x() / b.z(), b.y() / b.z());

    // calculate velocity
    auto vel = (nip - nip0) / dt_;

    // create the measurement tuple
    measurements.push_back(std::make_tuple(id, pt, prob, nip, ell, vel));
  }

  measurements.swap(measurements_);
}

} // namespace anticipation
