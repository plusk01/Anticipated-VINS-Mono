/**
 * @file feature_tracker.cpp
 * @author Parker Lusk <parkerclusk@gmail.com>
 */

#pragma once

#include <string>
#include <algorithm>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <camodocal/camera_models/Camera.h>
#include <camodocal/camera_models/CameraFactory.h>

#include "anticipation/cvmodified.h"

// convenience location identifiers for measurement_t
enum : int { mID=0, mPT=1, mSCORE=2, mNIP=3, mLIFE=4, mVEL=5 };

namespace anticipation
{

  class FeatureTracker
  {
  public:
    /**
     * @brief      Feature tracker parameters
     */
    struct Parameters
    {
      // general image processing
      bool equalize = true;
      int borderMargin = 10;
      // GFTT Detector parameters
      double minDistance = 30.0;
      int maxFeatures = 1000;
      // fundamental matrix outlier rejection
      double reprojErrorF = 1.0;
    };

    // measurement: <id, pt, score/prob, nip, lifetime, vel>
    using measurement_t = std::tuple<unsigned int, cv::Point2f, float,
                                  cv::Point2f, unsigned int, cv::Point2f>;

    FeatureTracker(const std::string& calib_file, const Parameters& params);
    ~FeatureTracker() = default;

    /**
     * @brief      Process a new image to track features
     *
     * @param[in]  img        The image
     * @param[in]  timestamp  The timestamp of the image
     */
    void process(const cv::Mat& img, double timestamp);

    // getters
    const std::vector<measurement_t>& measurements() const { return measurements_; }
    const camodocal::CameraPtr& camera() const { return m_camera_; }

  private:
    camodocal::CameraPtr m_camera_; ///< geometric camera model
    Parameters params_; ///< feature tracker parameters
    cv::Ptr<cv::CLAHE> clahe_; ///< contrast-limited adaptive histogram equalization
    cv::Ptr<cv::SparsePyrLKOpticalFlow> flow_; ///< optical flow

    // features and related data
    std::vector<cv::Point2f> features1_;    ///< current features
    std::vector<unsigned int> ids1_;        ///< unique id for each feature
    std::vector<unsigned int> lifetimes1_;  ///< how long each feat. has been tracked
    std::vector<double> scores1_;           ///< detction score of each feature

    // counter for unique id generation
    unsigned int nextId_ = 1;

    std::vector<measurement_t> measurements_; ///< feature, id, velocity, lifetime

    double dt_ = 1.0; ///< current period of the images (used for velocity calc)

    cv::Ptr<cv::GFTTDetector> initGFTTDetector();
    cv::Ptr<cv::SparsePyrLKOpticalFlow> initOpticalFlow();

    /**
     * @brief      Use optical flow to find where features moved to
     *
     * @param[in]  grey0      Previous greyscale image where original features are
     * @param[in]  grey1      Current greyscale image where features moved to
     * @param[in]  features0  Original features
     * @param      features1  Original features propagated into current frame
     * @param      matches    Vector indicating which pairs of features are correspondences
     */
    void calculateFlow(const cv::Mat& grey0, const cv::Mat& grey1,
                       const std::vector<cv::Point2f>& features0,
                       std::vector<cv::Point2f>& features1,
                       std::vector<unsigned char>& matches);

    /**
     * @brief      Detect new features in a greyscale image
     *
     * @param[in]  grey        Greyscale image to find features in
     * @param      features    The detected images
     * @param[in]  maxCorners  Maximum number of corners to detect
     * @param[in]  mask        A mask indicated which areas of the 
     *                         image to detect features in
     */ 
    void detectFeatures(const cv::Mat& grey,
                        std::vector<cv::Point2f>& features,
                        std::vector<float>& scores,
                        int maxCorners,
                        const cv::Mat& mask = cv::Mat());

    /**
     * @brief      Check if point is within the image + a margin
     *
     * @param[in]  pt    The point to check
     *
     * @return     Boolean
     */
    bool withinBorder(const cv::Point2f& pt);

    /**
     * @brief      Enforce a minimum distance between features.
     *              Features that are too close are culled.
     *
     * @param      features0  Features from the previous frame
     *
     * @return     A mask is created to indicate where new
     *              features should be found.
     */
    cv::Mat enforceMinDist(std::vector<cv::Point2f>& features0);

    /**
     * @brief      Geometric validation using fundamental matrix
     *
     * @param      features0  Features from the previous frame
     *
     * @return     Boolean indicating if validation was performed
     */
    bool rejectWithF(std::vector<cv::Point2f>& features0);

    /**
     * @brief      Create measurements (measurement_t)
     *
     * @param[in]  features0  Features from the previous frame
     */
    void createMeasurements(const std::vector<cv::Point2f>& features0);
  };

}
