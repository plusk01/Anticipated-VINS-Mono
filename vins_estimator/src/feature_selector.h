#pragma once

#include <map>
#include <utility>
#include <iostream>
#include <vector>

#include <std_msgs/Header.h>

#include <Eigen/Dense>

#include "estimator.h"

class FeatureSelector
{
public:
  using image_t = std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;

  FeatureSelector(Estimator& estimator);
  ~FeatureSelector() = default;

  void processImage(const image_t& image, const std_msgs::Header& header);

private:
  Estimator& estimator_;

};
