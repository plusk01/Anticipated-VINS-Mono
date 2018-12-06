#pragma once

#include <Eigen/Dense>

#define HORIZON 10 ///< number of frames to look into the future

// state vector type definitions
enum : int { xPOS = 0, xVEL = 3, xB_A = 6, xSIZE = 9 };
using xVector = Eigen::Matrix<double, xSIZE, 1>;
using xhVector = Eigen::Matrix<double, xSIZE*(HORIZON + 1), 1>;

static const Eigen::Vector3d gravity = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, -9.80665;
  return tmp;
}();