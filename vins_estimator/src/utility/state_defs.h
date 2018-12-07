#pragma once

#include <vector>
#include <utility>

#include <Eigen/Dense>

#define HORIZON 2 ///< number of frames to look into the future

#define STATE_SIZE 9 ///< size of state as defined in paper III-B1,
                     ///< which comes from the linear IMU model.


// state vector type definitions
enum : int { xTIMESTAMP = 0, xPOS = 1, xVEL = 4, xB_A = 7, xSIZE = 10 };
using xVector = Eigen::Matrix<double, xSIZE, 1>;

using state_t = std::pair<xVector, Eigen::Quaterniond>;
using state_horizon_t = std::array<state_t, HORIZON+1>;

// information matrices
using omega_t = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>;
using omega_horizon_t = Eigen::Matrix<double, STATE_SIZE*(HORIZON+1), STATE_SIZE*(HORIZON+1)>;

// Ablk -- the non-zero, non-identity matrix in equation (50)
using ablk_t = Eigen::Matrix<double, STATE_SIZE, STATE_SIZE>;



static const Eigen::Vector3d gravity = [] {
  Eigen::Vector3d tmp;
  tmp << 0, 0, -9.80665;
  return tmp;
}();