// Copyright 2021 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/velocity_smoother/smoother/smoother_base.hpp"

#include "autoware/motion_utils/resample/resample.hpp"
#include "autoware/motion_utils/trajectory/conversion.hpp"
#include "autoware/motion_utils/trajectory/trajectory.hpp"
#include "autoware/velocity_smoother/resample.hpp"
#include "autoware/velocity_smoother/trajectory_utils.hpp"
#include "autoware/interpolation/linear_interpolation.hpp"

#include <autoware_utils_geometry/geometry.hpp>
#include <autoware_utils_math/unit_conversion.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace autoware::velocity_smoother
{

namespace
{
TrajectoryPoints applyPreProcess(
  const TrajectoryPoints & input, const double interval, const bool use_resampling)
{
  using autoware::motion_utils::calcArcLength;
  using autoware::motion_utils::convertToTrajectory;
  using autoware::motion_utils::convertToTrajectoryPointArray;
  using autoware::motion_utils::resampleTrajectory;

  if (!use_resampling) {
    return input;
  }

  TrajectoryPoints output;
  std::vector<double> arc_length;

  // since the resampling takes a long time, omit the resampling when it is not requested
  const auto traj_length = calcArcLength(input);
  for (double s = 0; s < traj_length; s += interval) {
    arc_length.push_back(s);
  }

  const auto points = resampleTrajectory(convertToTrajectory(input), arc_length);
  output = convertToTrajectoryPointArray(points);
  output.back() = input.back();  // keep the final speed.

  return output;
}
}  // namespace

SmootherBase::SmootherBase(
  rclcpp::Node & node, const std::shared_ptr<autoware_utils_debug::TimeKeeper> time_keeper)
: time_keeper_(time_keeper)
{
  auto & p = base_param_;
  p.max_accel = node.declare_parameter<double>("normal.max_acc");
  p.min_decel = node.declare_parameter<double>("normal.min_acc");
  p.stop_decel = node.declare_parameter<double>("stop_decel");
  p.max_jerk = node.declare_parameter<double>("normal.max_jerk");
  p.min_jerk = node.declare_parameter<double>("normal.min_jerk");
  p.min_decel_for_lateral_acc_lim_filter =
    node.declare_parameter<double>("min_decel_for_lateral_acc_lim_filter");
  p.sample_ds = node.declare_parameter<double>("resample_ds");
  p.curvature_threshold = node.declare_parameter<double>("curvature_threshold");
  p.lateral_acceleration_limits =
    node.declare_parameter<std::vector<double>>("lateral_acceleration_limits");
  p.velocity_thresholds = node.declare_parameter<std::vector<double>>("velocity_thresholds");
  p.steering_angle_rate_limits =
    node.declare_parameter<std::vector<double>>("steering_angle_rate_limits");
  p.curvature_calculation_distance =
    node.declare_parameter<double>("curvature_calculation_distance");
  p.decel_distance_before_curve = node.declare_parameter<double>("decel_distance_before_curve");
  p.decel_distance_after_curve = node.declare_parameter<double>("decel_distance_after_curve");
  p.min_curve_velocity = node.declare_parameter<double>("min_curve_velocity");
  p.resample_param.max_trajectory_length = node.declare_parameter<double>("max_trajectory_length");
  p.resample_param.min_trajectory_length = node.declare_parameter<double>("min_trajectory_length");
  p.resample_param.resample_time = node.declare_parameter<double>("resample_time");
  p.resample_param.dense_resample_dt = node.declare_parameter<double>("dense_resample_dt");
  p.resample_param.dense_min_interval_distance =
    node.declare_parameter<double>("dense_min_interval_distance");
  p.resample_param.sparse_resample_dt = node.declare_parameter<double>("sparse_resample_dt");
  p.resample_param.sparse_min_interval_distance =
    node.declare_parameter<double>("sparse_min_interval_distance");
  p.enable_4ws = node.declare_parameter<bool>("enable_4ws", false);
  p.k_ref_lut = node.declare_parameter<std::vector<double>>("k_ref_lut", std::vector<double>{});
  p.rr_lut = node.declare_parameter<std::vector<double>>("rr_lut", std::vector<double>{});
  p.delta_f_lut = node.declare_parameter<std::vector<double>>("delta_f_lut", std::vector<double>{});
}

void SmootherBase::setWheelBase(const double wheel_base)
{
  base_param_.wheel_base = wheel_base;
}

void SmootherBase::setMaxAccel(const double max_acceleration)
{
  base_param_.max_accel = max_acceleration;
}

void SmootherBase::setMaxJerk(const double max_jerk)
{
  base_param_.max_jerk = max_jerk;
}

void SmootherBase::setParam(const BaseParam & param)
{
  base_param_ = param;
}

SmootherBase::BaseParam SmootherBase::getBaseParam() const
{
  return base_param_;
}

double SmootherBase::getMaxAccel() const
{
  return base_param_.max_accel;
}

double SmootherBase::getMinDecel() const
{
  return base_param_.min_decel;
}

double SmootherBase::getMaxJerk() const
{
  return base_param_.max_jerk;
}

double SmootherBase::getMinJerk() const
{
  return base_param_.min_jerk;
}

// Template function for computing ratio limits
template <typename ThresholdType, typename ComputeRatioFunc>
std::vector<std::pair<double, double>> SmootherBase::computeRatioLimits(
  const std::vector<double> & velocity_thresholds,
  const std::vector<ThresholdType> & threshold_values, ComputeRatioFunc compute_ratio_func) const
{
  std::vector<std::pair<double, double>> output;
  constexpr double epsilon = 1e-5;

  // Process each velocity threshold
  for (size_t i = 0; i < velocity_thresholds.size(); ++i) {
    ThresholdType threshold = threshold_values[i];
    double vi = velocity_thresholds[i];
    double vi_1 = (i == 0) ? 0.0 : velocity_thresholds[i - 1];

    // Add the ratio pair for this threshold using the provided function
    output.push_back(
      std::make_pair(
        compute_ratio_func(threshold, vi_1, epsilon), compute_ratio_func(threshold, vi, epsilon)));
  }

  // Add the last part
  auto v_max = velocity_thresholds.back();
  output.push_back(
    std::make_pair(compute_ratio_func(threshold_values.back(), v_max, epsilon), 0.0));

  return output;
}

// Template function for computing velocity limits
template <typename RatioType, typename ThresholdType, typename ComputeVelocityFunc>
double SmootherBase::computeVelocityLimit(
  const RatioType local_ratio, const std::vector<std::pair<double, double>> & ratio_limits,
  const std::vector<double> & velocity_thresholds,
  const std::vector<ThresholdType> & threshold_values,
  ComputeVelocityFunc compute_velocity_func) const
{
  // Iterate through the limits in reverse order
  for (int i = static_cast<int>(ratio_limits.size()) - 1; i >= 0; --i) {
    const double lower = ratio_limits[i].second;
    const double higher = ratio_limits[i].first;

    // If ratio is higher than the upper bound, continue to the next range
    if (local_ratio > higher) {
      continue;
    }

    // If ratio is lower than or equal to the lower bound
    if (local_ratio <= lower) {
      if (i == static_cast<int>(ratio_limits.size()) - 1) {
        return std::numeric_limits<double>::max();  // max_double equivalent
      } else {
        return velocity_thresholds[i];
      }
    }

    // Between lower and higher, calculate the velocity threshold
    ThresholdType current_threshold;
    if (i == static_cast<int>(ratio_limits.size()) - 1) {
      current_threshold = threshold_values.back();
    } else {
      current_threshold = threshold_values[i];
    }

    return compute_velocity_func(current_threshold, local_ratio);
  }

  // If no appropriate range is found
  return 0.0;
}

std::vector<std::pair<double, double>>
SmootherBase::computeLateralAccelerationVelocitySquareRatioLimits() const
{
  auto compute_lateral_acc_ratio = [](double acc, double v, double epsilon) {
    return acc / (v * v + epsilon);
  };

  return computeRatioLimits<double>(
    base_param_.velocity_thresholds, base_param_.lateral_acceleration_limits,
    compute_lateral_acc_ratio);
}

std::vector<std::pair<double, double>> SmootherBase::computeSteerRateVelocityRatioLimits() const
{
  auto compute_steer_rate_ratio = [](double rate_deg, double v, double epsilon) {
    return autoware_utils_math::deg2rad(rate_deg) / (v + epsilon);
  };

  return computeRatioLimits<double>(
    base_param_.velocity_thresholds, base_param_.steering_angle_rate_limits,
    compute_steer_rate_ratio);
}

double SmootherBase::computeVelocityLimitFromLateralAcc(
  const double local_curvature,
  const std::vector<std::pair<double, double>> lateral_acceleration_velocity_square_ratio_limits)
  const
{
  auto compute_velocity = [](double acc, double curvature) {
    return std::sqrt(acc / std::max(curvature, 1.0E-5));
  };

  return computeVelocityLimit<double, double>(
    local_curvature, lateral_acceleration_velocity_square_ratio_limits,
    base_param_.velocity_thresholds, base_param_.lateral_acceleration_limits, compute_velocity);
}

double SmootherBase::computeVelocityLimitFromSteerRate(
  const double local_steer_rate_velocity_ratio,
  const std::vector<std::pair<double, double>> steer_rate_velocity_ratio_limits) const
{
  auto compute_velocity = [](double rate_deg, double ratio) {
    return autoware_utils_math::deg2rad(rate_deg) / ratio;
  };

  return computeVelocityLimit<double, double>(
    local_steer_rate_velocity_ratio, steer_rate_velocity_ratio_limits,
    base_param_.velocity_thresholds, base_param_.steering_angle_rate_limits, compute_velocity);
}

TrajectoryPoints SmootherBase::applyLateralAccelerationFilter(
  const TrajectoryPoints & input, [[maybe_unused]] const double v0,
  [[maybe_unused]] const double a0, [[maybe_unused]] const bool enable_smooth_limit,
  const bool use_resampling, const double input_points_interval) const
{
  autoware_utils_debug::ScopedTimeTrack st(__func__, *time_keeper_);

  if (input.size() < 3) {
    return input;  // cannot calculate lateral acc. do nothing.
  }

  // Interpolate with constant interval distance for lateral acceleration calculation.
  TrajectoryPoints output;
  const double points_interval =
    use_resampling ? base_param_.sample_ds : input_points_interval;  // [m]
  // since the resampling takes a long time, omit the resampling when it is not requested
  if (use_resampling) {
    std::vector<double> out_arclength;
    const auto traj_length = autoware::motion_utils::calcArcLength(input);
    for (double s = 0; s < traj_length; s += points_interval) {
      out_arclength.push_back(s);
    }
    const auto output_traj = autoware::motion_utils::resampleTrajectory(
      autoware::motion_utils::convertToTrajectory(input), out_arclength);
    output = autoware::motion_utils::convertToTrajectoryPointArray(output_traj);
    output.back() = input.back();  // keep the final speed.
  } else {
    output = input;
  }

  const size_t idx_dist = static_cast<size_t>(
    std::max(static_cast<int>((base_param_.curvature_calculation_distance) / points_interval), 1));

  // Calculate curvature assuming the trajectory points interval is constant
  const auto curvature_v = trajectory_utils::calcTrajectoryCurvatureFrom3Points(output, idx_dist);

  //  Decrease speed according to lateral G
  const size_t before_decel_index =
    static_cast<size_t>(std::round(base_param_.decel_distance_before_curve / points_interval));
  const size_t after_decel_index =
    static_cast<size_t>(std::round(base_param_.decel_distance_after_curve / points_interval));

  const auto latacc_min_vel_arr =
    enable_smooth_limit ? trajectory_utils::calcVelocityProfileWithConstantJerkAndAccelerationLimit(
                            output, v0, a0, base_param_.min_jerk, base_param_.max_accel,
                            base_param_.min_decel_for_lateral_acc_lim_filter)
                        : std::vector<double>{};

  const auto lateral_acceleration_velocity_square_ratio_limits =
    computeLateralAccelerationVelocitySquareRatioLimits();

  for (size_t i = 0; i < output.size(); ++i) {
    double curvature = 0.0;
    const size_t start = i > after_decel_index ? i - after_decel_index : 0;
    const size_t end = std::min(output.size(), i + before_decel_index + 1);
    for (size_t j = start; j < end; ++j) {
      if (j >= curvature_v.size()) return output;
      curvature = std::max(curvature, std::fabs(curvature_v.at(j)));
    }
    double v_curvature_max = computeVelocityLimitFromLateralAcc(
      curvature, lateral_acceleration_velocity_square_ratio_limits);
    if (base_param_.enable_4ws) {
      // Look up the rear angle from the LUT directly, rather than reading
      // rear_wheel_angle_rad, which is not populated until
      // applySteeringRateLimit runs later in the pipeline.
      double rear_angle = 0.0;
      if (curvature > base_param_.k_ref_lut.back()) {
        rear_angle = base_param_.rr_lut.back() * base_param_.delta_f_lut.back();
      } else {
        const double rr_i = autoware::interpolation::lerp(
          base_param_.k_ref_lut, base_param_.rr_lut, curvature);
        const double front_i = autoware::interpolation::lerp(
          base_param_.k_ref_lut, base_param_.delta_f_lut, curvature);
        rear_angle = rr_i * front_i;
      }
      const double cos_r = std::max(std::cos(rear_angle), 0.0);
      v_curvature_max *= std::sqrt(cos_r);
    }

    v_curvature_max = std::max(v_curvature_max, base_param_.min_curve_velocity);

    if (enable_smooth_limit) {
      if (i >= latacc_min_vel_arr.size()) return output;
      v_curvature_max = std::max(v_curvature_max, latacc_min_vel_arr.at(i));
    }
    if (output.at(i).longitudinal_velocity_mps > v_curvature_max) {
      output.at(i).longitudinal_velocity_mps = static_cast<float>(v_curvature_max);
    }
  }
  return output;
}

TrajectoryPoints SmootherBase::applySteeringRateLimit(
  const TrajectoryPoints & input, const bool use_resampling,
  const double input_points_interval) const
{
  autoware_utils_debug::ScopedTimeTrack st(__func__, *time_keeper_);

  if (input.size() < 3) {
    return input;  // cannot calculate the desired velocity. do nothing.
  }

  const auto steer_rate_velocity_ratio_limits = computeSteerRateVelocityRatioLimits();

  // Interpolate with constant interval distance for lateral acceleration calculation.
  const double points_interval = use_resampling ? base_param_.sample_ds : input_points_interval;

  auto output = applyPreProcess(input, points_interval, use_resampling);

  const size_t idx_dist = static_cast<size_t>(
    std::max(static_cast<int>((base_param_.curvature_calculation_distance) / points_interval), 1));

  // Step1. Calculate curvature assuming the trajectory points interval is constant.
  const auto curvature_v = trajectory_utils::calcTrajectoryCurvatureFrom3Points(output, idx_dist);

  /* // Step2. Calculate steer rate for each trajectory point.
  std::vector<double> steer_rate_velocity_ratio_arr(output.size());
  for (size_t i = 0; i < output.size() - 1; i++) {
    // steer
    auto & steer_front = output.at(i + 1).front_wheel_angle_rad;
    auto & steer_back = output.at(i).front_wheel_angle_rad;

    // calculate the just 2 steering angle
    steer_front = static_cast<float>(std::atan(base_param_.wheel_base * curvature_v.at(i + 1)));
    steer_back = static_cast<float>(std::atan(base_param_.wheel_base * curvature_v.at(i)));

    const auto steering_diff = std::fabs(steer_front - steer_back);

    steer_rate_velocity_ratio_arr.at(i) =
      steering_diff / (points_interval + std::numeric_limits<double>::epsilon());
  }*/ //2WS autoware original implementation

  // Step2. Calculate steer rate for each trajectory point.
  // 4WS: split into two passes, mirroring ComputeSteeringRateAngle4WS.m.
  //   Pass 1 fills both wheel angles; Pass 2 derives the rate ratio.
  // Stock Autoware did both in one loop, which wrote every point twice.
  std::vector<double> steer_rate_velocity_ratio_arr(output.size());

  // --- Step2-1. Fill front and rear steering angles for every point. ---
  for (size_t i = 0; i < output.size(); i++) {
    const double k_signed = curvature_v.at(i);

    // The LUT is magnitude-only (k_ref_lut spans 0 .. k_max), so query on
    // |curvature| and restore the turn direction afterwards. This matches
    // Autoware's own fabs() convention on curvature used in Step4 below.
    const double k_abs = std::fabs(k_signed);
    const double s = (k_signed > 0.0) ? 1.0 : ((k_signed < 0.0) ? -1.0 : 0.0);

    double rr_i = 0.0;     // rear-to-front steering ratio at this curvature
    double front_i = 0.0;  // front wheel angle [rad]

    if (base_param_.enable_4ws) {
      if (k_abs > base_param_.k_ref_lut.back()) {
        // Beyond the table: saturate at its last row. Read from the LUT
        // rather than hardcoding, so the clamp cannot drift from the table
        // if it is ever regenerated.
        rr_i = base_param_.rr_lut.back();
        front_i = s * base_param_.delta_f_lut.back();
      } else {
        // C++ equivalent of MATLAB interp1(). NOTE: lerp throws
        // std::invalid_argument if the query is outside base_keys, unlike
        // interp1 which returns NaN. The k_abs + guard above keeps every
        // query inside the table's domain.
        rr_i = autoware::interpolation::lerp(
          base_param_.k_ref_lut, base_param_.rr_lut, k_abs);
        front_i = s * autoware::interpolation::lerp(
          base_param_.k_ref_lut, base_param_.delta_f_lut, k_abs);
      }
    } else {
      // 2WS fallback: identical to stock Autoware. Keeps behaviour unchanged
      // while enable_4ws is false.
      front_i = std::atan(base_param_.wheel_base * k_signed);
      rr_i = 0.0;
    }

    // Populate the trajectory message. front_wheel_angle_rad was already
    // written by stock Autoware; rear_wheel_angle_rad was left at 0 and is
    // read by applyLateralAccelerationFilter's cosine correction.
    output.at(i).front_wheel_angle_rad = static_cast<float>(front_i);
    output.at(i).rear_wheel_angle_rad = static_cast<float>(rr_i * front_i);
  }

  // --- Step2-2. Steering rate from whichever axle must move further. ---
  // 4WS has two independent actuators; the binding constraint is the larger
  // angular change, not the sum or difference.
  for (size_t i = 0; i + 1 < output.size(); i++) {
    const double front_diff = std::fabs(
      output.at(i + 1).front_wheel_angle_rad - output.at(i).front_wheel_angle_rad);
    const double rear_diff = std::fabs(
      output.at(i + 1).rear_wheel_angle_rad - output.at(i).rear_wheel_angle_rad);

    steer_rate_velocity_ratio_arr.at(i) =
      std::max(front_diff, rear_diff) /
      (points_interval + std::numeric_limits<double>::epsilon());
  }

  steer_rate_velocity_ratio_arr.back() = steer_rate_velocity_ratio_arr.at((output.size() - 2));

  // Step3. Remove noise by mean filter.
  for (size_t i = 1; i < steer_rate_velocity_ratio_arr.size() - 1; i++) {
    steer_rate_velocity_ratio_arr.at(i) =
      (steer_rate_velocity_ratio_arr.at(i - 1) + steer_rate_velocity_ratio_arr.at(i) +
       steer_rate_velocity_ratio_arr.at(i + 1)) /
      3.0;
  }

  // Step4. Limit velocity by steer rate.
  for (size_t i = 0; i < output.size() - 1; i++) {
    if (fabs(curvature_v.at(i)) < base_param_.curvature_threshold) {
      continue;
    }

    const auto mean_vel =
      (output.at(i).longitudinal_velocity_mps + output.at(i + 1).longitudinal_velocity_mps) / 2.0;

    const auto local_velocity_limit = computeVelocityLimitFromSteerRate(
      steer_rate_velocity_ratio_arr.at(i), steer_rate_velocity_ratio_limits);

    if (mean_vel < local_velocity_limit) {
      continue;
    }

    for (size_t k = 0; k < 2; k++) {
      auto & velocity = output.at(i + k).longitudinal_velocity_mps;
      const float target_velocity = static_cast<float>(std::max(
        base_param_.min_curve_velocity,
        std::min(local_velocity_limit, velocity * (local_velocity_limit / mean_vel))));
      velocity = std::min(velocity, target_velocity);
    }
  }

  return output;
}

}  // namespace autoware::velocity_smoother
