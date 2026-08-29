// Copyright 2022 Tier IV, Inc.
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

#include "autoware/velocity_smoother/smoother/jerk_filtered_smoother.hpp"
#include "autoware/velocity_smoother/smoother/smoother_base.hpp"
#include "autoware/velocity_smoother/trajectory_utils.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>

using autoware::velocity_smoother::JerkFilteredSmoother;
using autoware::velocity_smoother::SmootherBase;
using autoware::velocity_smoother::TrajectoryPoints;

// Test fixture to create a SmootherBase instance with controlled parameters
class TestSmootherBase : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::init(0, nullptr);
    auto node_options = rclcpp::NodeOptions{};
    node_options.append_parameter_override("algorithm_type", "JerkFiltered");
    node_options.append_parameter_override("publish_debug_trajs", false);
    const auto autoware_test_utils_dir =
      ament_index_cpp::get_package_share_directory("autoware_test_utils");
    const auto velocity_smoother_dir =
      ament_index_cpp::get_package_share_directory("autoware_velocity_smoother");
    node_options.arguments(
      {"--ros-args", "--params-file", autoware_test_utils_dir + "/config/test_common.param.yaml",
       "--params-file", autoware_test_utils_dir + "/config/test_nearest_search.param.yaml",
       "--params-file", autoware_test_utils_dir + "/config/test_vehicle_info.param.yaml",
       "--params-file", velocity_smoother_dir + "/config/default_velocity_smoother.param.yaml",
       "--params-file", velocity_smoother_dir + "/config/default_common.param.yaml",
       "--params-file", velocity_smoother_dir + "/config/JerkFiltered.param.yaml"});
    node = std::make_shared<rclcpp::Node>("test_smoother_base_node", node_options);
    debug_processing_time_detail_ =
      node->create_publisher<autoware_utils_debug::ProcessingTimeDetail>(
        "~/debug/processing_time_detail_ms", 1);//3 lines added

    auto time_keeper =
      std::make_shared<autoware_utils_debug::TimeKeeper>(debug_processing_time_detail_);

    SmootherBase::BaseParam params;
    params.max_accel = 1.0;
    params.min_decel = -1.0;
    params.stop_decel = 0.5;
    params.max_jerk = 0.8;
    params.min_jerk = -0.8;
    params.min_decel_for_lateral_acc_lim_filter = -0.5;
    params.sample_ds = 0.1;
    params.curvature_threshold = 0.01;
    params.lateral_acceleration_limits = {2.0, 1.8, 1.5};
    params.velocity_thresholds = {10.0, 20.0, 30.0};
    params.steering_angle_rate_limits = {30.0, 20.0, 10.0};
    params.curvature_calculation_distance = 1.0;
    params.decel_distance_before_curve = 3.0;
    params.decel_distance_after_curve = 2.0;
    params.min_curve_velocity = 2.0;
    params.wheel_base = 2.7;
    params.resample_param.max_trajectory_length = 200.0;
    params.resample_param.min_trajectory_length = 30.0;
    params.resample_param.resample_time = 0.1;
    params.resample_param.dense_resample_dt = 0.1;
    params.resample_param.dense_min_interval_distance = 0.1;
    params.resample_param.sparse_resample_dt = 0.5;
    params.resample_param.sparse_min_interval_distance = 4.0;

    smoother_base = std::make_shared<JerkFilteredSmoother>(*node, time_keeper);
    // smoother_base->setWheelBase(2.7); // Set a typical wheelbase value
    std::dynamic_pointer_cast<SmootherBase>(smoother_base)->setParam(params);
  }

  void TearDown() override { rclcpp::shutdown(); }

  std::shared_ptr<rclcpp::Node> node;
  std::shared_ptr<JerkFilteredSmoother> smoother_base;
  rclcpp::Publisher<autoware_utils_debug::ProcessingTimeDetail>::SharedPtr
    debug_processing_time_detail_;
};

TEST_F(TestSmootherBase, ComputeLateralAccelerationVelocitySquareRatioLimits)
{
  const auto limits = smoother_base->computeLateralAccelerationVelocitySquareRatioLimits();

  // Check that we have the expected number of limits
  // Number of thresholds + 1 for the final segment
  EXPECT_EQ(limits.size(), 4);

  // Check the values for the first limit pair
  constexpr double epsilon = 1e-5;
  EXPECT_NEAR(limits[0].first, 2.0 / (0.0 * 0.0 + epsilon), 1e-10);
  EXPECT_NEAR(limits[0].second, 2.0 / (10.0 * 10.0 + epsilon), 1e-10);

  // Check the values for the second limit pair
  EXPECT_NEAR(limits[1].first, 1.8 / (10.0 * 10.0 + epsilon), 1e-10);
  EXPECT_NEAR(limits[1].second, 1.8 / (20.0 * 20.0 + epsilon), 1e-10);

  // Check the values for the third limit pair
  EXPECT_NEAR(limits[2].first, 1.5 / (20.0 * 20.0 + epsilon), 1e-10);
  EXPECT_NEAR(limits[2].second, 1.5 / (30.0 * 30.0 + epsilon), 1e-10);

  // Check the values for the last limit pair
  EXPECT_NEAR(limits[3].first, 1.5 / (30.0 * 30.0 + epsilon), 1e-10);
  EXPECT_NEAR(limits[3].second, 0.0, 1e-10);
}

TEST_F(TestSmootherBase, ComputeSteerRateVelocityRatioLimits)
{
  const auto limits = smoother_base->computeSteerRateVelocityRatioLimits();

  // Check that we have the expected number of limits
  // Number of thresholds + 1 for the final segment
  EXPECT_EQ(limits.size(), 4);

  constexpr double epsilon = 1e-5;
  constexpr double deg2rad = M_PI / 180.0;

  // Check the values for the first limit pair
  EXPECT_NEAR(limits[0].first, 30.0 * deg2rad / (0.0 + epsilon), 1e-10);
  EXPECT_NEAR(limits[0].second, 30.0 * deg2rad / (10.0 + epsilon), 1e-10);

  // Check the values for the second limit pair
  EXPECT_NEAR(limits[1].first, 20.0 * deg2rad / (10.0 + epsilon), 1e-10);
  EXPECT_NEAR(limits[1].second, 20.0 * deg2rad / (20.0 + epsilon), 1e-10);

  // Check the values for the third limit pair
  EXPECT_NEAR(limits[2].first, 10.0 * deg2rad / (20.0 + epsilon), 1e-10);
  EXPECT_NEAR(limits[2].second, 10.0 * deg2rad / (30.0 + epsilon), 1e-10);

  // Check the values for the last limit pair
  EXPECT_NEAR(limits[3].first, 10.0 * deg2rad / (30.0 + epsilon), 1e-10);
  EXPECT_NEAR(limits[3].second, 0.0, 1e-10);
}

TEST_F(TestSmootherBase, ComputeVelocityLimitFromLateralAcc)
{
  const auto limits = smoother_base->computeLateralAccelerationVelocitySquareRatioLimits();

  // Test case 1: Curvature below the lowest threshold
  // Should return a large velocity
  double curvature = 0.000001;  // Very small curvature
  double velocity_limit = smoother_base->computeVelocityLimitFromLateralAcc(curvature, limits);
  EXPECT_GT(velocity_limit, 100.0);

  // Test case 2: Curvature between thresholds
  // For curvature = 0.0025 (between first and second threshold)
  // Expected velocity = sqrt(1.5 / 0.0025) = 24.49
  curvature = 0.0025;
  velocity_limit = smoother_base->computeVelocityLimitFromLateralAcc(curvature, limits);
  EXPECT_NEAR(velocity_limit, 24.494897, 1e-5);

  // Test case 3: Curvature above all thresholds
  // Expected velocity = sqrt(2.0 / 1.0) = 1.414
  curvature = 1.0;  // Very high curvature
  velocity_limit = smoother_base->computeVelocityLimitFromLateralAcc(curvature, limits);
  EXPECT_NEAR(velocity_limit, 1.414, 1e-3);

  // Test case 4: Curvature exactly at a threshold
  // For curvature = 0.00166... (exactly at second threshold)
  // Should return the second velocity threshold (20.0)
  curvature = 1.8 / (20.0 * 20.0);
  velocity_limit = smoother_base->computeVelocityLimitFromLateralAcc(curvature, limits);
  EXPECT_NEAR(velocity_limit, 20.0, 1e-10);
}

TEST_F(TestSmootherBase, ComputeVelocityLimitFromSteerRate)
{
  const auto limits = smoother_base->computeSteerRateVelocityRatioLimits();
  // Limits: [+inf, 3.0, 1.0, 0.33]

  constexpr double deg2rad = M_PI / 180.0;

  // Test case 1: Steer rate ratio below the lowest threshold
  // Should return a large velocity
  // Expected velocity = 0.33 (selected threshold) / 0.1 (input) * 30 (speed threshold) = 100.0
  double steer_rate_ratio = 0.1 * deg2rad;  // Very small ratio
  double velocity_limit =
    smoother_base->computeVelocityLimitFromSteerRate(steer_rate_ratio, limits);
  EXPECT_NEAR(velocity_limit, 100.0, 1e-10);

  // Test case 2: Steer rate ratio on the threshold
  // For ratio = 1.0 * deg2rad (between first and second threshold)
  // Expected velocity = 20.0 * deg2rad / (1.0 * deg2rad) = 20.0
  steer_rate_ratio = 1.0 * deg2rad;
  velocity_limit = smoother_base->computeVelocityLimitFromSteerRate(steer_rate_ratio, limits);
  EXPECT_NEAR(velocity_limit, 20.0, 1e-10);

  // Test case 3: Steer rate ratio above all thresholds
  // For ratio = 1.0 * deg2rad (between first and second threshold)
  // Expected velocity = 3.0 (selected threshold) / 100 (input) * 10 (speed threshold) = 0.3
  steer_rate_ratio = 100.0 * deg2rad;  // Very high ratio
  velocity_limit = smoother_base->computeVelocityLimitFromSteerRate(steer_rate_ratio, limits);
  EXPECT_NEAR(velocity_limit, 0.3, 1e-10);

  // Test case 4: Steer rate ratio between thresholds
  // For ratio = 0.4 * deg2rad (exactly at second threshold)
  // Expected velocity = 0.33 (selected threshold) / 0.4 (input) * 30 (speed threshold) = 25.0
  steer_rate_ratio = 0.4 * deg2rad;
  velocity_limit = smoother_base->computeVelocityLimitFromSteerRate(steer_rate_ratio, limits);
  EXPECT_NEAR(velocity_limit, 25.0, 1e-10);
}

TEST_F(TestSmootherBase, FourWheelSteeringProducesValidAngles)
{
  auto smoother = std::dynamic_pointer_cast<SmootherBase>(smoother_base);
  auto params = smoother->getBaseParam();

  // SetUp's setParam() overwrites BaseParam with a hand-built struct, so the
  // LUTs loaded from the YAML are lost there - but they remain declared on
  // the node, so recover them from the parameters.
  params.enable_4ws = true;
  params.k_ref_lut = node->get_parameter("k_ref_lut").as_double_array();
  params.rr_lut = node->get_parameter("rr_lut").as_double_array();
  params.delta_f_lut = node->get_parameter("delta_f_lut").as_double_array();
  params.curvature_calculation_distance = 1.0;
  smoother->setParam(params);

  ASSERT_FALSE(params.k_ref_lut.empty());
  ASSERT_EQ(params.k_ref_lut.size(), params.rr_lut.size());
  ASSERT_EQ(params.k_ref_lut.size(), params.delta_f_lut.size());

  const double ds = 0.1;
  const size_t n = 80;
  const size_t mid = 40;

  // Points on a circle of radius R have curvature exactly 1/R, which injects
  // a known curvature through an interface that computes its own.
  auto make_arc = [&](const double kappa) {
    TrajectoryPoints traj;
    const double radius = 1.0 / std::abs(kappa);
    const double sgn = (kappa > 0.0) ? 1.0 : -1.0;
    for (size_t i = 0; i < n; ++i) {
      const double theta = (static_cast<double>(i) * ds) / radius;
      autoware_planning_msgs::msg::TrajectoryPoint p;
      p.pose.position.x = radius * std::sin(theta);
      p.pose.position.y = sgn * radius * (1.0 - std::cos(theta));
      p.longitudinal_velocity_mps = 5.0;
      traj.push_back(p);
    }
    return traj;
  };

  // use_resampling = false keeps the points exactly where we placed them.
  const auto left = smoother->applySteeringRateLimit(make_arc(0.3), false, ds);
  const auto right = smoother->applySteeringRateLimit(make_arc(-0.3), false, ds);

  const double fl = left.at(mid).front_wheel_angle_rad;
  const double rl = left.at(mid).rear_wheel_angle_rad;
  const double fr = right.at(mid).front_wheel_angle_rad;
  const double rr = right.at(mid).rear_wheel_angle_rad;

  EXPECT_GT(std::abs(fl), 1e-6);          // the LUT produced a real angle
  EXPECT_LT(fl * rl, 0.0);                // rear counter-phases front
  EXPECT_LE(std::abs(fl), 0.7 + 1e-6);    // within the steering limit
  EXPECT_LE(std::abs(rl), 0.7 + 1e-6);
  EXPECT_NEAR(fl, -fr, 1e-4);             // right mirrors left
  EXPECT_NEAR(rl, -rr, 1e-4);
}

TEST_F(TestSmootherBase, DumpFourWheelSteeringSweep)
{
  auto smoother = std::dynamic_pointer_cast<SmootherBase>(smoother_base);
  auto params = smoother->getBaseParam();
  params.enable_4ws = true;
  params.k_ref_lut = node->get_parameter("k_ref_lut").as_double_array();
  params.rr_lut = node->get_parameter("rr_lut").as_double_array();
  params.delta_f_lut = node->get_parameter("delta_f_lut").as_double_array();
  params.curvature_calculation_distance = 1.0;
  smoother->setParam(params);

  const double ds = 0.1;
  const size_t n = 600;

  // Curvature sweep: 0 -> +0.6 -> 0 -> -0.6 -> 0.
  // Covers the full LUT domain and both turn directions.
  auto kappa_at = [&](size_t i) {
    const double t = static_cast<double>(i) / static_cast<double>(n);
    return 0.6 * std::sin(2.0 * M_PI * t);
  };

  // Integrate heading to build the path from the demanded curvature.
  TrajectoryPoints traj;
  double x = 0.0, y = 0.0, th = 0.0;
  for (size_t i = 0; i < n; ++i) {
    autoware_planning_msgs::msg::TrajectoryPoint p;
    p.pose.position.x = x;
    p.pose.position.y = y;
    p.longitudinal_velocity_mps = 5.0;
    traj.push_back(p);
    th += kappa_at(i) * ds;
    x += ds * std::cos(th);
    y += ds * std::sin(th);
  }

  const auto out = smoother->applySteeringRateLimit(traj, false, ds);

  std::ofstream f("/tmp/cpp_4ws_sweep.csv");
  f << "i,x,y,kappa_demanded,front_rad,rear_rad,velocity\n";
  for (size_t i = 0; i < out.size(); ++i) {
    f << i << ","
      << out.at(i).pose.position.x << ","
      << out.at(i).pose.position.y << ","
      << kappa_at(i) << ","
      << out.at(i).front_wheel_angle_rad << ","
      << out.at(i).rear_wheel_angle_rad << ","
      << out.at(i).longitudinal_velocity_mps << "\n";
  }
  f.close();
  std::cout << "wrote /tmp/cpp_4ws_sweep.csv with " << out.size() << " rows\n";

  EXPECT_EQ(out.size(), n);
}

TEST_F(TestSmootherBase, DumpNinetyDegreeLeftTurnCritical)
{
  auto smoother = std::dynamic_pointer_cast<SmootherBase>(smoother_base);
  auto params = smoother->getBaseParam();
  params.enable_4ws = true;
  params.k_ref_lut = node->get_parameter("k_ref_lut").as_double_array();
  params.rr_lut = node->get_parameter("rr_lut").as_double_array();
  params.delta_f_lut = node->get_parameter("delta_f_lut").as_double_array();
  params.curvature_calculation_distance = 1.0;
  smoother->setParam(params);

  // --- Direct port of the Simulink MATLAB Function
  // "Trajectory Generator: 90 deg left turn critical for 2WS" (chart_317).
  // Values kept identical to the model, including R = 1 (note: the block
  // title says R = 2.5 m, which disagrees with the code).
  constexpr size_t N = 100;
  constexpr size_t N1 = 30;   // first straight
  constexpr size_t N2 = 40;   // circular arc
  constexpr size_t N3 = 30;   // second straight
  constexpr double R = 1.0;
  constexpr double L1 = 2.0;
  constexpr double L2 = 15.0;
  const double v = 4.0 / 3.6;

  TrajectoryPoints traj(N);
  for (auto & p : traj) {
    p.longitudinal_velocity_mps = static_cast<float>(v);
  }

  auto set_yaw = [](autoware_planning_msgs::msg::TrajectoryPoint & p, const double yaw) {
    p.pose.orientation.z = std::sin(yaw / 2.0);
    p.pose.orientation.w = std::cos(yaw / 2.0);
  };

  for (size_t i = 0; i < N1; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(N1 - 1);
    traj.at(i).pose.position.x = t * L1;
    traj.at(i).pose.position.y = 0.0;
    set_yaw(traj.at(i), 0.0);
  }

  const double cx = L1;
  const double cy = R;
  for (size_t k = 0; k < N2; ++k) {
    const double theta =
      -M_PI / 2.0 + (M_PI / 2.0) * static_cast<double>(k) / static_cast<double>(N2 - 1);
    const size_t i = N1 + k;
    traj.at(i).pose.position.x = cx + R * std::cos(theta);
    traj.at(i).pose.position.y = cy + R * std::sin(theta);
    set_yaw(traj.at(i), theta + M_PI / 2.0);
  }

  const double x0 = traj.at(N1 + N2 - 1).pose.position.x;
  const double y0 = traj.at(N1 + N2 - 1).pose.position.y;
  for (size_t k = 1; k <= N3; ++k) {
    const double t = static_cast<double>(k) / static_cast<double>(N3);
    const size_t i = N1 + N2 + k - 1;
    traj.at(i).pose.position.x = x0;
    traj.at(i).pose.position.y = y0 + t * L2;
    set_yaw(traj.at(i), M_PI / 2.0);
  }

  // NOTE: check the "input_points_interval [m]" constant in the Simulink
  // velocity_smoother_node block and set this to the same value.
  const double points_interval = 0.1;

  const auto out = smoother->applySteeringRateLimit(traj, false, points_interval);

  const size_t idx_dist = static_cast<size_t>(
    std::max(static_cast<int>(params.curvature_calculation_distance / points_interval), 1));
  const auto kappa =
    autoware::velocity_smoother::trajectory_utils::calcTrajectoryCurvatureFrom3Points(
      out, idx_dist);

  std::ofstream f("/tmp/cpp_90deg_critical.csv");
  f << "i,segment,x,y,spacing,kappa_autoware,front_rad,rear_rad,rr_effective,velocity\n";
  for (size_t i = 0; i < out.size(); ++i) {
    const double fr = out.at(i).front_wheel_angle_rad;
    const double re = out.at(i).rear_wheel_angle_rad;
    const double sp =
      (i == 0) ? 0.0
               : std::hypot(
                   out.at(i).pose.position.x - out.at(i - 1).pose.position.x,
                   out.at(i).pose.position.y - out.at(i - 1).pose.position.y);
    const char * seg = (i < N1) ? "straight1" : ((i < N1 + N2) ? "arc" : "straight2");
    f << i << "," << seg << "," << out.at(i).pose.position.x << ","
      << out.at(i).pose.position.y << "," << sp << "," << kappa.at(i) << "," << fr << ","
      << re << "," << (std::fabs(fr) > 1e-9 ? re / fr : 0.0) << ","
      << out.at(i).longitudinal_velocity_mps << "\n";
  }
  f.close();
  std::cout << "wrote /tmp/cpp_90deg_critical.csv (" << out.size() << " rows)\n";

  EXPECT_EQ(out.size(), N);
}


int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
