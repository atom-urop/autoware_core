// Copyright 2026
//
// Test for 4WS steering extension in velocity smoother

#include "autoware/velocity_smoother/smoother/jerk_filtered_smoother.hpp"
#include "autoware/velocity_smoother/smoother/smoother_base.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>

#include <tf2/LinearMath/Quaternion.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>


namespace
{

using TrajectoryPoint = autoware_planning_msgs::msg::TrajectoryPoint;
using TrajectoryPoints = std::vector<TrajectoryPoint>;

using autoware::velocity_smoother::JerkFilteredSmoother;
using autoware::velocity_smoother::SmootherBase;


// ------------------------------------------------
// Helper functions
// ------------------------------------------------

TrajectoryPoints createCircularTrajectory(
  const double radius,
  const double velocity,
  const int num_points)
{
  TrajectoryPoints trajectory;

  const double angle_step =
    M_PI / 2.0 / (num_points - 1);


  for (int i = 0; i < num_points; ++i) {

    const double angle =
      i * angle_step;


    TrajectoryPoint point;


    // Position
    point.pose.position.x =
      radius * std::sin(angle);

    point.pose.position.y =
      radius * (1.0 - std::cos(angle));


    // Orientation (yaw)
    tf2::Quaternion q;
    q.setRPY(
      0.0,
      0.0,
      angle);

    point.pose.orientation.x = q.x();
    point.pose.orientation.y = q.y();
    point.pose.orientation.z = q.z();
    point.pose.orientation.w = q.w();


    // Velocity
    point.longitudinal_velocity_mps =
      velocity;


    trajectory.push_back(point);
  }


  return trajectory;
}

void checkSteeringOutput(
  const TrajectoryPoints & output,
  const bool expect_rear,
  const double rear_ratio)
{
  bool has_front = false;
  bool has_rear = false;

  for (const auto & point : output) {
    std::cout
      << "front = " << point.front_wheel_angle_rad
      << " rear = " << point.rear_wheel_angle_rad
      << std::endl;
      
    if (std::fabs(point.front_wheel_angle_rad) > 1e-3) {
      has_front = true;
    }

    if (std::fabs(point.rear_wheel_angle_rad) > 1e-3) {
      has_rear = true;
    }

    if (expect_rear) {

      EXPECT_NEAR(
        point.rear_wheel_angle_rad,
        rear_ratio * point.front_wheel_angle_rad,
        1e-3);

      // 4WS counter phase check
      EXPECT_LT(
        point.front_wheel_angle_rad *
        point.rear_wheel_angle_rad,
        0.0);

    } else {

      EXPECT_NEAR(
        point.rear_wheel_angle_rad,
        0.0,
        1e-3);
    }
  }

  EXPECT_TRUE(has_front);

  if (expect_rear) {
    EXPECT_TRUE(has_rear);
  } else {
    EXPECT_FALSE(has_rear);
  }
}

// ----------------
// Test fixture
// ----------------
class TestSteeringRate2WS : public ::testing::Test
{
protected:

  void SetUp() override
  {
    rclcpp::init(0, nullptr);

    auto node_options = rclcpp::NodeOptions{};

    // Parameters required by SmootherBase constructor

    node_options.append_parameter_override(
      "normal.max_acc",
      2.0);

    node_options.append_parameter_override(
      "normal.min_acc",
      -2.0);

    node_options.append_parameter_override(
      "stop_decel",
      -2.0);

    node_options.append_parameter_override(
      "normal.max_jerk",
      1.0);

    node_options.append_parameter_override(
      "normal.min_jerk",
      -1.0);

    node_options.append_parameter_override(
      "min_decel_for_lateral_acc_lim_filter",
      -1.0);

    node_options.append_parameter_override(
      "resample_ds",
      0.1);

    node_options.append_parameter_override(
      "curvature_threshold",
      0.01);

    node_options.append_parameter_override(
      "lateral_acceleration_limits",
      std::vector<double>{1.0});

    node_options.append_parameter_override(
      "velocity_thresholds",
      std::vector<double>{0.0});


    node_options.append_parameter_override(
      "steering_angle_rate_limits",
      std::vector<double>{30.0, 20.0, 10.0});


    node_options.append_parameter_override(
      "curvature_calculation_distance",
      1.0);

    node_options.append_parameter_override(
      "decel_distance_before_curve",
      5.0);

    node_options.append_parameter_override(
      "decel_distance_after_curve",
      5.0);

    node_options.append_parameter_override(
      "min_curve_velocity",
      1.0);


    // Resampling parameters

    node_options.append_parameter_override(
      "max_trajectory_length",
      200.0);

    node_options.append_parameter_override(
      "min_trajectory_length",
      10.0);

    node_options.append_parameter_override(
      "resample_time",
      1.0);

    node_options.append_parameter_override(
      "dense_resample_dt",
      0.1);

    node_options.append_parameter_override(
      "dense_min_interval_distance",
      0.1);

    node_options.append_parameter_override(
      "sparse_resample_dt",
      0.5);

    node_options.append_parameter_override(
      "sparse_min_interval_distance",
      1.0);
    
    // JerkFilteredSmoother parameters

    node_options.append_parameter_override(
      "jerk_weight",
      10.0);

    node_options.append_parameter_override(
      "over_v_weight",
      100000.0);

    node_options.append_parameter_override(
      "over_a_weight",
      5000.0);

    node_options.append_parameter_override(
      "over_j_weight",
      2000.0);

    node_options.append_parameter_override(
      "jerk_filter_ds",
      0.1);

    node_options.append_parameter_override(
      "enable_4ws",
      false);

    node_options.append_parameter_override(
      "rear_steering_ratio",
      -0.3);

    auto node =
      std::make_shared<rclcpp::Node>(
        "test_steering_rate_2ws_node",
        node_options);

    debug_processing_time_detail_ =
      node->create_publisher<autoware_utils_debug::ProcessingTimeDetail>(
        "~/debug/processing_time",
        1);

    auto time_keeper =
      std::make_shared<autoware_utils_debug::TimeKeeper>(
        debug_processing_time_detail_);

    smoother_base =
      std::make_shared<JerkFilteredSmoother>(
        *node,
        time_keeper);

    const auto params =
      std::dynamic_pointer_cast<SmootherBase>(
        smoother_base)
        ->getBaseParam();

    EXPECT_FALSE(params.enable_4ws);

    EXPECT_DOUBLE_EQ(
      params.rear_steering_ratio,
      -0.3);

    std::dynamic_pointer_cast<SmootherBase>(
      smoother_base)
      ->setWheelBase(2.7);
  }


  void TearDown() override
  {
    rclcpp::shutdown();
  }


  std::shared_ptr<JerkFilteredSmoother> smoother_base;

  rclcpp::Publisher<
    autoware_utils_debug::ProcessingTimeDetail>::SharedPtr
      debug_processing_time_detail_;
};


class TestSteeringRate4WS : public ::testing::Test
{
protected:

  void SetUp() override
  {
    rclcpp::init(0, nullptr);


    auto node_options = rclcpp::NodeOptions{};


    // Parameters required by SmootherBase constructor

    node_options.append_parameter_override(
      "normal.max_acc",
      2.0);

    node_options.append_parameter_override(
      "normal.min_acc",
      -2.0);

    node_options.append_parameter_override(
      "stop_decel",
      -2.0);

    node_options.append_parameter_override(
      "normal.max_jerk",
      1.0);

    node_options.append_parameter_override(
      "normal.min_jerk",
      -1.0);

    node_options.append_parameter_override(
      "min_decel_for_lateral_acc_lim_filter",
      -1.0);

    node_options.append_parameter_override(
      "resample_ds",
      0.1);

    node_options.append_parameter_override(
      "curvature_threshold",
      0.01);

    node_options.append_parameter_override(
      "lateral_acceleration_limits",
      std::vector<double>{1.0});

    node_options.append_parameter_override(
      "velocity_thresholds",
      std::vector<double>{0.0});


    node_options.append_parameter_override(
      "steering_angle_rate_limits",
      std::vector<double>{30.0, 20.0, 10.0});


    node_options.append_parameter_override(
      "curvature_calculation_distance",
      1.0);

    node_options.append_parameter_override(
      "decel_distance_before_curve",
      5.0);

    node_options.append_parameter_override(
      "decel_distance_after_curve",
      5.0);

    node_options.append_parameter_override(
      "min_curve_velocity",
      1.0);


    // Resampling parameters

    node_options.append_parameter_override(
      "max_trajectory_length",
      200.0);

    node_options.append_parameter_override(
      "min_trajectory_length",
      10.0);

    node_options.append_parameter_override(
      "resample_time",
      1.0);

    node_options.append_parameter_override(
      "dense_resample_dt",
      0.1);

    node_options.append_parameter_override(
      "dense_min_interval_distance",
      0.1);

    node_options.append_parameter_override(
      "sparse_resample_dt",
      0.5);

    node_options.append_parameter_override(
      "sparse_min_interval_distance",
      1.0);
    
    // JerkFilteredSmoother parameters

    node_options.append_parameter_override(
      "jerk_weight",
      10.0);

    node_options.append_parameter_override(
      "over_v_weight",
      100000.0);

    node_options.append_parameter_override(
      "over_a_weight",
      5000.0);

    node_options.append_parameter_override(
      "over_j_weight",
      2000.0);

    node_options.append_parameter_override(
      "jerk_filter_ds",
      0.1);

    // 4WS parameters

    node_options.append_parameter_override(
      "enable_4ws",
      true);

    node_options.append_parameter_override(
      "rear_steering_ratio",
      -0.3);

    auto node =
      std::make_shared<rclcpp::Node>(
        "test_steering_rate_4ws_node",
        node_options);

    debug_processing_time_detail_ =
      node->create_publisher<autoware_utils_debug::ProcessingTimeDetail>(
        "~/debug/processing_time",
        1);

    auto time_keeper =
      std::make_shared<autoware_utils_debug::TimeKeeper>(
        debug_processing_time_detail_);


    smoother_base =
      std::make_shared<JerkFilteredSmoother>(
        *node,
        time_keeper);

    const auto params =
      std::dynamic_pointer_cast<SmootherBase>(
        smoother_base)
        ->getBaseParam();


    EXPECT_TRUE(params.enable_4ws);

    EXPECT_DOUBLE_EQ(
      params.rear_steering_ratio,
      -0.3);
    
    std::dynamic_pointer_cast<SmootherBase>(
      smoother_base)
      ->setWheelBase(2.7);
  }


  void TearDown() override
  {
    rclcpp::shutdown();
  }


  std::shared_ptr<JerkFilteredSmoother> smoother_base;


  rclcpp::Publisher<
    autoware_utils_debug::ProcessingTimeDetail>::SharedPtr
      debug_processing_time_detail_;
};


// ------------------------------------------------
// Tests
// ------------------------------------------------

TEST_F(TestSteeringRate4WS, SteeringRate4WS)
{
  auto trajectory =
    createCircularTrajectory(
      10.0,
      5.0,
      20);

  const auto output =
    smoother_base->applySteeringRateLimit(
      trajectory,
      false,
      0.5);

  ASSERT_EQ(output.size(), trajectory.size());

  checkSteeringOutput(
    output,
    true,
    -0.3);
}

TEST_F(TestSteeringRate2WS, SteeringRate2WS)
{
  auto trajectory =
    createCircularTrajectory(
      10.0,
      5.0,
      20);

  const auto output =
    smoother_base->applySteeringRateLimit(
      trajectory,
      false,
      0.5);

  ASSERT_EQ(output.size(), trajectory.size());

  checkSteeringOutput(
    output,
    false,
    0.0);
}

}