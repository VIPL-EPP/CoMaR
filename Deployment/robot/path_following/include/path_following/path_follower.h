#include <math.h>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int8.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joy.hpp>

#include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

const double PI = 3.1415926;
const double EPS = 1e-5;

class PathFollower : public rclcpp::Node
{
public:
    PathFollower();

private:
    void odomCallback(const unitree_go::msg::SportModeState::SharedPtr odomIn);
    void pathCallback(const nav_msgs::msg::Path::SharedPtr pathIn);
    void joystickCallback(const sensor_msgs::msg::Joy::SharedPtr joy);
    void speedCallback(const std_msgs::msg::Float32::SharedPtr speed);
    void stopCallback(const std_msgs::msg::Int8::SharedPtr stop);

    void controlLoopCallback();

    rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr subOdom_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subPath_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr subJoystick_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subSpeed_;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr subStop_;

    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr pubReq_;
    rclcpp::TimerBase::SharedPtr timer_;

    nav_msgs::msg::Path path;      // Path message
    unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
    SportClient sportReq;          // Unitree Go2 ROS2 sport client

    // Parameters
    double sensorOffsetX = 0;
    double sensorOffsetY = 0;
    int pubSkipNum = 1;
    int pubSkipCount = 0;
    bool twoWayDrive = true;
    double lookAheadDis = 0.5;
    double yawRateGain = 7.5;
    double stopYawRateGain = 7.5;
    double maxYawRate = 45.0;
    double maxSpeed = 1.0;
    double maxAccel = 1.0;
    double switchTimeThre = 1.0;
    double dirDiffThre = 0.1;
    double stopDisThre = 0.2;
    double slowDwnDisThre = 1.0;
    bool useInclRateToSlow = false;
    double inclRateThre = 120.0;
    double slowRate1 = 0.25;
    double slowRate2 = 0.5;
    double slowTime1 = 2.0;
    double slowTime2 = 2.0;
    bool useInclToStop = false;
    double inclThre = 45.0;
    double stopTime = 5.0;
    bool noRotAtStop = false;
    bool noRotAtGoal = true;
    bool autonomyMode = false;
    double autonomySpeed = 1.0;
    double joyToSpeedDelay = 2.0;
    bool relativeMode = false;

    float joySpeed = 0;
    float joySpeedRaw = 0;
    float joyYaw = 0;
    int safetyStop = 0;

    float vehicleX = 0;
    float vehicleY = 0;
    float vehicleZ = 0;
    float vehicleRoll = 0;
    float vehiclePitch = 0;
    float vehicleYaw = 0;

    float vehicleXRec = 0;
    float vehicleYRec = 0;
    float vehicleZRec = 0;
    float vehicleRollRec = 0;
    float vehiclePitchRec = 0;
    float vehicleYawRec = 0;

    float vehicleYawRate = 0;
    float vehicleSpeed = 0;

    double dirMomentum = 0.25;
    double lastDiffDir = 0;
    double odomTime = 0;
    double joyTime = 0;
    double slowInitTime = 0;
    double stopInitTime = false;
    int pathPointID = 0;
    bool pathInit = false;
    bool navFwd = true;
    double switchTime = 0;
    bool baseInverted = false;

    std::string odomTopic = "/sportmodestate";
    std::string pathTopic = "/viplanner/path/world";
    std::string baseFrame = "base_link";
};