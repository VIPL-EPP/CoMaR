// Reference from: https://github.com/HongbiaoZ/autonomous_exploration_development_environment/blob/noetic/src/local_planner/src/pathFollower.cpp

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
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

// #include <unitree/robot/go2/sport/sport_client.hpp>
#include "unitree_go/msg/sport_mode_state.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

const double PI = 3.1415926;
const double EPS = 1e-5;

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
std::string commandTopic = "/cmd_vel";
std::string baseFrame = "base";

class PathFollower : public rclcpp::Node
{
public:
    PathFollower() : Node("path_follower")
    {
        this->declare_parameter<double>("sensorOffsetX", sensorOffsetX);
        this->declare_parameter<double>("sensorOffsetY", sensorOffsetY);
        this->declare_parameter<int>("pubSkipNum", pubSkipNum);
        this->declare_parameter<bool>("twoWayDrive", twoWayDrive);
        this->declare_parameter<double>("lookAheadDis", lookAheadDis);
        this->declare_parameter<double>("yawRateGain", yawRateGain);
        this->declare_parameter<double>("stopYawRateGain", stopYawRateGain);
        this->declare_parameter<double>("maxYawRate", maxYawRate);
        this->declare_parameter<double>("maxSpeed", maxSpeed);
        this->declare_parameter<double>("maxAccel", maxAccel);
        this->declare_parameter<double>("switchTimeThre", switchTimeThre);
        this->declare_parameter<double>("dirDiffThre", dirDiffThre);
        this->declare_parameter<double>("stopDisThre", stopDisThre);
        this->declare_parameter<double>("slowDwnDisThre", slowDwnDisThre);
        this->declare_parameter<bool>("useInclRateToSlow", useInclRateToSlow);
        this->declare_parameter<double>("inclRateThre", inclRateThre);
        this->declare_parameter<double>("slowRate1", slowRate1);
        this->declare_parameter<double>("slowRate2", slowRate2);
        this->declare_parameter<double>("slowTime1", slowTime1);
        this->declare_parameter<double>("slowTime2", slowTime2);
        this->declare_parameter<bool>("useInclToStop", useInclToStop);
        this->declare_parameter<double>("inclThre", inclThre);
        this->declare_parameter<double>("stopTime", stopTime);
        this->declare_parameter<bool>("noRotAtStop", noRotAtStop);
        this->declare_parameter<bool>("noRotAtGoal", noRotAtGoal);
        this->declare_parameter<bool>("autonomyMode", autonomyMode);
        this->declare_parameter<double>("autonomySpeed", autonomySpeed);
        this->declare_parameter<double>("joyToSpeedDelay", joyToSpeedDelay);
        this->declare_parameter<std::string>("odomTopic", odomTopic);
        this->declare_parameter<std::string>("pathTopic", pathTopic);
        this->declare_parameter<std::string>("commandTopic", commandTopic);
        this->declare_parameter<std::string>("baseFrame", baseFrame);
        this->declare_parameter<bool>("baseInverted", baseInverted);
        this->declare_parameter<double>("relativeMode", relativeMode);

        this->get_parameter("sensorOffsetX", sensorOffsetX);
        this->get_parameter("sensorOffsetY", sensorOffsetY);
        this->get_parameter("pubSkipNum", pubSkipNum);
        this->get_parameter("twoWayDrive", twoWayDrive);
        this->get_parameter("lookAheadDis", lookAheadDis);
        this->get_parameter("yawRateGain", yawRateGain);
        this->get_parameter("stopYawRateGain", stopYawRateGain);
        this->get_parameter("maxYawRate", maxYawRate);
        this->get_parameter("maxSpeed", maxSpeed);
        this->get_parameter("maxAccel", maxAccel);
        this->get_parameter("switchTimeThre", switchTimeThre);
        this->get_parameter("dirDiffThre", dirDiffThre);
        this->get_parameter("stopDisThre", stopDisThre);
        this->get_parameter("slowDwnDisThre", slowDwnDisThre);
        this->get_parameter("useInclRateToSlow", useInclRateToSlow);
        this->get_parameter("inclRateThre", inclRateThre);
        this->get_parameter("slowRate1", slowRate1);
        this->get_parameter("slowRate2", slowRate2);
        this->get_parameter("slowTime1", slowTime1);
        this->get_parameter("slowTime2", slowTime2);
        this->get_parameter("useInclToStop", useInclToStop);
        this->get_parameter("inclThre", inclThre);
        this->get_parameter("stopTime", stopTime);
        this->get_parameter("noRotAtStop", noRotAtStop);
        this->get_parameter("noRotAtGoal", noRotAtGoal);
        this->get_parameter("autonomyMode", autonomyMode);
        this->get_parameter("autonomySpeed", autonomySpeed);
        this->get_parameter("joyToSpeedDelay", joyToSpeedDelay);
        this->get_parameter("odomTopic", odomTopic);
        this->get_parameter("pathTopic", pathTopic);
        this->get_parameter("commandTopic", commandTopic);
        this->get_parameter("baseFrame", baseFrame);
        this->get_parameter("baseInverted", baseInverted);
        this->get_parameter("relativeMode", relativeMode);

        // log offset info
        //  RCLCPP_INFO(this->get_logger(), "sensorOffsetX: %f", sensorOffsetX);
        lookAheadDis += std::hypot(sensorOffsetX, sensorOffsetY);

        // subOdom_ = this->create_subscription<nav_msgs::msg::Odometry>(
        //     odomTopic, 5, std::bind(&PathFollower::odomCallback, this, _1));
        subOdom_ = this->create_subscription<unitree_go::msg::SportModeState>(
            odomTopic, 10, std::bind(&PathFollower::odomCallback, this, _1));
        subPath_ = this->create_subscription<nav_msgs::msg::Path>(
            pathTopic, 5, std::bind(&PathFollower::pathCallback, this, _1));
        subJoystick_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 5, std::bind(&PathFollower::joystickCallback, this, _1));
        subSpeed_ = this->create_subscription<std_msgs::msg::Float32>(
            "/speed", 5, std::bind(&PathFollower::speedCallback, this, _1));
        subStop_ = this->create_subscription<std_msgs::msg::Int8>(
            "/stop", 5, std::bind(&PathFollower::stopCallback, this, _1));

        pubCmd_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
            commandTopic, 5);
        pubReq_ = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 10);

        cmd_vel.header.frame_id = baseFrame;
        if (autonomyMode)
        {
            joySpeed = autonomySpeed / maxSpeed;
            if (joySpeed < 0)
                joySpeed = 0;
            else if (joySpeed > 1.0)
                joySpeed = 1.0;
        }
        // sport_client.SetTimeout(10.0f);
        // sport_client.Init();
        timer_ = this->create_wall_timer(10ms, std::bind(&PathFollower::controlLoopCallback, this));
    }

private:
    // void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odomIn)
    // {
    //     odomTime = odomIn->header.stamp.sec + odomIn->header.stamp.nanosec * 1e-9;

    //     double roll, pitch, yaw;
    //     geometry_msgs::msg::Quaternion geoQuat = odomIn->pose.pose.orientation;
    //     tf2::Quaternion tfQuat(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w);
    //     tf2::Matrix3x3(tfQuat).getRPY(roll, pitch, yaw);

    //     vehicleRoll = roll;
    //     vehiclePitch = pitch;
    //     vehicleYaw = yaw;
    //     vehicleX = odomIn->pose.pose.position.x;
    //     vehicleY = odomIn->pose.pose.position.y;
    //     vehicleZ = odomIn->pose.pose.position.z;

    //     if ((fabs(roll) > inclThre * PI / 180.0 || fabs(pitch) > inclThre * PI / 180.0) && useInclToStop)
    //     {
    //     stopInitTime = odomTime;
    //     }
    //     RCLCPP_INFO(this->get_logger(), "Received odom message");
    // }

    void odomCallback(const unitree_go::msg::SportModeState::SharedPtr odomIn)
    {
        // odomTime = odomIn->stamp.sec + odomIn->stamp.nanosec * 1e-9;
        // odomtime should be now
        odomTime = this->get_clock()->now().seconds();

        double roll, pitch, yaw;
        // geometry_msgs::msg::Quaternion geoQuat = odomIn->pose.pose.orientation;
        // tf2::Quaternion tfQuat(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w);
        // tf2::Matrix3x3(tfQuat).getRPY(roll, pitch, yaw);
        roll = odomIn->imu_state.rpy[0];
        pitch = odomIn->imu_state.rpy[1];
        yaw = odomIn->imu_state.rpy[2];

        vehicleRoll = roll;
        vehiclePitch = pitch;
        vehicleYaw = yaw;
        vehicleX = odomIn->position[0];
        vehicleY = odomIn->position[1];
        vehicleZ = odomIn->position[2];

        if ((fabs(roll) > inclThre * PI / 180.0 || fabs(pitch) > inclThre * PI / 180.0) && useInclToStop)
        {
            stopInitTime = odomTime;
        }
        // RCLCPP_INFO(this->get_logger(), "Received odom message");
    }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr pathIn)
    {
        int pathSize = pathIn->poses.size();
        path.poses.resize(pathSize);
        for (int i = 0; i < pathSize; i++)
        {
            path.poses[i].pose.position.x = pathIn->poses[i].pose.position.x;
            path.poses[i].pose.position.y = pathIn->poses[i].pose.position.y;
            path.poses[i].pose.position.z = pathIn->poses[i].pose.position.z;
        }

        if (relativeMode)
        {
            vehicleXRec = vehicleX;
            vehicleYRec = vehicleY;
            vehicleZRec = vehicleZ;
            vehicleRollRec = vehicleRoll;
            vehiclePitchRec = vehiclePitch;
            vehicleYawRec = vehicleYaw;
        }
        else
        {
            vehicleXRec = 0;
            vehicleYRec = 0;
            vehicleZRec = 0;
            vehicleRollRec = 0;
            vehiclePitchRec = 0;
            vehicleYawRec = 0;
        }

        pathPointID = 0;
        pathInit = true;
        // RCLCPP_INFO(this->get_logger(), "Received path message");
    }

    void joystickCallback(const sensor_msgs::msg::Joy::SharedPtr joy)
    {
        joyTime = this->get_clock()->now().seconds();

        joySpeedRaw = std::sqrt(joy->axes[3] * joy->axes[3] + joy->axes[4] * joy->axes[4]);
        joySpeed = joySpeedRaw;
        if (joySpeed > 1.0)
            joySpeed = 1.0;
        if (joy->axes[4] == 0)
            joySpeed = 0;
        joyYaw = joy->axes[3];
        if (joySpeed == 0 && noRotAtStop)
            joyYaw = 0;

        if (joy->axes[4] < 0 && !twoWayDrive)
        {
            joySpeed = 0;
            joyYaw = 0;
        }

        if (joy->axes[2] > -0.1)
        {
            autonomyMode = false;
        }
        else
        {
            autonomyMode = true;
        }
        // RCLCPP_INFO(this->get_logger(), "Received joystick message");
    }

    void speedCallback(const std_msgs::msg::Float32::SharedPtr speed)
    {
        double speedTime = this->get_clock()->now().seconds();

        if (autonomyMode && speedTime - joyTime > joyToSpeedDelay && joySpeedRaw == 0)
        {
            joySpeed = speed->data / maxSpeed;

            if (joySpeed < 0)
                joySpeed = 0;
            else if (joySpeed > 1.0)
                joySpeed = 1.0;
        }
        // RCLCPP_INFO(this->get_logger(), "Received speed message");
    }

    void stopCallback(const std_msgs::msg::Int8::SharedPtr stop)
    {
        safetyStop = stop->data;
        // RCLCPP_INFO(this->get_logger(), "Received stop message");
    }

    void controlLoopCallback()
    {
        if (pathInit)
        {
            float vehicleXRel = cos(vehicleYawRec) * (vehicleX - vehicleXRec) + sin(vehicleYawRec) * (vehicleY - vehicleYRec);
            float vehicleYRel = -sin(vehicleYawRec) * (vehicleX - vehicleXRec) + cos(vehicleYawRec) * (vehicleY - vehicleYRec);

            int pathSize = path.poses.size();
            float endDisX = path.poses[pathSize - 1].pose.position.x - vehicleXRel;
            float endDisY = path.poses[pathSize - 1].pose.position.y - vehicleYRel;
            float endDis = sqrt(endDisX * endDisX + endDisY * endDisY);

            float disX, disY, dis;
            while (pathPointID < pathSize - 1)
            {
                disX = path.poses[pathPointID].pose.position.x - vehicleXRel;
                disY = path.poses[pathPointID].pose.position.y - vehicleYRel;
                dis = sqrt(disX * disX + disY * disY);
                if (dis < lookAheadDis)
                {
                    pathPointID++;
                }
                else
                {
                    break;
                }
            }

            disX = path.poses[pathPointID].pose.position.x - vehicleXRel;
            disY = path.poses[pathPointID].pose.position.y - vehicleYRel;
            // RCLCPP_INFO(this->get_logger(), "pathSize: %d", pathSize);
            // RCLCPP_INFO(this->get_logger(), "vehicleXRel: %f, vehicleYRel: %f", vehicleXRel, vehicleYRel);
            // RCLCPP_INFO(this->get_logger(), "aheadDis: %f", lookAheadDis);
            // RCLCPP_INFO(this->get_logger(), "endDis: %f", endDis);
            // RCLCPP_INFO(this->get_logger(), "dis: %f", dis);
            // RCLCPP_INFO(this->get_logger(), "pathPointID: %d", pathPointID);
            // RCLCPP_INFO(this->get_logger(), "tgX: %f, tgY: %f", path.poses[pathPointID].pose.position.x, path.poses[pathPointID].pose.position.y);
            // RCLCPP_INFO(this->get_logger(), "vehicleX: %f, vehicleY: %f", vehicleX, vehicleY);
            // RCLCPP_INFO(this->get_logger(), "disX: %f, disY: %f", disX, disY);
            dis = sqrt(disX * disX + disY * disY);
            float pathDir = atan2(disY, disX);

            double dirDiff = vehicleYaw - vehicleYawRec - pathDir;
            // RCLCPP_INFO(this->get_logger(), "pathDir: %f, : %f", disX, disY);
            if (dirDiff > PI)
                dirDiff -= 2 * PI;
            else if (dirDiff < -PI)
                dirDiff += 2 * PI;
            if (dirDiff > PI)
                dirDiff -= 2 * PI;
            else if (dirDiff < -PI)
                dirDiff += 2 * PI;

            if (twoWayDrive)
            {
                double time = this->get_clock()->now().seconds();
                if (fabs(dirDiff) > PI / 2 && navFwd && time - switchTime > switchTimeThre)
                {
                    navFwd = false;
                    switchTime = time;
                }
                else if (fabs(dirDiff) < PI / 2 && !navFwd && time - switchTime > switchTimeThre)
                {
                    navFwd = true;
                    switchTime = time;
                }
            }

            float joySpeed2 = maxSpeed * joySpeed;
            if (!navFwd)
            {
                dirDiff += PI;
                if (dirDiff > PI)
                    dirDiff -= 2 * PI;
                joySpeed2 *= -1;
            }

            // Add momentum to dirDiff
            if (fabs(dirDiff) > dirDiffThre - EPS && dis > lookAheadDis + EPS)
            {
                if (lastDiffDir - dirDiff > PI)
                    dirDiff += 2 * PI;
                else if (lastDiffDir - dirDiff < -PI)
                    dirDiff -= 2 * PI;
                dirDiff = (1.0 - dirMomentum) * dirDiff + dirMomentum * lastDiffDir;
                dirDiff = std::max(std::min(dirDiff, PI - EPS), -PI + EPS);
                lastDiffDir = dirDiff;
            }
            else
            {
                lastDiffDir = 0.0;
            }

            if (fabs(vehicleSpeed) < 2.0 * maxAccel / 100.0)
                vehicleYawRate = -stopYawRateGain * dirDiff;
            else
                vehicleYawRate = -yawRateGain * dirDiff;

            if (vehicleYawRate > maxYawRate * PI / 180.0)
                vehicleYawRate = maxYawRate * PI / 180.0;
            else if (vehicleYawRate < -maxYawRate * PI / 180.0)
                vehicleYawRate = -maxYawRate * PI / 180.0;

            if (joySpeed2 == 0 && !autonomyMode)
            {
                vehicleYawRate = maxYawRate * joyYaw * PI / 180.0;
            }
            else if (pathSize <= 1 || (dis < stopDisThre && noRotAtGoal))
            {
                vehicleYawRate = 0;
            }

            if (pathSize <= 1)
            {
                joySpeed2 = 0;
            }
            else if (endDis / slowDwnDisThre < joySpeed)
            {
                joySpeed2 *= endDis / slowDwnDisThre;
            }

            float joySpeed3 = joySpeed2;
            if (odomTime < slowInitTime + slowTime1 && slowInitTime > 0)
                joySpeed3 *= slowRate1;
            else if (odomTime < slowInitTime + slowTime1 + slowTime2 && slowInitTime > 0)
                joySpeed3 *= slowRate2;

            if (fabs(dirDiff) < dirDiffThre && dis > stopDisThre)
            {
                if (vehicleSpeed < joySpeed3)
                    vehicleSpeed += maxAccel / 100.0;
                else if (vehicleSpeed > joySpeed3)
                    vehicleSpeed -= maxAccel / 100.0;
            }
            else
            {
                if (vehicleSpeed > 0)
                    vehicleSpeed -= maxAccel / 100.0;
                else if (vehicleSpeed < 0)
                    vehicleSpeed += maxAccel / 100.0;
            }

            if (odomTime < stopInitTime + stopTime && stopInitTime > 0)
            {
                vehicleSpeed = 0;
                vehicleYawRate = 0;
            }

            if (safetyStop >= 1)
                vehicleSpeed = 0;
            if (safetyStop >= 2)
                vehicleYawRate = 0;

            pubSkipCount--;
            // RCLCPP_INFO(this->get_logger(), "Control loop callback");
            if (pubSkipCount < 0)
            {
                // cmd_vel.header.stamp = ros::Time().fromSec(odomTime);
                // cmd_vel.header.stamp = rclcpp::Time(odomTime * 1e9);
                // if (fabs(vehicleSpeed) <= maxAccel / 100.0) cmd_vel.twist.linear.x = 0;
                // else if (baseInverted) cmd_vel.twist.linear.x = -vehicleSpeed;
                // else cmd_vel.twist.linear.x = vehicleSpeed;
                // cmd_vel.twist.angular.z = vehicleYawRate;
                // pubCmd_->publish(cmd_vel);
                // unitree sdk
                if (fabs(vehicleSpeed) <= maxAccel / 100.0)
                    vehicleSpeed = 0;
                else if (baseInverted)
                    vehicleSpeed = -vehicleSpeed;
                // else cmd_vel.twist.linear.x = vehicleSpeed;
                // sport_client.Move(vehicleSpeed, 0, vehicleYawRate);
                sportReq.Move(req, vehicleSpeed, 0, vehicleYawRate);
                // RCLCPP_INFO(this->get_logger(), "speed: %f, yaw: %f ", vehicleSpeed, vehicleYawRate);
                // Publish request messages
                pubReq_->publish(req);
                pubSkipCount = pubSkipNum;
            }
        }
    }

    nav_msgs::msg::Path path;
    geometry_msgs::msg::TwistStamped cmd_vel; // TODO replace this with unitree sdk
    // rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom_;
    rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr subOdom_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subPath_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr subJoystick_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subSpeed_;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr subStop_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr pubCmd_;
    rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr pubReq_;
    rclcpp::TimerBase::SharedPtr timer_;

    // unitree::robot::go2::SportClient sport_client;
    unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
    SportClient sportReq;          // Unitree Go2 ROS2 sport client
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathFollower>());
    rclcpp::shutdown();
    return 0;
}