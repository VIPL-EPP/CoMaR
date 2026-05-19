#include "path_follower.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

PathFollower::PathFollower() : Node("path_follower")
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
    this->declare_parameter<std::string>("baseFrame", baseFrame);
    this->declare_parameter<bool>("baseInverted", baseInverted);
    this->declare_parameter<bool>("relativeMode", relativeMode);

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

    pubReq_ = this->create_publisher<unitree_api::msg::Request>("/api/sport/request", 10);

    if (autonomyMode)
    {
        joySpeed = autonomySpeed / maxSpeed;
        if (joySpeed < 0)
            joySpeed = 0;
        else if (joySpeed > 1.0)
            joySpeed = 1.0;
    }
    timer_ = this->create_wall_timer(10ms, std::bind(&PathFollower::controlLoopCallback, this));
}

void PathFollower::odomCallback(const unitree_go::msg::SportModeState::SharedPtr odomIn)
{
    // odomTime = odomIn->stamp.sec + odomIn->stamp.nanosec * 1e-9;
    // odomtime should be now. the received stamp is not reliable (delay ≈58s)
    odomTime = this->get_clock()->now().seconds();

    double roll, pitch, yaw;
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
}

void PathFollower::pathCallback(const nav_msgs::msg::Path::SharedPtr pathIn)
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
}

void PathFollower::joystickCallback(const sensor_msgs::msg::Joy::SharedPtr joy)
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
}

void PathFollower::speedCallback(const std_msgs::msg::Float32::SharedPtr speed)
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
}

void PathFollower::stopCallback(const std_msgs::msg::Int8::SharedPtr stop)
{
    safetyStop = stop->data;
}

void PathFollower::controlLoopCallback()
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
        if (pubSkipCount < 0)
        {
            // unitree sdk
            if (fabs(vehicleSpeed) <= maxAccel / 100.0)
                vehicleSpeed = 0;
            else if (baseInverted)
                vehicleSpeed = -vehicleSpeed;
            sportReq.Move(req, vehicleSpeed, 0, vehicleYawRate);
            // Publish request messages
            pubReq_->publish(req);
            pubSkipCount = pubSkipNum;
        }
    }
}