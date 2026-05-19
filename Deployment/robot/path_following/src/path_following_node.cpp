#include <rclcpp/rclcpp.hpp>
#include "path_follower.h"

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathFollower>());
    rclcpp::shutdown();
    return 0;
}