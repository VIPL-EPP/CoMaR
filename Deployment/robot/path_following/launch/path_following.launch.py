import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('path_following'),
        'config',
        'unitree_go2.yaml'
    )

    return LaunchDescription([
        Node(
            package='path_following',
            executable='path_following_node',
            name='path_following',
            output='screen',
            parameters=[config_file]
        )
    ])

if __name__ == '__main__':
    generate_launch_description()
