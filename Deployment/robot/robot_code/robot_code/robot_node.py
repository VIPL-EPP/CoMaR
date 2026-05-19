import os
import sys
sys.path.append(f'{os.environ["HOME"]}/anaconda3/envs/viplanner/lib/python3.8/site-packages/')


import rclpy
from rclpy.node import Node
import message_filters

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Int8
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

import cv2
from cv_bridge import CvBridge
import requests
import json
import base64
import threading


class VLNNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')

        # --- 服务端地址 ---
        self.server_url = "http://118.180.19.242:5000/process"

        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- 发布者 ---
        self.path_pub = self.create_publisher(Path, '/viplanner/path/world_frame', 1)
        self.stop_pub = self.create_publisher(Int8, '/stop', 1)


        # --- 订阅 Topic ---
        rgb_topic = '/camera/camera/color/image_raw'
        depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
        odom_topic = '/go2_base/odometry'

        self.get_logger().info(f"Subscribing to: {rgb_topic}, {depth_topic}, {odom_topic}")

        self.sub_rgb = message_filters.Subscriber(self, Image, rgb_topic)
        self.sub_depth = message_filters.Subscriber(self, Image, depth_topic)
        self.sub_odom = message_filters.Subscriber(self, Odometry, odom_topic)

        # --- 时间同步 ---
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_rgb, self.sub_depth, self.sub_odom],
            queue_size=50,
            slop=0.1,
            allow_headerless=False
        )
        self.ts.registerCallback(self.sync_store_callback)

        # --- 最新数据缓存 & 状态 ---
        self._lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_odom = None
        self.have_data = False

        self.request_in_flight = False  # 是否有 HTTP 请求在等待响应
        self.pending_send = False       # 等待期间是否有新数据到来

        self.get_logger().info(f"Node Initialized. Target Server: {self.server_url}")

    def publish_path(self, waypoints, header=Header()):
        path_msg = Path()
        path_msg.header.stamp = header.stamp
        path_msg.header.frame_id = 'world'
        import time
        t0 = time.perf_counter()
        for waypoint in waypoints:
            pose = PoseStamped()
            # self.get_logger().info(waypoints)
            pose.pose.position.x = float(waypoint[0])
            pose.pose.position.y = float(waypoint[1])
            pose.pose.position.z = float(waypoint[2])
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def _extract_subgoal(self, result_data: dict):
        """
        直接就是 subgoal
        """
        if not isinstance(result_data, dict):
            return None

        time_step = None
        status = None
        subgoal = None
        # self.get_logger().info(f"Result Data: {result_data}")
        waypoints = result_data["subgoal"]['position']
        time_step = result_data["timestamp"]
        status = result_data["status"]

        return waypoints, status, time_step

    # =========================
    # 1) 同步回调：更新缓存 + 触发发送
    # =========================
    def sync_store_callback(self, rgb_msg, depth_msg, odom_msg):
        start_send = False

        with self._lock:
            self.latest_rgb = rgb_msg
            self.latest_depth = depth_msg
            self.latest_odom = odom_msg
            self.have_data = True

            # 若正在等待响应，只标记“有新数据”
            if self.request_in_flight:
                self.pending_send = True
                # self.get_logger().info("New data arrived while request in flight; marked for pending send.")
            else:
                # 没有在途请求：立刻发
                start_send = True
                self.request_in_flight = True
                self.pending_send = False

        if start_send:
            self._spawn_http_thread()

    def _spawn_http_thread(self):
        t = threading.Thread(target=self._send_latest_payload, daemon=True)
        t.start()

    # =========================
    # 2) HTTP 发送线程
    # =========================
    def _send_latest_payload(self):
        # 取“发送瞬间”的最新缓存快照
        with self._lock:
            rgb_msg = self.latest_rgb
            depth_msg = self.latest_depth
            odom_msg = self.latest_odom

        # 防御式检查
        if rgb_msg is None or depth_msg is None or odom_msg is None:
            self._finish_request()
            return

        try:
            timestamp_str = f"{rgb_msg.header.stamp.sec}.{rgb_msg.header.stamp.nanosec:09d}"

            # --- 1) RGB -> JPG Base64 ---
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            _, rgb_encoded = cv2.imencode('.jpg', cv_rgb)
            rgb_base64 = base64.b64encode(rgb_encoded.tobytes()).decode('utf-8')

            # --- 2) Depth -> PNG Base64 ---
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            _, depth_encoded = cv2.imencode('.png', cv_depth)
            depth_base64 = base64.b64encode(depth_encoded.tobytes()).decode('utf-8')

            # --- 3) Odom -> dict ---
            odom_dict = self.parse_odom_to_dict(odom_msg, timestamp_str)

            payload = {
                "timestamp": timestamp_str,
                "rgb_image": rgb_base64,
                "depth_image": depth_base64,
                "odometry": odom_dict
            }

            self.get_logger().info(f"Sending request to {self.server_url} at {timestamp_str}...")

            # 按你要求：等待响应（不设超时）
            response = requests.post(self.server_url, json=payload, timeout=None)

            msg = Int8()
            msg.data = 0

            if response.status_code == 200:
                result_data = response.json()
                waypoints, status, _ = self._extract_subgoal(result_data)
                self.get_logger().info(f"Status: {status}.")
                if status == "keep_subgoal":
                    self.get_logger().info("Status indicates to keep current subgoal; not publishing new path.")
                elif status == "observe_rotate":
                    msg.data = 1
                    self.stop_pub.publish(msg)
                    self.publish_path(waypoints, header=rgb_msg.header)
                    self.get_logger().info(f"Published Observe & Rotate Path: {waypoints}.")
                elif status == "new_subgoal":
                    msg.data = 0
                    self.stop_pub.publish(msg)
                    self.publish_path(waypoints, header=rgb_msg.header)
                    self.get_logger().info(f"Published Subgoal: {waypoints}.")
                elif status == "stop":
                    msg.data = 2
                    self.stop_pub.publish(msg)
                    self.get_logger().info("Status indicates to stop; no further path will be published.")
            else:
                self.get_logger().warn(f"Server returned stop code: {response.status_code}")

        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"HTTP Request failed: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing data: {str(e)}")
        finally:
            self._finish_request()

    # =========================
    # 3) 结束请求：如有新数据，立即再发最新
    # =========================
    def _finish_request(self):
        resend = False

        with self._lock:
            self.request_in_flight = False
            if self.pending_send and self.have_data:
                # 等待期间有新数据 -> 立刻补发最新
                self.pending_send = False
                self.request_in_flight = True
                resend = True

        if resend:
            self._spawn_http_thread()

    # =========================
    # 4) Odom 转 dict
    # =========================
    def parse_odom_to_dict(self, odom_msg, timestamp_str):
        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation
        vel_lin = odom_msg.twist.twist.linear
        vel_ang = odom_msg.twist.twist.angular

        return {
            "timestamp": timestamp_str,
            "frame_id": odom_msg.header.frame_id,
            "position": {"x": pos.x, "y": pos.y, "z": pos.z},
            "orientation": {"x": ori.x, "y": ori.y, "z": ori.z, "w": ori.w},
            "linear_velocity": {"x": vel_lin.x, "y": vel_lin.y, "z": vel_lin.z},
            "angular_velocity": {"x": vel_ang.x, "y": vel_ang.y, "z": vel_ang.z}
        }


def main(args=None):
    rclpy.init(args=args)
    node = VLNNode()
    try:
        # 这里用普通 spin 就够了（HTTP 在独立线程，不会阻塞回调）
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
