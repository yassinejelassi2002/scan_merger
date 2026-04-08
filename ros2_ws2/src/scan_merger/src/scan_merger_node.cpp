#include <deque>
#include <filesystem>
#include <string>

#include <Eigen/Dense>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

class ScanMergerNode : public rclcpp::Node
{
public:
  ScanMergerNode()
  : Node("scan_merger_node")
  {
    this->declare_parameter("max_scans", 400);
    this->declare_parameter("min_dist_trigger", 0.2);
    this->declare_parameter("min_time_trigger", 0.5);
    this->declare_parameter("lidar_topic", "/in/lidar_top/points/filtered");
    this->declare_parameter("odom_topic", "/genz/odometry");
    this->declare_parameter("save_ply", true);
    this->declare_parameter("ply_output_path", "merged_map.ply");

    max_scans_ = this->get_parameter("max_scans").as_int();
    min_dist_trigger_ = this->get_parameter("min_dist_trigger").as_double();
    min_time_trigger_ = this->get_parameter("min_time_trigger").as_double();
    const std::string lidar_topic = this->get_parameter("lidar_topic").as_string();
    const std::string odom_topic = this->get_parameter("odom_topic").as_string();
    save_ply_ = this->get_parameter("save_ply").as_bool();
    ply_output_path_ = this->get_parameter("ply_output_path").as_string();

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic,
      10,
      std::bind(&ScanMergerNode::odomCallback, this, std::placeholders::_1));

    scan_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic,
      10,
      std::bind(&ScanMergerNode::scanCallback, this, std::placeholders::_1));

    merged_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/merged_map",
      rclcpp::QoS(1));

    rclcpp::on_shutdown([this]() {
      saveFinalMergedMap();
    });

    if (max_scans_ <= 0) {
      RCLCPP_INFO(
        this->get_logger(),
        "ScanMerger started | max_scans=unlimited | dist_trigger=%.2fm | time_trigger=%.2fs",
        min_dist_trigger_,
        min_time_trigger_);
    } else {
      RCLCPP_INFO(
        this->get_logger(),
        "ScanMerger started | max_scans=%d | dist_trigger=%.2fm | time_trigger=%.2fs",
        max_scans_,
        min_dist_trigger_,
        min_time_trigger_);
    }
  }

private:
  struct StampedScan
  {
    sensor_msgs::msg::PointCloud2 cloud;
    Eigen::Matrix4f pose;
  };

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_pose_ = poseToMatrix(msg);
    latest_pos_ = Eigen::Vector3f(
      msg->pose.pose.position.x,
      msg->pose.pose.position.y,
      msg->pose.pose.position.z);
    odom_received_ = true;
  }

  void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!odom_received_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Waiting for odometry...");
      return;
    }

    const double now = this->now().seconds();
    const double dist = (latest_pos_ - last_saved_pos_).norm();
    const double dt = now - last_saved_time_;

    if (dist > min_dist_trigger_ || dt > min_time_trigger_) {
      StampedScan entry;
      entry.cloud = *msg;
      entry.pose = latest_pose_;
      buffer_.push_back(entry);

      if (max_scans_ > 0 && static_cast<int>(buffer_.size()) > max_scans_) {
        buffer_.pop_front();
      }

      last_saved_pos_ = latest_pos_;
      last_saved_time_ = now;

      if (max_scans_ <= 0) {
        RCLCPP_INFO(
          this->get_logger(),
          "Scan saved | buffer=%zu/unlimited | dist_since_last=%.2fm",
          buffer_.size(),
          dist);
      } else {
        RCLCPP_INFO(
          this->get_logger(),
          "Scan saved | buffer=%zu/%d | dist_since_last=%.2fm",
          buffer_.size(),
          max_scans_,
          dist);
      }

      publishMergedMap();
    }
  }

  void publishMergedMap()
  {
    pcl::PointCloud<pcl::PointXYZ> merged;

    for (const auto & entry : buffer_) {
      pcl::PointCloud<pcl::PointXYZ> cloud_in;
      pcl::PointCloud<pcl::PointXYZ> cloud_out;
      pcl::fromROSMsg(entry.cloud, cloud_in);
      pcl::transformPointCloud(cloud_in, cloud_out, entry.pose);
      merged += cloud_out;
    }

    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(merged, out_msg);
    out_msg.header.frame_id = "odom";
    out_msg.header.stamp = this->now();
    merged_pub_->publish(out_msg);
  }

  void saveFinalMergedMap()
  {
    if (!save_ply_ || final_ply_saved_) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> merged;
    for (const auto & entry : buffer_) {
      pcl::PointCloud<pcl::PointXYZ> cloud_in;
      pcl::PointCloud<pcl::PointXYZ> cloud_out;
      pcl::fromROSMsg(entry.cloud, cloud_in);
      pcl::transformPointCloud(cloud_in, cloud_out, entry.pose);
      merged += cloud_out;
    }

    const std::filesystem::path output_path(ply_output_path_);
    const std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Failed to create PLY output directory '%s': %s",
          parent.string().c_str(),
          ec.message().c_str());
        return;
      }
    }

    const int result = pcl::io::savePLYFileBinary(ply_output_path_, merged);
    if (result == 0) {
      final_ply_saved_ = true;
      RCLCPP_INFO(
        this->get_logger(),
        "Saved final merged map to PLY: %s",
        ply_output_path_.c_str());
    } else {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to save final PLY file: %s",
        ply_output_path_.c_str());
    }
  }

  Eigen::Matrix4f poseToMatrix(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const auto & p = msg->pose.pose.position;
    const auto & q = msg->pose.pose.orientation;

    Eigen::Quaternionf quat(q.w, q.x, q.y, q.z);
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = quat.toRotationMatrix();
    transform(0, 3) = p.x;
    transform(1, 3) = p.y;
    transform(2, 3) = p.z;
    return transform;
  }

  std::deque<StampedScan> buffer_;
  Eigen::Matrix4f latest_pose_ = Eigen::Matrix4f::Identity();
  Eigen::Vector3f latest_pos_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f last_saved_pos_ = Eigen::Vector3f::Zero();
  double last_saved_time_ = 0.0;
  bool odom_received_ = false;

  int max_scans_;
  double min_dist_trigger_;
  double min_time_trigger_;
  bool save_ply_;
  bool final_ply_saved_ = false;
  std::string ply_output_path_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scan_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr merged_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScanMergerNode>());
  rclcpp::shutdown();
  return 0;
}
