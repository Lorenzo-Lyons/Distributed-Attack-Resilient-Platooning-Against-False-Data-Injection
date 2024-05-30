#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/radius_outlier_removal.h>

class LaserScanCluster {
public:
    LaserScanCluster(ros::NodeHandle& nh, const std::string& car_number) : nh_(nh) {
        // Set up LaserScan subscriber
        laser_scan_sub_ = nh_.subscribe("/scan_" + car_number, 1, &LaserScanCluster::laserScanCallback, this);

        // Set up MarkerArray publisher
        marker_array_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/clusters_" + car_number, 1);
    }

    void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& scan_msg) {

        // Convert LaserScan to PointCloud2
        sensor_msgs::PointCloud2::Ptr cloud_msg(new sensor_msgs::PointCloud2);
        convertLaserScanToPointCloud2(scan_msg, cloud_msg);


        // Convert PointCloud2 to pcl PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud);



        // Apply Voxel Grid Downsampling
        pcl::VoxelGrid<pcl::PointXYZ> vox;
        vox.setInputCloud(cloud);
        vox.setLeafSize(0.03, 0.03, 0.03);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        vox.filter(*cloud_filtered);

        // Set the maximum allowed distance
        double max_distance = 1.7;  // Set your desired maximum distance

        // Create a filtered point cloud based on distance
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_distance(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& point : cloud_filtered->points) {
            double distance = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (distance <= max_distance) {
                cloud_filtered_distance->points.push_back(point);
            }
        }

        // Apply Euclidean Clustering to the filtered point cloud
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud_filtered_distance);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.2);  // Adjust based on your environment
        ec.setMinClusterSize(4);
        ec.setMaxClusterSize(10);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_filtered_distance);
        ec.extract(cluster_indices);

        // Create MarkerArray
        visualization_msgs::MarkerArray marker_array;

        // Iterate through clusters
        for (std::size_t i = 0; i < cluster_indices.size(); ++i) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (std::size_t j = 0; j < cluster_indices[i].indices.size(); ++j) {
                cluster->points.push_back(cloud_filtered_distance->points[cluster_indices[i].indices[j]]);
            }

            // Create Marker for the cluster
            visualization_msgs::Marker marker = createClusterMarker(cluster, scan_msg->header, i);

            // Append Marker to MarkerArray
            marker_array.markers.push_back(marker);
        }

        // Publish MarkerArray
        marker_array_pub_.publish(marker_array);
    }

    void convertLaserScanToPointCloud2(const sensor_msgs::LaserScan::ConstPtr& scan_msg,
                                    sensor_msgs::PointCloud2::Ptr& cloud_msg) {
        // Create a PointCloudXYZ
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        // Reserve space for points
        cloud->points.resize(scan_msg->ranges.size());

        // Populate PointCloudXYZ with points from LaserScan
        for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
            // Calculate the angle of the current point
            double angle = scan_msg->angle_min + i * scan_msg->angle_increment;

            // Calculate the Cartesian coordinates of the point
            double x = scan_msg->ranges[i] * cos(angle);
            double y = scan_msg->ranges[i] * sin(angle);

            // Set the point in the PointCloudXYZ
            cloud->points[i].x = x;
            cloud->points[i].y = y;
            cloud->points[i].z = 0.0;  // Assuming 2D laser scan, so z is set to 0
        }

        // Convert PointCloudXYZ to PointCloud2
        pcl::toROSMsg(*cloud, *cloud_msg);

        // Set the header of the PointCloud2 message
        cloud_msg->header = scan_msg->header;
        cloud_msg->height = 1;
        cloud_msg->width = cloud->points.size();
    }

    visualization_msgs::Marker createClusterMarker(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
                                                   const std_msgs::Header& header, std::size_t cluster_id) {
        visualization_msgs::Marker marker;
        marker.header = header;
        marker.id = cluster_id;
        marker.type = visualization_msgs::Marker::POINTS;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05;  // Point size


        // Fixed colors for each cluster (you can customize these)
        std::vector<std::array<float, 3>> fixed_colors = {
            {1.0, 0.0, 0.0},  // Red
            {0.0, 1.0, 0.0},  // Green
            {0.0, 0.0, 1.0},  // Blue
            {1.0, 1.0, 0.0},  // 
            {1.0, 0.0, 1.0},  // 
            {0.0, 1.0, 1.0},  // 
            // Add more colors as needed
        };

        // Select a color based on cluster_id
        std::array<float, 3> cluster_color;
        if (cluster_id < fixed_colors.size()) {
            cluster_color = fixed_colors[cluster_id];
        } else {
            // If there are more clusters than predefined colors, use a default color
            cluster_color = {0.5, 0.5, 0.5};  // Gray
        }

        marker.color.r = cluster_color[0];
        marker.color.g = cluster_color[1];
        marker.color.b = cluster_color[2];
        marker.color.a = 0.5;  // Alpha

        // Convert cluster points to geometry_msgs/Point
        for (const auto& point : cluster->points) {
            geometry_msgs::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            marker.points.push_back(p);
        }

        return marker;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber laser_scan_sub_;
    ros::Publisher marker_array_pub_;
};

int main(int argc, char** argv) {
    // Retrieve the value of the "car_number" environment variable
    const char* car_numberEnv = std::getenv("car_number");
    if (car_numberEnv == nullptr) {
        ROS_ERROR("Environment variable 'car_number' not set.");
        return 1;  // Exit with an error code
    }

    // Convert car number to string
    std::string car_number(car_numberEnv);

    // Initialize the ROS node with the obtained value
    ros::init(argc, argv, "laser_scan_clustering_" + car_number);
    ros::NodeHandle nh;

    LaserScanCluster laser_scan_cluster(nh,car_number);

    ros::spin();

    return 0;
}

