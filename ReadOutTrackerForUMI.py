import time
import openvr
import csv
import math
import numpy as np
from win_precise_time import sleep
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# Define the sampling rate (in Hz)
SAMPLING_RATE = 120 

def precise_wait(duration):
    """
    Wait for a specified duration with high precision.
    Uses sleep for durations >= 1 ms, otherwise uses busy-wait.
    """
    now = time.time()
    end = now + duration
    if duration >= 0.001:
        sleep(duration)
    while now < end:
        now = time.time()

class VRSystemManager:
    def __init__(self):
        """
        Initialize the VR system manager.
        """
        self.vr_system = None

    def initialize_vr_system(self):
        """
        Initialize the VR system.
        """
        try:
            openvr.init(openvr.VRApplication_Other)
            self.vr_system = openvr.VRSystem()
            print(f"Starting Capture")
        except Exception as e:
            print(f"Failed to initialize VR system: {e}")
            return False
        return True

    def get_tracker_data(self):
        """
        Retrieve tracker data from the VR system.
        """
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
        return poses

    def get_connected_trackers(self):
        """
        Get list of connected tracker indices.
        """
        tracker_indices = []
        poses = self.get_tracker_data()
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if poses[i].bPoseIsValid:
                device_class = self.vr_system.getTrackedDeviceClass(i)
                if device_class == openvr.TrackedDeviceClass_GenericTracker:
                    tracker_indices.append(i)
        return tracker_indices

    def print_discovered_objects(self):
        """
        Print information about discovered VR devices.
        """
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self.vr_system.getTrackedDeviceClass(device_index)
            if device_class != openvr.TrackedDeviceClass_Invalid:
                serial_number = self.vr_system.getStringTrackedDeviceProperty(
                    device_index, openvr.Prop_SerialNumber_String)
                model_number = self.vr_system.getStringTrackedDeviceProperty(
                    device_index, openvr.Prop_ModelNumber_String)
                print(f"Device {device_index}: {serial_number} ({model_number})")

    def shutdown_vr_system(self):
        """
        Shutdown the VR system.
        """
        if self.vr_system:
            openvr.shutdown()

class CSVLogger:
    def __init__(self):
        """
        Initialize the CSV logger.
        """
        self.files = {}
        self.csv_writers = {}
        self.frame_counters = {}
        self.base_dir = "data"

    def init_csv(self, tracker_indices):
        """
        Initialize CSV files for multiple trackers.
        """
        try:
            # Create base directory if it doesn't exist
            os.makedirs(self.base_dir, exist_ok=True)
            
            # Create timestamped subdirectory
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S.%f")
            session_dir = os.path.join(self.base_dir, timestamp)
            os.makedirs(session_dir, exist_ok=True)
            
            # Initialize CSV files for each tracker
            for tracker_idx in tracker_indices:
                tracker_number = tracker_idx - 1  # Convert to 0, 1, 2
                if tracker_number == 0:
                    filename = f"right_trajectory_{timestamp}.csv"
                elif tracker_number == 1:
                    filename = f"left_trajectory_{timestamp}.csv"
                else:
                    filename = f"root_{timestamp}.csv"
                filepath = os.path.join(session_dir, filename)
                
                self.files[tracker_number] = open(filepath, 'w', newline='')
                self.csv_writers[tracker_number] = csv.writer(self.files[tracker_number])
                self.frame_counters[tracker_number] = 0
                
                # Write header
                self.csv_writers[tracker_number].writerow(['frame_idx', 'timestamp', 'x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w'])

            print(f"Initialized CSV logging in directory: {session_dir}")
            return True
        except Exception as e:
            print(f"Failed to initialize CSV files: {e}")
            return False

    def log_data_csv(self, tracker_number, current_time, position):
        """
        Log tracker data to CSV file.
        """
        try:
            if tracker_number in self.csv_writers:
                frame_idx = self.frame_counters[tracker_number]
                dt_string = datetime.fromtimestamp(current_time).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                
                # Format: frame_idx,datetime,x,y,z,q_x,q_y,q_z,q_w
                self.csv_writers[tracker_number].writerow([
                    frame_idx, current_time, 
                    position[0], position[1], position[2],  # x, y, z
                    position[4], position[5], position[6], position[3]  # q_x, q_y, q_z, q_w
                ])
                
                self.frame_counters[tracker_number] += 1
        except Exception as e:
            print(f"Failed to write data to CSV file: {e}")

    def check_consistency(self):
        """
        Check timestamp consistency and 3D position jumps.
        Timestamp threshold: 0.08s, Position threshold: 0.05m
        """
        print("\n=== CSV Data Quality Check ===")
        print("时间戳阈值: 0.08s")
        print("3D位置跳跃阈值: 0.05m")
        print("=" * 50)
        
        for tracker_number, file in self.files.items():
            if file and not file.closed:
                file.flush()  # Ensure all data is written
                
        # Read and check each CSV file
        for tracker_number in self.csv_writers.keys():
            tracker_name = "right" if tracker_number == 0 else ("left" if tracker_number == 1 else "root")
            
            # Find the CSV file path
            csv_file_path = None
            for file_obj in self.files.values():
                if file_obj and hasattr(file_obj, 'name'):
                    if tracker_name in file_obj.name or f"tracker_{tracker_number}" in file_obj.name:
                        csv_file_path = file_obj.name
                        break
            
            if not csv_file_path:
                continue
                
            try:
                # Read the CSV file
                data_rows = []
                with open(csv_file_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Skip header but store it
                    for row in reader:
                        if len(row) >= 9:  # frame_idx,timestamp,x,y,z,q_x,q_y,q_z,q_w
                            try:
                                frame_idx = int(row[0])
                                timestamp = float(row[1])
                                x, y, z = float(row[2]), float(row[3]), float(row[4])
                                q_x, q_y, q_z, q_w = float(row[5]), float(row[6]), float(row[7]), float(row[8])
                                data_rows.append([frame_idx, timestamp, x, y, z, q_x, q_y, q_z, q_w])
                            except ValueError:
                                continue  # 跳过无效的数据行
                
                if len(data_rows) < 2:
                    print(f"Tracker {tracker_number} ({tracker_name}): 数据不足，无法检查一致性")
                    continue
                
                # Check timestamp differences and position jumps
                timestamp_violations = []
                position_violations = []
                max_timestamp_diff = 0
                max_position_jump = 0
                total_intervals = len(data_rows) - 1
                
                for i in range(1, len(data_rows)):
                    prev_row = data_rows[i-1]
                    curr_row = data_rows[i]
                    
                    # Check timestamp difference
                    timestamp_diff = curr_row[1] - prev_row[1]  # timestamp
                    max_timestamp_diff = max(max_timestamp_diff, timestamp_diff)
                    if timestamp_diff > 0.08:
                        timestamp_violations.append({
                            'frame': curr_row[0],
                            'diff': timestamp_diff,
                            'prev_row': prev_row,
                            'curr_row': curr_row
                        })
                    
                    # Check 3D position jump
                    prev_pos = np.array([prev_row[2], prev_row[3], prev_row[4]])  # x, y, z
                    curr_pos = np.array([curr_row[2], curr_row[3], curr_row[4]])  # x, y, z
                    position_jump = np.linalg.norm(curr_pos - prev_pos)  # 3D distance
                    max_position_jump = max(max_position_jump, position_jump)
                    if position_jump > 0.05:
                        position_violations.append({
                            'frame': curr_row[0],
                            'jump': position_jump,
                            'prev_row': prev_row,
                            'curr_row': curr_row
                        })
                
                # Print results
                print(f"Tracker {tracker_number} ({tracker_name}):")
                print(f"  总帧数: {len(data_rows)}")
                print(f"  时间间隔总数: {total_intervals}")
                
                # Timestamp check results
                print(f"  最大时间间隔: {max_timestamp_diff:.6f}s")
                print(f"  超过0.08s的间隔数: {len(timestamp_violations)}")
                timestamp_violation_rate = len(timestamp_violations)/total_intervals*100 if total_intervals > 0 else 0
                print(f"  时间戳违规率: {timestamp_violation_rate:.2f}%")
                
                # Position check results
                print(f"  最大位置跳跃: {max_position_jump:.6f}m")
                print(f"  超过0.05m的跳跃数: {len(position_violations)}")
                position_violation_rate = len(position_violations)/total_intervals*100 if total_intervals > 0 else 0
                print(f"  位置违规率: {position_violation_rate:.2f}%")
                
                # Overall status
                total_violations = len(timestamp_violations) + len(position_violations)
                if total_violations == 0:
                    print(f"  ✓ 数据质量检查通过")
                else:
                    print(f"  ✗ 数据质量检查失败 (总违规: {total_violations})")
                
                # Print violation details
                if timestamp_violations:
                    print(f"\n  时间戳违规详情:")
                    print(f"  {'帧号':<8} {'时间差(s)':<12} {'前一帧数据':<50} {'当前帧数据'}")
                    print(f"  {'-'*8} {'-'*12} {'-'*50} {'-'*50}")
                    for violation in timestamp_violations[:10]:  # 只显示前10个违规
                        prev_data = f"[{violation['prev_row'][0]}, {violation['prev_row'][1]:.6f}, {violation['prev_row'][2]:.4f}, {violation['prev_row'][3]:.4f}, {violation['prev_row'][4]:.4f}]"
                        curr_data = f"[{violation['curr_row'][0]}, {violation['curr_row'][1]:.6f}, {violation['curr_row'][2]:.4f}, {violation['curr_row'][3]:.4f}, {violation['curr_row'][4]:.4f}]"
                        print(f"  {violation['frame']:<8} {violation['diff']:<12.6f} {prev_data:<50} {curr_data}")
                    if len(timestamp_violations) > 10:
                        print(f"  ... 还有 {len(timestamp_violations) - 10} 个时间戳违规")
                
                if position_violations:
                    print(f"\n  位置违规详情:")
                    print(f"  {'帧号':<8} {'跳跃距离(m)':<15} {'前一帧位置':<30} {'当前帧位置'}")
                    print(f"  {'-'*8} {'-'*15} {'-'*30} {'-'*30}")
                    for violation in position_violations[:10]:  # 只显示前10个违规
                        prev_pos = f"[{violation['prev_row'][2]:.4f}, {violation['prev_row'][3]:.4f}, {violation['prev_row'][4]:.4f}]"
                        curr_pos = f"[{violation['curr_row'][2]:.4f}, {violation['curr_row'][3]:.4f}, {violation['curr_row'][4]:.4f}]"
                        print(f"  {violation['frame']:<8} {violation['jump']:<15.6f} {prev_pos:<30} {curr_pos}")
                    if len(position_violations) > 10:
                        print(f"  ... 还有 {len(position_violations) - 10} 个位置违规")
                
                print()
                
            except Exception as e:
                print(f"检查 Tracker {tracker_number} ({tracker_name}) 时出错: {e}")

    def close_csv(self):
        """
        Close all CSV files if they're open.
        """
        for file in self.files.values():
            if file:
                file.close()

class DataConverter:
    @staticmethod
    def convert_to_quaternion(pose_mat):
        """
        Convert pose matrix to quaternion and position.
        """
        r_w = math.sqrt(abs(1 + pose_mat[0][0] + pose_mat[1][1] + pose_mat[2][2])) / 2
        if r_w == 0: r_w = 0.0001
        r_x = (pose_mat[2][1] - pose_mat[1][2]) / (4 * r_w)
        r_y = (pose_mat[0][2] - pose_mat[2][0]) / (4 * r_w)
        r_z = (pose_mat[1][0] - pose_mat[0][1]) / (4 * r_w)

        x = pose_mat[0][3]
        y = pose_mat[1][3]
        z = pose_mat[2][3]

        return [x, y, z, r_w, r_x, r_y, r_z]

class LivePlotter:
    def __init__(self):
        """
        Initialize the live plotter.
        """
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.x_data = deque()
        self.y_data = deque()
        self.z_data = deque()
        self.time_data = deque()
        self.first = True
        self.firstx = 0
        self.firsty = 0
        self.firstz = 0
        self.start_time = time.time()
        self.vive_PosVIVE = np.zeros([3])

    def init_live_plot(self):
        """
        Initialize the live plot for VIVE tracker data.
        """
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1)
        self.ax1.set_title('X Position')
        self.ax2.set_title('Y Position')
        self.ax3.set_title('Z Position')
        self.x_line, = self.ax1.plot([], [], 'r-')
        self.y_line, = self.ax2.plot([], [], 'g-')
        self.z_line, = self.ax3.plot([], [], 'b-')
        plt.ion()
        plt.show()

    def update_live_plot(self, vive_PosVIVE):
        """
        Update the live plot with new VIVE tracker data.
        """
        current_time = time.time()
        self.x_data.append(vive_PosVIVE[0])
        self.y_data.append(vive_PosVIVE[1])
        self.z_data.append(vive_PosVIVE[2])
        self.time_data.append(current_time - self.start_time)

        self.x_line.set_data(self.time_data, self.x_data)
        self.y_line.set_data(self.time_data, self.y_data)
        self.z_line.set_data(self.time_data, self.z_data)

        if self.first:
            self.firstx = self.x_data[0]
            self.firsty = self.y_data[0]
            self.firstz = self.z_data[0]
            self.first = False

        self.ax1.set_xlim(self.time_data[0], self.time_data[-1])
        self.ax1.set_ylim([self.firstx - 1.5, self.firstx + 1.5])

        self.ax2.set_xlim(self.time_data[0], self.time_data[-1])
        self.ax2.set_ylim([self.firsty - 1.5, self.firsty + 1.5])

        self.ax3.set_xlim(self.time_data[0], self.time_data[-1])
        self.ax3.set_ylim([self.firstz - 1.5, self.firstz + 1.5])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def init_3d_plot(self):
        """
        Initialize the 3D live plot for VIVE tracker data.
        """
        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.view_init(elev=1, azim=180, roll=None, vertical_axis='y')

        self.maxlen_3d = 50
        self.x_data_3d = deque(maxlen=self.maxlen_3d)
        self.y_data_3d = deque(maxlen=self.maxlen_3d)
        self.z_data_3d = deque(maxlen=self.maxlen_3d)

        self.line_3d, = self.ax_3d.plot([], [], [], 'r-')

        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Tracker Position')

        plt.ion()
        plt.show()

    def update_3d_plot(self, vive_PosVIVE):
        """
        Update the 3D live plot with new VIVE tracker data.
        """
        x, y, z = vive_PosVIVE

        self.x_data_3d.append(x)
        self.y_data_3d.append(y)
        self.z_data_3d.append(z)

        self.line_3d.set_data(self.x_data_3d, self.y_data_3d)
        self.line_3d.set_3d_properties(self.z_data_3d)

        if len(self.x_data_3d) > 1:
            self.ax_3d.set_xlim(min(self.x_data_3d), max(self.x_data_3d))
            self.ax_3d.set_ylim(min(self.y_data_3d), max(self.y_data_3d))
            self.ax_3d.set_zlim(min(self.z_data_3d), max(self.z_data_3d))

        self.fig_3d.canvas.draw()
        self.fig_3d.canvas.flush_events()

def main():
    vr_manager = VRSystemManager()
    csv_logger = CSVLogger()
    plotter = LivePlotter()

    # enable or disable plots (for maximum performance disable all plots)
    plot_3d = False # live 3D plot (might affect performance)
    plot_t_xyz = False # live plot of x, y, z positions
    log_data = True # log data to CSV file
    print_data = True # print data to console

    if not vr_manager.initialize_vr_system():
        return

    # Check for connected trackers
    tracker_indices = vr_manager.get_connected_trackers()
    print(f"Found {len(tracker_indices)} trackers: {tracker_indices}")
    
    if len(tracker_indices) != 3:
        print(f"Error: Expected 3 trackers, but found {len(tracker_indices)}")
        vr_manager.shutdown_vr_system()
        return

    if log_data and not csv_logger.init_csv(tracker_indices):
        return

    if plot_t_xyz: plotter.init_live_plot()
    if plot_3d: plotter.init_3d_plot()

    try:
        while True:
            poses = vr_manager.get_tracker_data()
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                if poses[i].bPoseIsValid:
                    device_class = vr_manager.vr_system.getTrackedDeviceClass(i)
                    if device_class == openvr.TrackedDeviceClass_GenericTracker:
                        current_time = time.time()
                        position = DataConverter.convert_to_quaternion(poses[i].mDeviceToAbsoluteTracking)
                        tracker_number = i - 1  # Convert to 0, 1, 2
                        
                        if plot_t_xyz: plotter.update_live_plot(position[:3])
                        if plot_3d: plotter.update_3d_plot(position[:3])
                        if log_data: csv_logger.log_data_csv(tracker_number, current_time, position)
                        if print_data: print(f"Tracker {tracker_number}: {position}")
            precise_wait(1 / SAMPLING_RATE)
    except KeyboardInterrupt:
        print("Stopping data collection...")
    finally:
        if log_data:
            csv_logger.check_consistency()
        vr_manager.shutdown_vr_system()
        csv_logger.close_csv()


if __name__ == "__main__":
    main()