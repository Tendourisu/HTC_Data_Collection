import time
import openvr
import csv
import math
import numpy as np
from win_precise_time import sleep
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
import viser
import threading
import random

# Define the sampling rate (in Hz)
SAMPLING_RATE = 120 
file_name = "ultimate_tracker_data.csv"
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
        self.file = None
        self.csv_writer = None

    def init_csv(self, filename):
        """
        Initialize the CSV file for logging tracker data.
        """
        try:
            self.file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.file)
            self.csv_writer.writerow(['TrackerIndex', 'Time', 'PositionX', 'PositionY', 'PositionZ', 'RotationW', 'RotationX', 'RotationY', 'RotationZ'])
        except Exception as e:
            print(f"Failed to initialize CSV file: {e}")
            return False
        return True

    def log_data_csv(self, index, current_time, position):
        """
        Log tracker data to CSV file.
        """
        try:
            self.csv_writer.writerow([index, current_time, *position])
        except Exception as e:
            print(f"Failed to write data to CSV file: {e}")

    def close_csv(self):
        """
        Close the CSV file if it's open.
        """
        if self.file:
            self.file.close()

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

class WebVisualizer:
    def __init__(self, port=8081):
        self.port = port
        self.server = None
        self.tracker_spheres = {}  # Store sphere objects for each tracker
        self.tracker_labels = {}    # Store label objects for each tracker
        self.tracker_colors = {}   # Store colors for each tracker
        self.trail_lines = {}      # Store trail lines
        self.trail_points = {}     # Store trail points
        self.tracker_orient_axes = {}  # Small 3-axis lines showing sphere orientation (X:red,Y:green,Z:blue)
        self.show_trails = True
        self.is_running = True
        self.server_thread = None
        self.trail_length = 100
        
        # Default color options
        self.default_colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#FFA500', '#800080', '#FFC0CB', '#A52A2A', '#808080', '#000000'
        ]
        
    def init_web_server(self):
        """Initialize viser web server with enhanced UI controls"""
        try:
            self.server = viser.ViserServer(port=self.port)
            
            # Add coordinate axes with clear naming
            # smaller world axes for compact view
            try:
                self.server.scene.add_frame(
                    "/world_axes",
                    show_axes=True,
                    axes_length=0.4,
                    axes_radius=0.01
                )
            except Exception:
                # fallback if axes_radius not supported
                self.server.scene.add_frame(
                    "/world_axes",
                    show_axes=True,
                    axes_length=0.4
                )
            
            # Add ground grid with explicit parameters
            self.server.scene.add_grid(
                "/ground_grid",
                width=20.0, height=20.0,
                width_segments=20, height_segments=20,
                plane="xz"
            )
            
            # Add structured UI controls
            with self.server.gui.add_folder("Tracker Controls"):
                self.is_running_gui = self.server.gui.add_button(
                    "Start/Pause"
                )
                self.is_running_gui.on_click(self.toggle_running)
                
            with self.server.gui.add_folder("Visualization Settings"):
                self.show_trails_gui = self.server.gui.add_checkbox(
                    "Show Trails",
                    initial_value=True
                )
                self.show_trails_gui.on_update(self.toggle_trails)
                
                self.trail_length_gui = self.server.gui.add_slider(
                    "Trail Length",
                    min=10, max=500, step=10, initial_value=100
                )
                self.trail_length_gui.on_update(self._update_trail_length)
                
            with self.server.gui.add_folder("Statistics"):
                self.fps_text = self.server.gui.add_text(
                    "FPS",
                    initial_value="0"
                )
                self.tracker_count_text = self.server.gui.add_text(
                    "Active Trackers",
                    initial_value="0"
                )
            
            # Initialize color selectors and toggles dictionaries
            self.color_selectors = {}
            self.tracker_toggles = {}
            self.tracker_names = {}
            
            print(f"Web server started at http://localhost:{self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize web server: {e}")
            return False
    
    def add_tracker(self, tracker_id, color=None, name=None):
        """Add a new tracker to the visualization with enhanced features"""
        try:
            if color is None:
                color = random.choice(self.default_colors)
            
            # Validate color more strictly
            if not isinstance(color, str):
                color = "#FF0000"
            elif not color.startswith('#') or len(color) != 7:
                color = "#FF0000"
            else:
                # Validate hex characters
                hex_part = color[1:]
                if not all(c in '0123456789abcdefABCDEF' for c in hex_part):
                    color = "#FF0000"
            
            tracker_name = name or f"Tracker {tracker_id}"
            self.tracker_colors[tracker_id] = color
            self.tracker_names[tracker_id] = tracker_name
            
            # Convert hex color to RGB tuple for viser
            rgb_color = self._hex_to_rgb(color)
            
            # Add icosphere for tracker (smaller radius)
            sphere = self.server.scene.add_icosphere(
                f"/trackers/tracker_{tracker_id}",
                radius=0.02,
                color=rgb_color
            )
            self.tracker_spheres[tracker_id] = sphere
            
            # Add label for tracker identification
            label = self.server.scene.add_label(
                f"/trackers/tracker_{tracker_id}/label",
                text=tracker_name,
                position=np.array([0, 0.1, 0])
            )
            self.tracker_labels[tracker_id] = label
            
            # Initialize trail data structures FIRST
            self.trail_points[tracker_id] = []
            
            # Then create the trail line
            try:
                # Create with proper shape (0, 2, 3) for 0 line segments
                trail_line = self.server.scene.add_line_segments(
                    f"/trackers/trail_{tracker_id}",
                    points=np.empty((0, 2, 3)),  # Shape (N, 2, 3) for N line segments
                    colors=np.empty((0, 2, 3))   # Matching colors shape
                )
                self.trail_lines[tracker_id] = trail_line
            except Exception as trail_error:
                print(f"Warning: Failed to create trail line for tracker {tracker_id}: {trail_error}")
                # Create a placeholder to prevent KeyError
                self.trail_lines[tracker_id] = None
            
            # create an orientation indicator (small line) --- start as empty
            try:
                # initialize as empty; will set to shape (3,2,3) when first updated
                orient_axes = self.server.scene.add_line_segments(
                    f"/trackers/orient_axes_{tracker_id}",
                    points=np.empty((0, 2, 3)),
                    colors=np.empty((0, 2, 3))
                )
                self.tracker_orient_axes[tracker_id] = orient_axes
            except Exception as e:
                print(f"Warning: Failed to create orientation axes for tracker {tracker_id}: {e}")
                self.tracker_orient_axes[tracker_id] = None
            
            # Add individual tracker controls
            with self.server.gui.add_folder(f"Tracker {tracker_id}"):
                # Try different color selector methods
                try:
                    color_selector = self.server.gui.add_rgb(
                        "Color",
                        initial_value=rgb_color
                    )
                except AttributeError:
                    try:
                        color_selector = self.server.gui.add_color(
                            "Color",
                            initial_value=rgb_color
                        )
                    except AttributeError:
                        print(f"Warning: Color selector not available for tracker {tracker_id}")
                        color_selector = None
                
                if color_selector:
                    color_selector.on_update(
                        lambda rgb, tid=tracker_id: self._update_tracker_color(tid, rgb)
                    )
                    self.color_selectors[tracker_id] = color_selector
                
                # Add visibility toggle
                toggle = self.server.gui.add_checkbox(
                    "Visible",
                    initial_value=True
                )
                toggle.on_update(
                    lambda visible, tid=tracker_id: self._toggle_tracker_visibility(tid, visible)
                )
                self.tracker_toggles[tracker_id] = toggle
            
            # Update tracker count
            self._update_tracker_count()
            
            print(f"Successfully added tracker {tracker_id}")
            
        except Exception as e:
            import traceback
            print(f"Error adding tracker {tracker_id}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Ensure at least basic structures exist to prevent KeyErrors
            if tracker_id not in self.trail_points:
                self.trail_points[tracker_id] = []
            if tracker_id not in self.trail_lines:
                self.trail_lines[tracker_id] = None
    
    def update_tracker_pose(self, tracker_id, position, rotation):
        """Update tracker position and rotation with error handling"""
        try:
            if tracker_id not in self.tracker_spheres:
                self.add_tracker(tracker_id)
            
            # Check if tracker was successfully added
            if tracker_id not in self.tracker_spheres:
                print(f"Error: Failed to add tracker {tracker_id}")
                return
            
            # Validate position data
            if len(position) != 3:
                print(f"Warning: Invalid position data for tracker {tracker_id}: {position}")
                return
            
            # Convert position to float if needed
            try:
                position = [float(p) for p in position]
            except (ValueError, TypeError) as e:
                print(f"Error converting position to float: {e}")
                return
            
            # Convert to numpy array for viser
            position_array = np.array(position)
            
            # Update sphere position
            self.tracker_spheres[tracker_id].position = position_array
            
            # Update label position (above sphere)
            if tracker_id in self.tracker_labels:
                label_position = np.array([position[0], position[1] + 0.1, position[2]])
                self.tracker_labels[tracker_id].position = label_position
            
            # per-tracker local frame removed; orientation shown via colored axes (tracker_orient_axes)
            
            # Update 3 colored local axes to show sphere orientation (X:red, Y:green, Z:blue)
            try:
                orient_axes = self.tracker_orient_axes.get(tracker_id, None)
                if orient_axes is not None and rotation is not None and len(rotation) == 4:
                    # quaternion provided as [w, x, y, z]
                    w, x, y, z = rotation[0], rotation[1], rotation[2], rotation[3]
                    # local axes vectors
                    axis_len = 0.12
                    local_x = np.array([axis_len, 0.0, 0.0])
                    local_y = np.array([0.0, axis_len, 0.0])
                    local_z = np.array([0.0, 0.0, axis_len])
                    rm = self._quat_to_rotmat(w, x, y, z)
                    wx = rm.dot(local_x)
                    wy = rm.dot(local_y)
                    wz = rm.dot(local_z)
                    start = position_array
                    pts = np.empty((3, 2, 3))
                    pts[0, 0] = start; pts[0, 1] = start + wx
                    pts[1, 0] = start; pts[1, 1] = start + wy
                    pts[2, 0] = start; pts[2, 1] = start + wz
                    # colors: X=red, Y=green, Z=blue (normalized)
                    colors = np.empty((3, 2, 3))
                    colors[0, :, :] = np.array([1.0, 0.0, 0.0])
                    colors[1, :, :] = np.array([0.0, 1.0, 0.0])
                    colors[2, :, :] = np.array([0.0, 0.0, 1.0])
                    orient_axes.points = pts
                    orient_axes.colors = colors
            except Exception as e:
                print(f"Warning: Failed to update orientation axes for tracker {tracker_id}: {e}")
            
            # Update trail if enabled and running
            if self.show_trails and self.is_running:
                # Ensure trail_points exists for this tracker
                if tracker_id not in self.trail_points:
                    self.trail_points[tracker_id] = []
                self._update_trail(tracker_id, position)
                
        except Exception as e:
            import traceback
            print(f"Error updating tracker {tracker_id}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
    
    def toggle_running(self):
        """Toggle running state"""
        self.is_running = not self.is_running
        status = "Running" if self.is_running else "Paused"
        print(f"Trail display {status}")
    
    def toggle_trails(self, show):
        """Toggle trail visibility"""
        self.show_trails = show
        for tracker_id in self.trail_lines:
            if self.trail_lines[tracker_id] is None:
                continue
                
            try:
                if show and len(self.trail_points[tracker_id]) > 1:
                    # Convert points to line segments with proper shape (N, 2, 3)
                    points = self.trail_points[tracker_id]
                    num_segments = len(points) - 1
                    line_segments = np.empty((num_segments, 2, 3))
                    line_segments[:, 0] = points[:-1]  # Start points
                    line_segments[:, 1] = points[1:]   # End points
                    
                    # Create colors for the line segments
                    rgb_color = self._hex_to_rgb(self.tracker_colors.get(tracker_id, "#FF0000"))
                    rgb_normalized = tuple(c / 255.0 for c in rgb_color)
                    # Colors shape should match: (N, 2, 3) for N segments
                    colors = np.tile(rgb_normalized, (num_segments, 2, 1))
                    
                    self.trail_lines[tracker_id].points = line_segments
                    self.trail_lines[tracker_id].colors = colors
                else:
                    # Set empty arrays with correct shape
                    self.trail_lines[tracker_id].points = np.empty((0, 2, 3))
                    self.trail_lines[tracker_id].colors = np.empty((0, 2, 3))
            except Exception as e:
                print(f"Warning: Failed to toggle trail for tracker {tracker_id}: {e}")
    
    def _update_tracker_color(self, tracker_id, rgb):
        """Update tracker color"""
        try:
            if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
                return
                
            # Validate RGB values (0-255)
            rgb_tuple = tuple(max(0, min(255, int(c))) for c in rgb)
            
            # Store hex color for reference
            color = f"#{rgb_tuple[0]:02x}{rgb_tuple[1]:02x}{rgb_tuple[2]:02x}"
            self.tracker_colors[tracker_id] = color
            
            # Use RGB tuple for viser components
            try:
                self.tracker_spheres[tracker_id].color = rgb_tuple
            except (AttributeError, KeyError):
                pass
            
            try:
                if self.trail_lines[tracker_id] is not None:
                    self.trail_lines[tracker_id].color = rgb_tuple
            except (AttributeError, KeyError):
                pass
                
            try:
                if tracker_id in self.tracker_labels:
                    self.tracker_labels[tracker_id].color = rgb_tuple
            except (AttributeError, KeyError):
                pass
            # update orientation indicator color as well
            # axes keep standard RGB colors; no change needed here
            # (if you want the axes to follow tracker color, implement here)
            pass
        except Exception as e:
            print(f"Error updating tracker color: {e}")
    
    def _toggle_tracker_visibility(self, tracker_id, visible):
        """Toggle tracker visibility"""
        if visible:
            # Restore to last known position (will be updated in next frame)
            pass
        else:
            # Hide by moving underground - use numpy array
            hidden_position = np.array([0, -100, 0])
            self.tracker_spheres[tracker_id].position = hidden_position
            if tracker_id in self.tracker_labels:
                self.tracker_labels[tracker_id].position = hidden_position
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        if not hex_color or not isinstance(hex_color, str):
            return (255, 255, 255)  # Default to white
        
        hex_color = hex_color.lstrip('#')
        
        # Validate hex color format
        if len(hex_color) not in [3, 6] or not all(c in '0123456789abcdefABCDEF' for c in hex_color):
            return (255, 255, 255)  # Default to white
        
        # Handle 3-character hex colors
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except (ValueError, IndexError):
            return (255, 255, 255)  # Default to white on error
    
    def _quat_to_rotmat(self, w, x, y, z):
        """Convert quaternion (w,x,y,z) to 3x3 rotation matrix"""
        # normalize
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0:
            norm = 1.0
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
        ])

    def _update_trail(self, tracker_id, new_position):
        """Efficiently update tracker trail"""
        if not self.show_trails:
            return
        
        # Ensure trail_points exists for this tracker
        if tracker_id not in self.trail_points:
            self.trail_points[tracker_id] = []
        
        # Ensure trail_lines exists for this tracker and is not None
        if tracker_id not in self.trail_lines or self.trail_lines[tracker_id] is None:
            return  # Silently skip if trail line creation failed
            
        trail_points = self.trail_points[tracker_id]
        trail_points.append(new_position)
        
        # Limit trail length
        if len(trail_points) > self.trail_length:
            trail_points.pop(0)
        
        # Use numpy array operations for better performance
        if len(trail_points) > 1:
            try:
                points_array = np.array(trail_points)
                # Create line segment pairs with proper shape (N, 2, 3)
                num_segments = len(points_array) - 1
                line_segments = np.empty((num_segments, 2, 3))
                line_segments[:, 0] = points_array[:-1]  # Start points
                line_segments[:, 1] = points_array[1:]   # End points
                
                # Create colors for each segment (RGB values 0-1)
                rgb_color = self._hex_to_rgb(self.tracker_colors.get(tracker_id, "#FF0000"))
                rgb_normalized = tuple(c / 255.0 for c in rgb_color)  # Normalize to 0-1
                # Colors shape should match line_segments: (N, 2, 3)
                colors = np.tile(rgb_normalized, (num_segments, 2, 1))
                
                # Update trail line points and colors
                self.trail_lines[tracker_id].points = line_segments
                self.trail_lines[tracker_id].colors = colors
                
            except Exception as e:
                print(f"Warning: Failed to update trail for tracker {tracker_id}: {e}")
    
    def _update_trail_length(self, length):
        """Update trail length setting"""
        self.trail_length = length
        
        # Trim existing trails if needed
        for tracker_id in self.trail_points:
            if len(self.trail_points[tracker_id]) > length:
                self.trail_points[tracker_id] = self.trail_points[tracker_id][-length:]
    
    def _update_tracker_count(self):
        """Update the tracker count display"""
        if hasattr(self, 'tracker_count_text'):
            self.tracker_count_text.value = str(len(self.tracker_spheres))
    
    def start_server(self):
        """Start the web server in a separate thread with status monitoring"""
        def server_thread():
            try:
                print(f"Viser server starting on port {self.port}")
                # Keep the server running
                while True:
                    time.sleep(1)
            except Exception as e:
                print(f"Server error: {e}")
            finally:
                print("Viser server stopped")
        
        self.server_thread = threading.Thread(target=server_thread, daemon=True)
        self.server_thread.start()
        
        # Add server status monitoring
        self.server_start_time = time.time()
    
    def stop_server(self):
        """Stop the web server"""
        # The server will be stopped automatically when the thread ends
        if self.server_thread and self.server_thread.is_alive():
            print("Stopping server thread...")
            # The daemon thread will be terminated when the main program ends

def main():
    vr_manager = VRSystemManager()
    csv_logger = CSVLogger()
    plotter = LivePlotter()
    web_visualizer = WebVisualizer()  # Add web visualizer

    # enable or disable plots (for maximum performance disable all plots)
    plot_3d = False # live 3D plot (might affect performance)
    plot_t_xyz = False # live plot of x, y, z positions
    log_data = True # log data to CSV file
    print_data = False # print data to console
    web_visualization = True # enable web visualization

    if not vr_manager.initialize_vr_system():
        return

    if not csv_logger.init_csv(file_name):
        return

    if web_visualization:
        if not web_visualizer.init_web_server():
            print("Failed to initialize web visualization")
            web_visualization = False
        else:
            web_visualizer.start_server()

    if plot_t_xyz: plotter.init_live_plot()
    if plot_3d: plotter.init_3d_plot()

    try:
        while True:
            # Always run the main loop - web visualization is independent
            # The is_running flag now controls trail display, not the entire program
                
            poses = vr_manager.get_tracker_data()
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                if poses[i].bPoseIsValid:
                    device_class = vr_manager.vr_system.getTrackedDeviceClass(i)
                    if device_class == openvr.TrackedDeviceClass_GenericTracker:
                        current_time = time.time()
                        position = DataConverter.convert_to_quaternion(poses[i].mDeviceToAbsoluteTracking)
                        
                        if plot_t_xyz: plotter.update_live_plot(position[:3])
                        if plot_3d: plotter.update_3d_plot(position[:3])
                        if log_data: csv_logger.log_data_csv(i - 1, current_time, position)
                        if print_data: print(f"Tracker {i - 1}: {position}")
                        
                        # Update web visualization
                        if web_visualization:
                            web_visualizer.update_tracker_pose(i - 1, position[:3], position[3:])
                            
            precise_wait(1 / SAMPLING_RATE)
    except KeyboardInterrupt:
        print("Stopping data collection...")
    finally:
        vr_manager.shutdown_vr_system()
        csv_logger.close_csv()
        if web_visualization:
            web_visualizer.stop_server()


if __name__ == "__main__":
    main()