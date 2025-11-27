import time
import math
from collections import deque
from pymavlink import mavutil

class MotionEstimator:
    def __init__(self, distance_threshold=5.0, rotation_threshold=5):
        self.distance_threshold = distance_threshold  # meters
        self.rotation_threshold = rotation_threshold  # rad/s
        self.last_velocity = 0.0
        self.total_distance = 0.0
        self.last_time = None
        self.mavlink_connection = None
        self.buffer_size = 10
        self.acc_buffer = deque(maxlen=self.buffer_size)
        self.gyro_buffer = deque(maxlen=self.buffer_size)
        
    def connect_mavlink(self, connection_string=None):
        """
        Connect to Pixhawk via USB
        """
        try:
            # List available serial ports
            from serial.tools import list_ports
            ports = list(list_ports.comports())
            
            # Find Pixhawk USB connection
            pixhawk_port = None
            for port in ports:
                # Pixhawk usually identifies itself with 'PX4' or 'Pixhawk' in the description
                if 'PX4' in port.description or 'Pixhawk' in port.description:
                    pixhawk_port = port.device
                    break
            
            if pixhawk_port is None:
                # If no Pixhawk-specific port found, try common USB ports
                common_ports = [
                    '/dev/ttyACM0',  # Linux
                    '/dev/ttyUSB0',  # Linux
                    'COM3',          # Windows
                    'COM4',           # Windows
                    '/dev/tty.usbmodem14501'
                ]
                
                for port in common_ports:
                    try:
                        self.mavlink_connection = mavutil.mavlink_connection(port, baud=57600)
                        # If connection successful, break the loop
                        pixhawk_port = port
                        break
                    except:
                        continue
            
            if pixhawk_port is None:
                # raise Exception("No Pixhawk USB connection found")
                print("No Pixhawk USB connection found")
                return False
            
            # Connect to the detected port if not already connected
            if self.mavlink_connection is None:
                self.mavlink_connection = mavutil.mavlink_connection(pixhawk_port, baud=57600)
            
            # Wait for heartbeat to confirm connection
            print(f"Waiting for heartbeat from Pixhawk on {pixhawk_port}...")
            self.mavlink_connection.wait_heartbeat()
            print("Heartbeat received! Connected to Pixhawk")
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to Pixhawk via USB: {str(e)}")
            return False

    def get_telemetry_data(self):
        if not self.mavlink_connection:
            return None, None
            
        msg = self.mavlink_connection.recv_match(
            type=['RAW_IMU'],
            blocking=True,
            timeout=1.0
        )
        
        if msg is None:
            return None, None
            
        if msg.get_type() == 'RAW_IMU':
            # Convert accelerometer data from milli-g to m/s²
            xacc = msg.xacc * 9.81 / 1000.0
            
            # Convert gyroscope data from mdeg/s to rad/s
            # Calculate total horizontal rotation rate
            gyro_x_rad = msg.xgyro * (math.pi / (180 ))  # Convert to radians/s
            gyro_y_rad = msg.ygyro * (math.pi / (180 ))  # Convert to radians/s
            rotation_rate = math.sqrt(gyro_x_rad**2 + gyro_y_rad**2)
            
            #print(f"Total rotation rate: {rotation_rate} rad/s")
            
            return xacc, rotation_rate
        
        return None, None
    
    def update_motion_estimation(self):
        xacc, rotation_rate = self.get_telemetry_data()
        if xacc is None or rotation_rate is None:
            return False, False
            
        current_time = time.time()
        
        # Initialize last_time if this is the first update
        if self.last_time is None:
            self.last_time = current_time
            return False, False
            
        dt = current_time - self.last_time
        
        # Apply simple moving average filter to acceleration and rotation rate
        self.acc_buffer.append(xacc)
        self.gyro_buffer.append(rotation_rate)
        
        filtered_acc = sum(self.acc_buffer) / len(self.acc_buffer)
        filtered_rotation = sum(self.gyro_buffer) / len(self.gyro_buffer)
        
        # Add deadzone for acceleration to prevent drift
        acc_deadzone = 0.1  # m/s²
        if abs(filtered_acc) < acc_deadzone:
            filtered_acc = 0
        
        # Update velocity using filtered acceleration
        current_velocity = self.last_velocity + filtered_acc * dt
        
        # Add velocity decay to prevent drift
        velocity_decay = 0.95
        current_velocity *= velocity_decay
        
        # Update distance
        distance_increment = ((self.last_velocity + current_velocity) / 2) * dt
        self.total_distance += abs(distance_increment)
        
        # print(f"Filtered Acc: {filtered_acc:.6f} m/s²")
        # print(f"Current Velocity: {current_velocity:.6f} m/s")
        # print(f"Total Distance: {self.total_distance:.6f} m")
        
        # Update last values
        self.last_velocity = current_velocity
        self.last_time = current_time
        
        # Check distance and rotation conditions
        distance_threshold_met = self.total_distance >= self.distance_threshold
        rotation_threshold_met = filtered_rotation >= self.rotation_threshold
        
        # Reset distance if threshold met
        if distance_threshold_met:
            self.total_distance = 0.0
        
        return distance_threshold_met, rotation_threshold_met
