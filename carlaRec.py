import carla
import numpy as np
import os
import cv2
from datetime import datetime
import time
import queue
import weakref
import math

from scipy.io import savemat

class DataCollector:
    def __init__(self, host='localhost', port=2000, fps=250):
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0/fps
        self.world.apply_settings(settings)
        
        # Basic settings
        self.fps = fps
        self.baseline = 0.54
        self.output_dir = 'carla_event_dataset'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Camera settings
        self.image_width = 1280
        self.image_height = 720
        self.fov = 90
        
        # Feature tracking settings
        self.feature_position = None
        self.feature_template = None
        self.tracking_radius = 0.5
        self.template_size = 31
        
        # Event generation settings
        self.event_threshold = 0.1
        self.min_events = 1000
        
        # Setup queues and objects
        self.left_queue = queue.Queue()
        self.right_queue = queue.Queue()
        self.cameras = []
        self.feature_marker = None
        self.tracking_vehicle = None
        
        # Camera matrices
        self.camera_matrix = np.array([
            [self.image_width/(2*np.tan(self.fov*np.pi/360)), 0, self.image_width/2],
            [0, self.image_height/(2*np.tan(self.fov*np.pi/360)), self.image_height/2],
            [0, 0, 1]
        ])
        
        # Initialize scene
        self.setup_scene()

    def handle_camera_data(self, image, is_left):
        """Handle camera data with explicit left/right handling"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image_height, self.image_width, 4))
            array = array[:, :, :3]
            
            if is_left:
                self.left_queue.put(array)
                print("Left image captured")
            else:
                self.right_queue.put(array)
                print("Right image captured")
                
        except Exception as e:
            print(f"Error in handle_camera_data: {e}")

    def setup_scene(self):
        try:
            # Setup cameras
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.image_width))
            camera_bp.set_attribute('image_size_y', str(self.image_height))
            camera_bp.set_attribute('fov', str(self.fov))
            camera_bp.set_attribute('sensor_tick', str(1.0/self.fps))
            
            # Get a reasonable spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            if len(spawn_points) > 0:
                spawn_point = spawn_points[0]
            else:
                spawn_point = carla.Transform(carla.Location(x=0, y=0, z=2))
            
            # Define left and right camera transforms relative to spawn point
            left_transform = carla.Transform(
                carla.Location(x=spawn_point.location.x, 
                             y=spawn_point.location.y - self.baseline/2, 
                             z=spawn_point.location.z + 2),
                carla.Rotation(pitch=0, yaw=0, roll=0)
            )
            right_transform = carla.Transform(
                carla.Location(x=spawn_point.location.x, 
                             y=spawn_point.location.y + self.baseline/2, 
                             z=spawn_point.location.z + 2),
                carla.Rotation(pitch=0, yaw=0, roll=0)
            )
            
            # Spawn cameras
            self.left_camera = self.world.spawn_actor(camera_bp, left_transform)
            self.right_camera = self.world.spawn_actor(camera_bp, right_transform)
            
            # Set up callbacks
            self.left_camera.listen(self.handle_camera_data_left)
            self.right_camera.listen(self.handle_camera_data_right)
            
            self.cameras = [self.left_camera, self.right_camera]
            
            # Add vehicle for tracking
            try:
                # Get a vehicle blueprint
                vehicle_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
                
                # Spawn the vehicle slightly ahead of the cameras
                vehicle_transform = carla.Transform(
                    carla.Location(x=spawn_point.location.x + 10, 
                                 y=spawn_point.location.y,
                                 z=spawn_point.location.z),
                    spawn_point.rotation
                )
                
                self.tracking_vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)
                print("Added vehicle for tracking")
                
            except Exception as e:
                print(f"Error adding tracking vehicle: {e}")
            
            # Wait for sensors
            print("Waiting for sensors to be ready...")
            self.world.tick()
            time.sleep(2.0)
            
        except Exception as e:
            print(f"Error in setup_scene: {e}")
            self.cleanup()
            raise

    def handle_camera_data_left(self, image):
        self.handle_camera_data(image, True)

    def handle_camera_data_right(self, image):
        self.handle_camera_data(image, False)

    def collect_data(self, num_sequences=53, start_frame=680, end_frame=689, frame_step=8):
        """Collect multiple sequences of data"""
        try:
            print(f"Starting data collection for {num_sequences} sequences...")
            
            for sequence in range(num_sequences):
                # Create sequence directory with just the number
                self.output_dir = os.path.join('carla_event_dataset', str(sequence))
                os.makedirs(self.output_dir, exist_ok=True)
                
                print(f"\nCollecting sequence {sequence + 1}/{num_sequences}")
                
                # Save Q matrix
                self.generate_q_matrix()
                
                # Save parameters
                self.save_parameters()
                
                # Move the vehicle to a slightly different position for variety
                if self.tracking_vehicle is not None:
                    current_transform = self.tracking_vehicle.get_transform()
                    new_transform = carla.Transform(
                        carla.Location(
                            x=current_transform.location.x + np.random.uniform(-2, 2),
                            y=current_transform.location.y + np.random.uniform(-2, 2),
                            z=current_transform.location.z
                        ),
                        current_transform.rotation
                    )
                    self.tracking_vehicle.set_transform(new_transform)
                    time.sleep(0.5)
                
                # Collect frame 680 (left camera)
                self.collect_frame(680, is_left=True)
                
                time.sleep(1.0)  # pause between frames
                
                # Collect frame 688 (right camera)
                self.collect_frame(688, is_left=False)
                
                time.sleep(0.5)  # Wait between sequences
                
            print("\nCompleted collecting all sequences")
            
        except Exception as e:
            print(f"Error in collect_data: {e}")
        finally:
            self.cleanup()

    def collect_frame(self, frame_number, is_left=True):
        """Collect data for a specific frame"""
        try:
            # Create frame directory
            frame_dir = os.path.join(self.output_dir, f'image_{frame_number}')
            os.makedirs(frame_dir, exist_ok=True)
            print(f"Created directory: {frame_dir}")
            
            queue_to_use = self.left_queue if is_left else self.right_queue
            prefix = 'left' if is_left else 'right'
            
            # Clear queue
            while not queue_to_use.empty():
                queue_to_use.get()
            print(f"Cleared {prefix} queue")
            
            # Capture first image with retry
            print(f"Capturing first {prefix} image...")
            img1 = None
            for _ in range(5):  # Try up to 5 times
                self.world.tick()
                try:
                    img1 = queue_to_use.get(timeout=1.0)
                    if img1 is not None:
                        break
                except queue.Empty:
                    time.sleep(0.1)
                    continue
                    
            if img1 is None:
                raise Exception(f"Failed to capture first {prefix} image")
                
            print(f"Captured first {prefix} image")
            
            # Capture second image with retry
            print(f"Capturing second {prefix} image...")
            img2 = None
            for _ in range(5):  # Try up to 5 times
                self.world.tick()
                try:
                    img2 = queue_to_use.get(timeout=1.0)
                    if img2 is not None:
                        break
                except queue.Empty:
                    time.sleep(0.1)
                    continue
                    
            if img2 is None:
                raise Exception(f"Failed to capture second {prefix} image")
                
            print(f"Captured second {prefix} image")
            
            # Save images
            print(f"Saving {prefix} images...")
            img1_path = os.path.join(frame_dir, f'{prefix}_1.png')
            img2_path = os.path.join(frame_dir, f'{prefix}_2.png')
            
            cv2.imwrite(img1_path, cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(img2_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
            
            # Generate and save events
            print("Generating events...")
            events = self.generate_events(img1, img2, 1.0/self.fps)
            np.save(os.path.join(self.output_dir, f'events_{frame_number}.npy'), events)
            
            # Generate and save ground truth data
            print("Generating ground truth...")
            gt_data = self.get_ground_truth()
            np.save(os.path.join(self.output_dir, f'gt_2d_{frame_number}.npy'), gt_data['2d'])
            
            if frame_number == 680:
                np.save(os.path.join(self.output_dir, 'gt_track.npy'), gt_data['3d'])
            
            # Save timestamp data
            print("Saving timestamp data...")
            timestamps = {
                'frame_ts': frame_number / self.fps,
                'trigger_ts': frame_number/self.fps + 0.5/self.fps
            }
            np.save(os.path.join(self.output_dir, f'ts_{frame_number}.npy'), timestamps['frame_ts'])
            
            if frame_number == 680:
                np.save(os.path.join(self.output_dir, 'ts_trigger_crop.npy'), timestamps['trigger_ts'])
            
            print(f"Successfully collected frame {frame_number}")
            return True
            
        except Exception as e:
            print(f"Error collecting frame {frame_number}: {str(e)}")
            return False

    def generate_events(self, img1, img2, timestamp_diff):
        """Generate events from two consecutive images"""
        events = []
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        # Calculate log intensity difference
        diff = np.log(img2.astype(float) + 1e-6) - np.log(img1.astype(float) + 1e-6)
        
        # Generate events for positive and negative changes
        events_pos = np.where(diff > self.event_threshold)
        events_neg = np.where(diff < -self.event_threshold)
        
        for x, y in zip(*events_pos):
            events.append([x, y, timestamp_diff * np.random.random(), 1])
        for x, y in zip(*events_neg):
            events.append([x, y, timestamp_diff * np.random.random(), -1])
        
        events = np.array(events) if len(events) > 0 else np.array([])
        
        # Ensure minimum number of events
        if len(events) < self.min_events:
            additional_events = self.min_events - len(events)
            random_events = np.array([
                [np.random.randint(0, self.image_width),
                 np.random.randint(0, self.image_height),
                 timestamp_diff * np.random.random(),
                 1 if np.random.random() > 0.5 else -1]
                for _ in range(additional_events)
            ])
            if len(events) > 0:
                events = np.vstack([events, random_events])
            else:
                events = random_events
        
        return events

    def generate_q_matrix(self):
        """Generate and save the Q matrix for stereo calibration"""
        # Calculate focal lengths
        fx = self.image_width/(2*np.tan(self.fov*np.pi/360))
        fy = self.image_height/(2*np.tan(self.fov*np.pi/360))
        cx = self.image_width/2
        cy = self.image_height/2
        
        # Create Q matrix
        Q = np.array([
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, fx],
            [0, 0, -1/self.baseline, 0]
        ])
        
        # Save Q matrix
        np.save(os.path.join(self.output_dir, 'Qmat.npy'), Q)

    def save_parameters(self):
        """Save parameters as Para.mat file"""
        params = {
            'image_width': self.image_width,
            'image_height': self.image_height,
            'fov': self.fov,
            'baseline': self.baseline,
            'fps': self.fps,
            'event_threshold': self.event_threshold,
            'min_events': self.min_events
        }
        
        # Save as Para.mat directly in the output directory
        savemat(os.path.join(self.output_dir, 'Para.mat'), params)

    def get_ground_truth(self):
        """Get ground truth data for the tracked feature"""
        # Get vehicle location as ground truth
        if self.tracking_vehicle is not None:
            location = self.tracking_vehicle.get_location()
            velocity = self.tracking_vehicle.get_velocity()
            
            # Project to 2D
            point_3d = np.array([[location.x], [location.y], [location.z]])
            point_2d = self.camera_matrix @ point_3d
            point_2d = point_2d[:2] / point_2d[2]
            
            return {
                '2d': point_2d.flatten(),
                '3d': {
                    'location': [location.x, location.y, location.z],
                    'velocity': [velocity.x, velocity.y, velocity.z]
                }
            }
        else:
            # Return default values if no vehicle
            return {
                '2d': np.array([self.image_width/2, self.image_height/2]),
                '3d': {
                    'location': [0, 0, 0],
                    'velocity': [0, 0, 0]
                }
            }

    def cleanup(self):
        """Clean up resources"""
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
            
            for camera in self.cameras:
                if camera is not None:
                    camera.destroy()
            
            if self.tracking_vehicle is not None:
                self.tracking_vehicle.destroy()
                
            print("Cleanup completed successfully")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    collector = None
    try:
        collector = DataCollector()
        collector.collect_data(num_sequences=54)  # Will create 53 sequence folders
    except Exception as e:
        print(f"An error occurred in main: {e}")
    finally:
        if collector is not None:
            collector.cleanup()
if __name__ == '__main__':
    main()
