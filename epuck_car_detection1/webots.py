# Import necessary libraries
import cv2  # OpenCV library for image processing
import numpy as np  # NumPy for numerical operations
from ultralytics import YOLO  # YOLO (You Only Look Once) model for object detection
from controller import Robot  # Webots Robot controller for controlling the robot
import time  # For time-related functions, used for saving images at intervals

# Set the time step for simulation (in milliseconds) controller will run once every 64 milliseconds.
TIME_STEP = 64

# Initialize the Webots Robot instance
robot = Robot()

# Enable distance sensors (proximity sensors for obstacle detection)
sensors = []  # Empty list to store sensor objects
sensor_names = [
    "ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"
]  # Names of the proximity sensors on the robot

# Loop to initialize and enable all the sensors
for name in sensor_names:
    sensor = robot.getDevice(name)  # Get sensor by name
    sensor.enable(TIME_STEP)  # Enable sensor with the given time step
    sensors.append(sensor)  # Add the sensor object to the list

# Enable the motors (left and right motors for movement control)
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

# Set the motors to velocity control (infinite position)
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Define the base speed of the robot (max speed in rad/s for the e-puck)
BASE_SPEED = 6.28

# Enable the robot's camera for image capturing
camera = robot.getDevice('camera')
camera.enable(TIME_STEP)

# Load the pre-trained YOLO model for object detection
model = YOLO('path_to_your_model.pt')  # Path to your YOLO model file (change as needed)

# Function to save the annotated image with bounding boxes and labels
def save_annotated_image(frame):
    timestamp = int(time.time())  # Get the current timestamp (used for filename)
    filename = f"annotated_image_{timestamp}.png"  # Create filename with timestamp
    cv2.imwrite(filename, frame)  # Save the frame as an image file
    print(f"Annotated image saved as {filename}")  # Print confirmation

# Function to retrieve the values from the distance sensors
def get_sensor_values():
    """Retrieve and return all sensor readings"""
    return [sensor.getValue() for sensor in sensors]  # Get values from all sensors

# Function for obstacle avoidance, adjusting motor speeds based on sensor readings
def avoid_obstacles(sensor_values):
    """
    Calculate motor speeds to avoid obstacles based on the sensor readings.
    """
    left_speed = BASE_SPEED  # Set the initial motor speeds (move forward)
    right_speed = BASE_SPEED
    threshold = 80.0  # Threshold for detecting obstacles

    # Check if there is an obstacle in the front (ps0 or ps7 sensor)
    if sensor_values[0] > threshold or sensor_values[7] > threshold:
        left_speed = -BASE_SPEED  # Reverse left motor (move backward)
        right_speed = BASE_SPEED  # Turn right
    # Check if there is an obstacle on the right (ps5 sensor)
    elif sensor_values[5] > threshold:
        left_speed = BASE_SPEED  # Move left
        right_speed = 0.5 * BASE_SPEED  # Slow down right motor to turn
    # Check if there is an obstacle on the left (ps2 sensor)
    elif sensor_values[2] > threshold:
        left_speed = 0.5 * BASE_SPEED  # Slow down left motor to turn right
        right_speed = BASE_SPEED

    return left_speed, right_speed  # Return the calculated motor speeds

# Main simulation loop, runs until the simulation ends
while robot.step(TIME_STEP) != -1:
    # Get the current sensor readings
    sensor_values = get_sensor_values()

    # Use the sensor readings to calculate motor speeds for obstacle avoidance
    left_speed, right_speed = avoid_obstacles(sensor_values)

    # Apply the calculated motor speeds to control the robot's movement
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)

    # Capture an image from the robot's camera
    width = camera.getWidth()  # Get image width
    height = camera.getHeight()  # Get image height
    image = camera.getImage()  # Capture image from camera

    # Convert the Webots image to an OpenCV-compatible format (NumPy array)
    np_image = np.frombuffer(image, np.uint8).reshape((height, width, 4))  # Convert to NumPy array
    frame = cv2.cvtColor(np_image, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR format

    # Run YOLO object detection on the captured image
    results = model.predict(frame)

    # Loop through all detected objects
    for result in results[0].boxes.data:
        # Extract coordinates, confidence, and class of the detected object
        x1, y1, x2, y2, conf, cls = result.tolist()
        
        # Check if the detected object is a 'car' (class 0 for cars in YOLO model)
        if int(cls) == 0:  # Change '0' to the correct class ID for your object
            label = f"Car: {conf:.2f}"  # Create label with confidence score
            
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add the label text to the image
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with object detection annotations for debugging purposes
    cv2.imshow("Car Detection", frame)

    # Save the annotated image to disk (call the function every frame)
    save_annotated_image(frame)

    # Check if the user pressed the ESC key to quit the simulation
    if cv2.waitKey(1) == 27:  # 'ESC' key code is 27
        break

# Cleanup: Close any OpenCV windows that were opened during the process
cv2.destroyAllWindows()
