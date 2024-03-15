# ME35 P6: program to navigate a Create3 through maze by turning 6 inches away from recognized objects

import rclpy
from rclpy.node import Node

# for rotation
from rclpy.action import ActionClient
from irobot_create_msgs.action import RotateAngle

# for cmd_vel
from geometry_msgs.msg import Twist

# for model
from picamera2 import Picamera2
from libcamera import controls
from keras.models import load_model
import cv2
import numpy as np
import sys

# for IR sensor
from rclpy.qos import qos_profile_sensor_data
from irobot_create_msgs.msg import IrIntensityVector
import time

# PARAMETERS
t = 0.25 # [s] time between each movement
s = 0.25 # [m/s] speed for motors
conf_thresh = 0.90 # model's confidence threshold
dist_thresh = 8 # distance threshold
# Index:Object
# 0:Bear
# 1:Cube
# 2:Elephant
# 3:Kiwi
# 4:Mario
# 5:Mug
# 6:Vader
# (Nothing doesn't need an index)
L = {0} # class indeces
R = {1,2,3,4,5,6}
LAST = 2 # special last class

# at 6 inches away, IR reading is ~50 --> 50 * gain = 6 --> gain = 0.12
gains = [1, 0.01,0.15,5,0.3,0.75,0.136,5] # Object multipliers to normalize IR sensor readings
cubeGain=[0.1053,0.2609] # [high. low]

# true: red, blue, green
# false: yellow, orange, white
cubeHigh = False

# array to track how many times elephant was seen
undetect = 0
udl = 3

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

camera = Picamera2()
camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})  # Sets auto focus mode
camera.start()  # Start the picam
time.sleep(1) # Give picamera time to start

# class to move the robot
class RotateNode(Node):
   def __init__(self):
       super().__init__('rotate_node')  # Initialize Node
       self.action_client = ActionClient(self, RotateAngle, 'rotate_angle') # Create client for rotate_angle
       self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)  # Create publisher for /cmd_vel

   def send_rotation_goal(self, angle_rad):
       goal_msg = RotateAngle.Goal()
       goal_msg.angle = angle_rad  # [rad] angle

       self.get_logger().info('Sending rotation goal...')
       self.action_client.wait_for_server()

       send_goal_future = self.action_client.send_goal_async(goal_msg)

       if send_goal_future.result() is not None:
           self.get_logger().info('Goal accepted')
       else:
           self.get_logger().info('Goal rejected')

   def goal_response_callback(self, future):
       goal_handle = future.result()
       if not goal_handle.accepted:
           self.get_logger().info('Goal rejected :(')
           return
       self.get_logger().info('Goal accepted :)')
       goal_handle.get_result_async(self.result_callback)

   def result_callback(self, future):
       result = future.result().result
       if result:
           self.get_logger().info('Rotation completed successfully')
       else:
           self.get_logger().info('Rotation failed')

   def send_goal(self, goal_msg):
       self.send_goal_future = self.action_client.send_goal_async(goal_msg)
       self.send_goal_future.add_done_callback(self.goal_response_callback)

   def publish_linear_speed(self, linear_speed):
       twist_msg = Twist()
       twist_msg.linear.x = float(linear_speed)
       twist_msg.linear.y = float(linear_speed)
       twist_msg.linear.z = float(linear_speed)

       self.get_logger().info('Publishing linear velocity...')
       self.cmd_vel_pub.publish(twist_msg)

# class to get distance values from the Create3's IR obstable sensors
class IRSubscriber(Node):
   def __init__(self):
       # calls constructor and names node
       super().__init__('IR_subscriber')

       self.cl = 0.0 # center left sensor

       # node is subscribing to the IrIntensityVector type over the '/ir_intensity' topic.      
       self.subscription = self.create_subscription(
           IrIntensityVector, '/ir_intensity', self.listener_callback,
           qos_profile_sensor_data)

   # This callback function sets what it hears to self.cl
   def listener_callback(self, msg:IrIntensityVector):
       self.cl = self.get_distance(msg)

   #subscriber's callback listens and as soon as it receives the message, this function runs
   def get_distance(self, msg):
       for reading in msg.readings:
           if reading.header.frame_id == "ir_intensity_front_center_left":
               self.cl = reading.value

       return self.cl

def main(args=None):

   global t,s,thresh,gains,L,R,LAST,undetect
   rclpy.init(args=args)

   #initialize nodes
   move = RotateNode()
   sensor = IRSubscriber()
 
   try:
       
       while True:

           # GETTING IMAGE:
          
           image = camera.capture_array("main")
         
           # Convert to RGB
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         
           # Show live video feed
           cv2.imshow("Live Feed", image)

           # Resize the raw image into (224-height,224-width) pixels
           image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

           # Make the image a numpy array and reshape it to the model's input shape
           image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

           # Normalize the image array
           image = (image / 127.5) - 1

           # PREDICT WITH MODEL
           prediction = model.predict(image)
           index = np.argmax(prediction)
           print('INDEX: ', index)
           class_name = class_names[index]
           confidence_score = prediction[0][index]

           # Log prediction and confidence score
           move.get_logger().info(f"Class: {class_name[2:]}, Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%")
           #print('_____________________________________________________\n',class_name[2:])
         
           print('OBJECT: ', class_name[2:])
           print('CONFIDENCE: ', str(round(confidence_score*100)))
           print(prediction)

           # get IR sensor readings
           rclpy.spin_once(sensor, timeout_sec=0.5)

           # set value from IR sensor
           IR_reading = sensor.cl

           #MULTIPLY BY GAIN TO CONVERT IR READING TO INCHES
           if index == 1:
               # if the object is a cube, the gain differs for different sides
               if cubeHigh == True:
                   distance = IR_reading*cubeGain[0]
               else:
                   distance = IR_reading*cubeGain[1]
           else:
               print (gains[index])
               distance = IR_reading*gains[index]

           print('DISTANCE: ', str(distance))

           # If the object is an elephant, too small for accurate distance readings --> move set amount then turn
           if confidence_score >= conf_thresh and index == 2 or index == 4 or index == 6:
               global undetect, udl
               #increase elephants array
               undetect += 1
               print('UNDECTED:, ', undetect)
           if undetect >= udl:
                move.publish_linear_speed(0)
                if index in L: 
                    move.send_rotation_goal(1.5708)
                    undetect = 0
                if index in R:
                    move.send_rotation_goal(-1.5708)
                    undetect = 0
                else:
                    print("Last object!")
                    for i in range(5):
                       move.send_rotation_goal(-0.25)
                       move.send_rotation_goal(0.5)
                    for i in range(100):
                       move.send_rotation_goal(6.2831)

           # if confident in object and correct distance away, then rotate
           elif confidence_score >= conf_thresh and distance >= dist_thresh:
             
               move.publish_linear_speed(0)

               if index in L:  # Check if index is in left set
                   print('DISTANCE: ', distance)
                   print('LEFT')
                   move.send_rotation_goal(1.5708)  # 90° = π/2 rad, positive → CCW
               if index in R:  # Check if index is in is right set
                   print('DISTANCE: ', distance)
                   print('RIGHT')
                   move.send_rotation_goal(-1.5708)  # Negative rotation for right
               if index == LAST:
                   # does a little dance... yipee!
                   print("Last object!")
                   for i in range(5):
                       move.send_rotation_goal(-0.25)
                       move.send_rotation_goal(0.5)
                   for i in range(100):
                       move.send_rotation_goal(6.2831)
               # if the object is just detected at the correct distance, then continue moving forward
               else:
                   print('DISTANCE: ', distance)
                   print('STRAIGHT')
                   move.get_logger().info('Condition not met, no rotation performed.')
                   move.publish_linear_speed(s)
                   time.sleep(t)

           # If confidence score is below threshold, publish linear value to /cmd_vel
           else:
               print('????? STRAIGHT ?????')
               move.publish_linear_speed(s)
               time.sleep(t)

           # Listen to the keyboard for presses.
           keyboard_input = cv2.waitKey(1)

           # 27 is the ASCII for the esc key
           if keyboard_input == 27:
               break

       # exit things in use
       camera.release()
       cv2.destroyAllWindows()
       rclpy.spin(move)
       rclpy.spin(sensor)
       move.destroy_node()
       sensor.destroy_node()
       rclpy.shutdown()

   except KeyboardInterrupt:
       print('All done')

if __name__ == '__main__':
   main()
