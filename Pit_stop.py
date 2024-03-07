import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import requests
import json
URL = ' https://api.airtable.com/v0/appNjaGBkp373VxXa/Tasks'
Headers = {'Authorization':'Bearer patLKfBRrXM982KBX.8a4a2b6d6124cf86fec4f8247e5fd5bd83f63665c1abce948f6afaf73d435533'}
class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
    def publish_velocities(self):
        r = requests.get(url = URL, headers = Headers, params = {})
        data = r.json()
        file_path = 'airtable_data.json'
        records = data.get('records', [])
        try:
            linear_velocity = data['records'][1]['fields']['Values']
            angular_velocity = data['records'][0]['fields'] ['Values']
        except KeyError:
            linear_velocity = 0
            angular_velocity = 0
        try:
            linear_velocity = float(linear_velocity)
            angular_velocity = float(angular_velocity)
        except ValueError or KeyError:
            linear_velocity = 0
            angular_velocity = 0
        twist = Twist()
        try:
            linDist = linear_velocity
            twist.linear.x = linDist
            angDist = angular_velocity
            twist.angular.z = angDist
        except AssertionError:
            linDist = float(0)
            twist.linear.x = float(0)
            angDist = float(0)
            twist.angular.z = float(0)
        print(twist.linear.x)
        print(twist.angular.z)
        self.publisher_.publish(twist)
        #return cont
def main(args=None):
    rclpy.init(args=args)
    node1 = PublisherNode()
    try:
        while True:
            # run function to move robot
            node1.publish_velocities()
    except KeyboardInterrupt:
        print('\nCaught Keyboard Interrupt')
        # Destroys the node that was created
        node1.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
