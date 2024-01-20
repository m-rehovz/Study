#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose



def pose_callback(pose):
    global pose1
    pose1 = Pose()
    pose1 = pose
    
  

if __name__ == '__main__':
    rospy.init_node("lab2_controller")
#   rospy.loginfo("Controller node started..")


    rate = rospy.Rate(2)
    subscriber = rospy.Subscriber("ns1_336792/turtle1/pose", Pose, callback=pose_callback) #читаем координаты черепашки1
    pub1 = rospy.Publisher("ns1_336792/turtle1/cmd_vel", Twist, queue_size=10)  # сюда будем публиковать скорость для черепашки1
    def move_turt(x,y):
        global pub1
        msg = Twist()
    
        msg.linear.x = x
        msg.linear.y = y
        msg.linear.z = 0

        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = 0

        #if pose1.x >= x+5.5 and pose1.y >= y+5:
        #    msg.linear.x = 0
        #    msg.linear.y = 0
        #    msg.linear.z = 0

            #msg.angular.x = 0
            #msg.angular.y = 0
            #msg.angular.z = 0
            
        pub1.publish(msg)
   


    
    move_turt(3,3)
    rate.sleep()
    rospy.spin()


    #pub2 = rospy.Publisher("ns2_336792/turtle1/cmd_vel", Twist, queue_size=10)
------------------------------------------
#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import Twist # geometry_msgs must be in package.xml also!!!

x=0

if __name__ == '__main__':
    rospy.init_node("draw_circle")
    rospy.loginfo("Node started...")
    rate = rospy.Rate(2)

    pub = rospy.Publisher("ns1_336792/turtle1/cmd_vel", Twist, queue_size=10)
    msg = Twist()
    while not rospy.is_shutdown():
        #publish cmd vel
        
        msg.linear.x = 1
        #msg.angular.z = 1.2
        x += 1
        if x > 2:
            break
        pub.publish(msg)
        rospy.loginfo("CMD upd...")
        rate.sleep()