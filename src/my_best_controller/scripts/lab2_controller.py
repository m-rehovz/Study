#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist # geometry_msgs must be in package.xml also!!!
from turtlesim.msg import Pose

pose1 = Pose()    # pose черепашки_1
pose2 = Pose()    # pose черепашки_2
msg1 = Twist()    # twist черепашки_1
msg2 = Twist()    # twist черепашки_2

def send_coord_to_turtle_2():
    while 1:
        msg2.linear.x = (pose1.x-pose2.x)     # записываем для черепашки2 x_координату черепашки_1
        msg2.linear.y = (pose1.y-pose2.y)     # записываем для черепашки_2 y_координату черепашки_1
        if abs(pose1.x-pose2.x) <= 0.09 and abs(pose1.y-pose2.y) <= 0.09:
            msg2.linear.x = 0
            msg2.linear.y = 0
            break
        pub2.publish(msg2)     # отправляем черепашке_2 координаты её цели


def move_turtle_1(target_x, target_y):
    while 1:
        msg1.linear.x = (target_x-pose1.x)
        msg1.linear.y = (target_y-pose1.y)
        if abs(pose1.x-target_x) <= 0.09 and abs(pose1.y-target_y) <= 0.09:
            msg1.linear.x = 0
            msg1.linear.y = 0
            break

        pub1.publish(msg1)
        send_coord_to_turtle_2()
        rospy.loginfo("CMD upd...")


def pose_callback_1(m):
    global pose1
    pose1 = m

def pose_callback_2(n):
    global pose2
    pose2 = n


if __name__ == '__main__':
    rospy.init_node("lab2_controller")
    rospy.loginfo("Lab2_controller started...")
    rate = rospy.Rate(80)
    sub1 = rospy.Subscriber("ns1_336792/turtle1/pose", Pose, callback=pose_callback_1)   # считывает pose черепашки_1
    sub2 = rospy.Subscriber("ns2_336792/turtle1/pose", Pose, callback=pose_callback_2)  # считывает pose черепашки_2

    pub1 = rospy.Publisher("ns1_336792/turtle1/cmd_vel", Twist, queue_size=10)         # отправляет twist черепашке_1
    pub2 = rospy.Publisher("ns2_336792/turtle1/cmd_vel", Twist, queue_size=10)         # отправляет twist черепашке_2

    move_turtle_1(2,3)
    rospy.loginfo('WOW, you reached the 1st point!')
    rospy.loginfo(pose1)
    rospy.loginfo(pose2)
    
    move_turtle_1(4,2)
    rospy.loginfo('WOW, you reached the 2nd point!')
    rospy.loginfo(pose1)
    rospy.loginfo(pose2)
    
    move_turtle_1(5,4)
    rospy.loginfo('WOW, you reached the 3rd point!')
    rospy.loginfo(pose1)
    rospy.loginfo(pose2)

    rospy.spin()
