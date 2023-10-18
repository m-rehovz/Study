#!/usr/bin/env python3

import rospy

# это subscriber и piblisher

from std_msgs.msg import Int32

a = 0   # глобальные переменные
b = 0

def num_callback1(msg):
    global a
    a = msg.data

def num_callback2(msg):
    global b
    b = msg.data

if __name__ == '__main__':
    rospy.init_node("node2")
    rospy.loginfo("Node 2 started...")
    
    rospy.Subscriber("read_msg1", Int32, callback=num_callback1)
    rospy.Subscriber("read_msg3", Int32, callback=num_callback2)
    pub = rospy.Publisher("result_336792", Int32, queue_size=10)
    msg = Int32()
    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        res = a + b
        msg.data = res
        pub.publish(msg)
        rospy.loginfo(msg)
        rate.sleep()



    # rospy.logwarn("Hello world worn")
    # rospy.logerr("Hello world err")

