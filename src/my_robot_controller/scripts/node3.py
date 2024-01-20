#!/usr/bin/env python3

import rospy

# это publisher

from std_msgs.msg import Int32

if __name__ == '__main__':
    rospy.init_node("node3")
    rospy.loginfo("Node 3 started...")
    rate = rospy.Rate(10)

    pub = rospy.Publisher("read_msg3", Int32, queue_size=10)
    msg = Int32()
    msg = 3


    while not rospy.is_shutdown():
        pub.publish(msg)
        rospy.loginfo("Sending message to node2...")
        rate.sleep()



    # rospy.logwarn("Hello world worn")
    # rospy.logerr("Hello world err")

