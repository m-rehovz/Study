#!/usr/bin/env python

from __future__ import print_function
from practice_3.srv import AddTwoInts, AddTwoIntsResponse
import rospy


if __name__ == "__main__":
    rospy.init_node('client_node')
    rate = rospy.Rate(1)
    print("Sending two ints...")
    service_call = rospy.ServiceProxy('add_two_ints', AddTwoInts)
    response = service_call(2, 4)
    print("Result = ", response)

    rospy.spin()