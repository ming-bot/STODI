from robot_model import z1_robot
import time

t = 0.0
stepsize = 1e-3
realtime = 0

robot = z1_robot()
robot.demo_mode()

while True:
    print(robot.getJointStates()[0])
    time.sleep(1)