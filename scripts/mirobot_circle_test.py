#!/usr/bin/python

import rospy
import math
from copy import deepcopy
from moveit_msgs.msg import RobotTrajectory, Constraints, JointConstraint
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
from geometry_msgs.msg import Pose

def init_circle(pose):
    circle_radius = math.sqrt(pose.position.x**2 + pose.position.y**2)
    print("circle radius is : ", circle_radius)
    circle_center_pose = Pose()
    circle_center_pose.position.x = 0.0
    circle_center_pose.position.y = 0.0
    circle_center_pose.position.z = 0.15
    return circle_radius, circle_center_pose

def get_end_pose(move_group):
    end_effector_pose = move_group.get_current_pose().pose
    print("current pose info:\n" , end_effector_pose)
    return end_effector_pose


def calculate_waypoints(radius, center, init_pose):
    waypoints = []
    num_points = 10
    
    for i in range(1, num_points):
        angle = (-math.pi * 2 * i) / (3 * num_points)
        pose = Pose()
        pose.position.x = center.position.x + radius * (math.cos(angle) * init_pose.position.x/radius - math.sin(angle) * init_pose.position.y/radius)
        pose.position.y = center.position.y + radius * (math.sin(angle) * init_pose.position.x/radius + math.cos(angle) * init_pose.position.y/radius)
        pose.position.z = center.position.z
        # 其他姿态信息
        pose.orientation.x = init_pose.orientation.x
        pose.orientation.y = init_pose.orientation.y
        pose.orientation.z = init_pose.orientation.z
        pose.orientation.w = init_pose.orientation.w
        print("\n", pose)
        waypoints.append(deepcopy(pose))
    
    return waypoints

if __name__=='__main__':
    rospy.init_node('mirobot_circle_trajectory_test')
    robot = RobotCommander()
    group = MoveGroupCommander('manipulator')

    # 设置路径规划器的约束条件，这里假设末端执行器的姿态保持不变
    constraints = Constraints()
    joint_constraint = JointConstraint()
    # 添加关节约束
    # ...

    # 设置路径规划的约束条件
    constraints.joint_constraints.append(joint_constraint)
    group.set_path_constraints(constraints)

    # 在笛卡尔空间规划路径
    group.set_start_state_to_current_state()
    end_pose = get_end_pose(group)
    R, circle_center = init_circle(end_pose)
    waypoints = calculate_waypoints(R, circle_center, end_pose)
    # 根据圆心、半径和起始点、终止点或者角度来生成路径上的点
    # ...

    # 生成圆弧轨迹
    (plan, fraction) = group.compute_cartesian_path(
        waypoints,  # waypoints
        0.01,       # eef_step
        0.0,        # jump_threshold
        True)       # avoid_collisions

    # 执行路径
    if fraction == 1.0:
        group.execute(plan)
    else:
        rospy.loginfo("Failed to plan trajectory to follow the circle")