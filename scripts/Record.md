# Records for robot control experiments

## 1. Moveit! Control Part

### 1.1 rostopic list（可能有用)

```Rostopic
/execute_trajectory/cancel
/execute_trajectory/feedback
/execute_trajectory/goal
/execute_trajectory/result
/execute_trajectory/status

/gazebo/link_states
/gazebo/model_states
/gazebo/set_link_state
/gazebo/set_model_state

/joint_states
/mirobot/arm_joint_controller/command
/mirobot/arm_joint_controller/follow_joint_trajectory/cancel
/mirobot/arm_joint_controller/follow_joint_trajectory/feedback
/mirobot/arm_joint_controller/follow_joint_trajectory/goal
/mirobot/arm_joint_controller/follow_joint_trajectory/result
/mirobot/arm_joint_controller/follow_joint_trajectory/status
/mirobot/arm_joint_controller/state
/mirobot/joint_states

/move_group/cancel
/move_group/display_contacts
/move_group/display_planned_path
/move_group/feedback
/move_group/goal
/move_group/monitored_planning_scene
/move_group/ompl/parameter_descriptions
/move_group/ompl/parameter_updates
/move_group/plan_execution/parameter_descriptions
/move_group/plan_execution/parameter_updates
/move_group/planning_scene_monitor/parameter_descriptions
/move_group/planning_scene_monitor/parameter_updates
/move_group/result
/move_group/sense_for_plan/parameter_descriptions
/move_group/sense_for_plan/parameter_updates
/move_group/status
/move_group/trajectory_execution/parameter_descriptions
/move_group/trajectory_execution/parameter_updates
```



### 1.2 轨迹规划器

利用Moveit!自带的算法，可以实现给定轨迹的巡迹功能，具体的代码可以参考mirobot_circle_test.py文件，大体流程即给定轨迹，采样一些关键点，将其输入给move_group中使得机械臂按照去这些关键点来实现追踪轨迹。

坑：机械臂工作空间并不代表，机械臂可以一直绕一个方向转，如果fail生成轨迹，建议在$\pi$前面加个“-”号。
