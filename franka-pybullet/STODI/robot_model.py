import pybullet as p
import numpy as np
import time
import pybullet_data
from pprint import pprint


class Panda:
    def __init__(self, stepsize=1e-3, realtime=0) -> object:
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime

        self.control_mode = "torque"

        self.position_control_gain_p = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self.position_control_gain_d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.max_torque = [100, 100, 100, 100, 100, 100, 100]

        # connect pybullet
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)  # 0: disable real-time simulation, 1: enable real-time simulation 启用实时仿真以允许拖动
        p.setGravity(0, 0, -9.81)

        # # example 1：机械臂在地面上
        # # load models
        p.setAdditionalSearchPath("../models")
        self.plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=.95)
        self.robot = p.loadURDF("panda/panda.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        # example 2：机械臂和小方块、小圆球在桌上
        # load models
        # p.setAdditionalSearchPath("../models")
        # self.plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
        # p.changeDynamics(self.plane, -1, restitution=.95)
        # self.robot = p.loadURDF("panda/panda.urdf", basePosition=[0.00000, -0.200000, 1.200000], baseOrientation=[0.000000, 0.000000,
        #                      0.000000, 1.000000], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        # self.ball_1 = p.loadURDF("ball/ball.urdf", basePosition=[0.9, -0.2, 1.3], globalScaling=1)

        # # load other models
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.table_1 = p.loadURDF("table/table.urdf", basePosition=[0.5, -0.2, 0.0],
        #                           baseOrientation=[0, 0, 1, 0], globalScaling=2)
        # self.cube_1 = p.loadURDF("cube.urdf", basePosition=[0.7, -0.2, 1.3], globalScaling=0.10)

        # robot parameters
        self.dof = p.getNumJoints(self.robot) - 1
        if self.dof != 7:
            raise Exception('wrong urdf file: number of joints is not 7')

        self.joints = []
        self.q_min = []
        self.q_max = []
        self.target_pos = []
        self.target_vel = []
        self.target_torque = []

        for j in range(self.dof):
            joint_info = p.getJointInfo(self.robot, j)
            self.joints.append(j)
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            self.target_pos.append((self.q_min[j] + self.q_max[j]) / 2.0)
            self.target_torque.append(0.)
            # set damping
            # p.changeDynamics(self.robot, j, linearDamping=0.1, angularDamping=100)

        # logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "./video.mp4")
        # p.stopStateLogging(logId)

        # 调整初始摄像头视角
        # camera parameters：摄像头到目标位置的距离，摄像头的偏航角（水平旋转角度），摄像头的仰角（垂直旋转角度），摄像头的目标位置，即摄像头对准的点的坐标
        p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

        self.reset()

    def reset(self):
        self.t = 0.0
        self.control_mode = "torque"
        self.target_pos = [0., 0., 0., -1.6, 0., 1.87, 0.]    # 低
        # self.target_pos = [0., -0.7, 0.0, -1.6, 0., 3.5, 0.7]  # 高 new
        for j in range(self.dof):
            self.target_torque[j] = 0.
            p.resetJointState(self.robot, j, targetValue=self.target_pos[j])
        # print("Successful Reset!")
        self.resetController()

    def step(self):
        self.t += self.stepsize
        p.stepSimulation()

    # robot functions
    def resetController(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=[0. for i in range(self.dof)])

    def setControlMode(self, mode):
        if mode == "position":
            self.control_mode = "position"
        elif mode == "velocity":
            self.control_mode = "velocity"
        elif mode == "torque":
            if self.control_mode != "torque":
                self.resetController()
            self.control_mode = "torque"
        else:
            raise Exception('wrong control mode')

    def setTargetPositions(self, target_pos):
        self.target_pos = target_pos
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.target_pos,
                                    forces=self.max_torque,
                                    positionGains=self.position_control_gain_p,
                                    velocityGains=self.position_control_gain_d)

    def setTargetVelocity(self, target_velocity):
        self.target_vel = target_velocity
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=self.target_vel)

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        # 控制单个关节的运动（好像有问题，先别用）
        # for j in range(p.getNumJoints(self.robot)):
        #     p.setJointMotorControl2(bodyUniqueId=self.robot, jointIndex=j, controlMode=p.TORQUE_CONTROL, force=target_torque[j-1])

        # 同时控制多个关节的运动
        p.setJointMotorControlArray(bodyUniqueId=self.robot,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=self.target_torque)

    def getJointStates(self):   # 获取关节的位置和速度 返回值：7*1 的关节位置，7*1 的关节速度
        joint_states = p.getJointStates(self.robot, self.joints)
        joint_pos = [x[0] for x in joint_states]
        joint_vel = [x[1] for x in joint_states]
        return joint_pos, joint_vel

    def solveInverseDynamics(self, pos, vel, acc) -> object:
        return list(p.calculateInverseDynamics(self.robot, pos, vel, acc))

    def solveInverseKinematics(self, pos, ori):
        return list(p.calculateInverseKinematics(self.robot, 7, pos, ori))

    def solveListKinematics(self, joints_array):
        joints_num = joints_array.shape[1] # joints_array should be N * 7
        if joints_num != self.dof:
            raise Exception('wrong joint trajectory: number of joints is not 7')

        original_joint_positions = []
        for j in range(joints_array.shape[1]):
            original_joint_state = p.getJointState(self.robot, j)
            original_joint_positions.append(original_joint_state[0])  # 保存原始关节角度

        end_effector_array = np.zeros(shape=(joints_array.shape[0], joints_array.shape[1]))
        for k in range(joints_array.shape[0]):
            for j in range(joints_array.shape[1]):
                p.resetJointState(self.robot, j, joints_array[k, j])
            link_state = p.getLinkState(self.robot, self.dof - 1)     # question1
            end_effector_array[k, 0: 3] = np.array(link_state[0])
            end_effector_array[k, 3: ] = np.array(link_state[1])

        # 恢复原始关节状态
        for i in range(len(original_joint_positions)):
            p.resetJointState(self.robot, i, original_joint_positions[i])

        return end_effector_array

    def traj_control(self, joints_array):
        duration = 0.5
        num = 0
        for i in range(int(duration/1e-3)):
            if i % int(0.025 / 1e-3) and i < joints_array.shape[0]:
                self.setTargetPositions(joints_array[num, :])
                num += 1
            self.step()
            time.sleep(self.stepsize)

    def traj_vel_control(self, joints_vel_array):
        duration = 1
        num = 0
        for i in range(int(duration/1e-3)):
            if i % int(0.025 / 1e-3) and i < joints_vel_array.shape[0]:
                self.setTargetVelocity(joints_vel_array[num, :])
                num += 1
            self.step()
            time.sleep(self.stepsize)

    def traj_torque_control(self, pos_planned, vel_planned, acc_planned):
        duration = 2  # 256 * 0.025
        num = 1
        self.setControlMode("torque")

        for i in range(int(duration / self.stepsize)):
            if i % 100 == 0:
                print("Simulation time: {:.3f}".format(self.t))

            if i % 10000 == 0:
                self.reset()
                self.setControlMode("torque")

            if i % int(0.005 / self.stepsize) == 0 and num < pos_planned.shape[0]:
                pos_desired_update = pos_planned[num]
                vel_desired_update = vel_planned[num]
                acc_desired_update = acc_planned[num]
                num += 1

            pos, vel = self.getJointStates()

            # acc = [0 for x in pos]
            kv, kp = 10, 100  # 位置和速度的比例控制器增益矩阵，但我懒就先写 1 假装是单位矩阵好了
            acc_feedback = list(10 * (acc_desired_update - kv * (np.array(vel) - vel_desired_update) - kp * (
                    np.array(pos) - pos_desired_update)))

            # solveInverseDynamics 的输入必须是 list 格式的
            joints_torque = self.solveInverseDynamics(list(pos), list(vel), acc_feedback)  # list(acc_planned[num + 1]))
            self.setTargetTorques(joints_torque)
            # self.setTargetTorques([t*0.5 for t in joints_torque])
            # num += 1
            self.step()
            # print(self.t)
            # print(self.target_torque)
            time.sleep(self.stepsize)
    
    def get_current_end_effector(self):
        end_effector_array = np.zeros(shape=(7,))
        link_state = p.getLinkState(self.robot, self.dof - 1)     # question1
        end_effector_array[0: 3] = np.array(link_state[0])
        end_effector_array[3: ] = np.array(link_state[1])

        return end_effector_array

    # 0603 update
    def disconnet(self):
        p.disconnect()

    def demo_mode(self):  # 记录 demo 时，重力设置为0；启用实时仿真；设置关节阻尼
        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(1)
        for j in range(self.dof):
            p.changeDynamics(self.robot, j, linearDamping=0.1, angularDamping=100)

    def read_cube_position(self):
        # 读取方块位置状态
        cube_position, cube_orientation = p.getBasePositionAndOrientation(self.cube_1)
        return cube_position, cube_orientation
        # print("Cube Position:", cube_position)
        # print("Cube Orientation:", cube_orientation)




