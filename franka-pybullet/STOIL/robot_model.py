import numpy as np
import time
import pybullet as p


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
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=30, cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])

        p.resetSimulation()
        p.setTimeStep(self.stepsize)
        p.setRealTimeSimulation(self.realtime)
        p.setGravity(0, 0, -9.81)

        # load models
        p.setAdditionalSearchPath("../models")

        self.plane = p.loadURDF("plane/plane.urdf", useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=.95)

        self.robot = p.loadURDF("panda/panda.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        
        # robot parameters
        self.dof = p.getNumJoints(self.robot) - 1   # Virtual fixed joint between the flange and last link
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
            self.target_pos.append((self.q_min[j] + self.q_max[j])/2.0)
            self.target_torque.append(0.)
            # set damping
            p.changeDynamics(self.robot, j, linearDamping=0.1, angularDamping=100)

        self.reset()

    def reset(self):
        self.t = 0.0        
        self.control_mode = "torque"
        for j in range(self.dof):
            self.target_pos[j] = (self.q_min[j] + self.q_max[j])/2.0
            self.target_torque[j] = 0.
            p.resetJointState(self.robot, j, targetValue=self.target_pos[j])

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

    def setControlMode(self, mode) -> object:
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
                                    targetVelocities=self.target_vel
                                    )

    def setTargetTorques(self, target_torque):
        self.target_torque = target_torque
        # 控制单个关节的运动
        for j in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(self.robot, j, p.TORQUE_CONTROL, force=target_torque[j-1]*10)

        # 同时控制多个关节的运动
        # p.setJointMotorControlArray(self.robot, self.joints, p.TORQUE_CONTROL, self.target_torque)

    def getJointStates(self):
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

    def traj_torque_control(self, joints_torque_array):
        print(joints_torque_array)
        duration = 256*0.025
        num = 0
        for i in range(int(duration/1e-3)):
            if i % int(0.025 / 1e-3) == 0 and num < joints_torque_array.shape[0]:
                self.setTargetTorques(joints_torque_array[num, :]*1000)
                num += 1
            self.step()
            print(self.t)
            print(self.target_torque)
            time.sleep(self.stepsize)
