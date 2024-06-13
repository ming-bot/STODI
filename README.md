# Proud: Professional Robot On Utilizing Dice
明宝science robotics作品

### 2024/1/11 更新记录

更新了一个python脚本，实现了通过预设的轨迹和move_group进行机械臂的轨迹追踪控制。详情请见内部Record。



### 2024/1/15 问题建模

```
Input: reference_trajectory
output: optimized_trajectory
optimize target: theta

temp_trajectory <—— reference_trajectory
OUTER_LOOP{
	


}

```

insight是，类似于模仿学习，为了完成某一任务（可以简单转化为优化theta<某一个机械臂的运动特性>），人类给出了一条参考的轨迹reference_trajectory，但其实最优的轨迹可能并不完全等价于此时的参考轨迹，而是参考轨迹可以看作是最优轨迹的邻域中采样出的一个轨迹。此时优化的目标即通过优化theta，找出最优轨迹；同时保留机械臂运动的连续性等特点。

同时如果我们还需要对参考轨迹进行评估，最终的优化轨迹和参考轨迹相似度？参考轨迹是否合理？

------

### 2024/6/13

新添加了Demo文件夹，目前架构如下所示：

代码集中在franka-pybullet下，包括：

**Demo**：主要负责论文实验部分的展示；

**model**：备份一些模型信息；

**example**：不用管，是原始pybullet示例自带的；

**src**：记录实验结果等等；

**Stoil**：主要是论文算法；

**Stomp**：主要改进过的Stomp算法；
