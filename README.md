# Stochastic Trajectory Optimization for Demonstration Imitation（STODI）

## 1. The Folder structure:

Main code files are stored in the **franka-pybullet** folder, which includes:

**Demo**: Mainly responsible for the presentation of the experimental part of the paper;

**model**: Back up some model information;

**src**: Record experimental results, etc.

**STODI**: Mainly thesis algorithms;

**Stomp**: The main improved Stomp algorithm;

**SpectrumAnalysis**: Some analysis for denoising methods.



## 2. How to use STODI in the pybullet simulator

Firstly, the main simulation environment package is pybullet, you should install pybullet in your environment.

### 2.1  Run the code mentioned in the paper

If you want to run the **improved Stomp algorithm**, you can run this in your command:

```python
python ./Stomp/main.py --expt-name='your expert name'
```

You can replace the `'your expert name'` with what you want. The result files will be written in the `./src/results/` folder.



If you want to run the **STODI algorithm**, you can run this in your command:

```python
python ./STODI/main.py --expt-name='your expert name'
```

You can replace the `'your expert name'` with what you want. The result files will be written in the `./src/results/` folder.



**Important Notion:** You may need to change the file path argument if your path is different with `E:/STODI/franka-pybullet` .



## 3. Other files

Other files in the folders (`Proud_hp, scripts`) are not important or under development. Future work might focus on employing the STODI into the ROS platform, so stay tuned!

The `src` folder contains images and some other supplementary materials for our paper.



## 4. Important Updating! (10.10)

We combine the STOMP and STODI in the Pybullet Folder, which also contains the Panda and Z1 Robot！

We add the obstacle cost into the Stochastic Optimization Process!





This project is licensed under the MIT license.
