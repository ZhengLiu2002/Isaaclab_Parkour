# 艾萨克实验室（Isaaclab）跑酷项目

基于艾萨克实验室的跑酷运动

基础模型：[极限跑酷](https://extreme-parkour.github.io/)

https://github.com/user-attachments/assets/aa9f7ece-83c1-404f-be50-6ae6a3ba3530

## 安装方法

```
cd IsaacLab                                   # 进入艾萨克实验室工作区
git clone https://github.com/CAI23sbP/Isaaclab_Parkour.git
cd Isaaclab_Parkour
pip install -e .                              # 安装根包

cd parkour_tasks
pip install --no-build-isolation -e .         # 避免网络获取；toml 文件已包含在内
# 如果之前装过旧版本失败，可先卸载再重装：
# pip uninstall -y parkour_tasks && pip install --no-build-isolation -e .
```

## 训练策略的方法

### 1.1 训练教师策略

```
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless
```

### 1.2 训练学生策略

```
python scripts/rsl_rl/train.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless
```

## 运行策略的方法

### 2.1 预训练教师策略

通过此[链接](https://drive.google.com/file/d/1JtGzwkBixDHUWD_npz2Codc82tsaec_w/view?usp=sharing)下载教师策略

### 2.2 运行教师策略

```
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16
```

[Screencast from 2025년 08월 16일 12시 43분 38초.webm](https://github.com/user-attachments/assets/ff1f58db-2439-449c-b596-5a047c526f1f)

### 2.3 评估教师策略

```
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 
```

### 3.1 预训练学生策略

通过此[链接](https://drive.google.com/file/d/1qter_3JZgbBcpUnTmTrexKnle7sUpDVe/view?usp=sharing)下载学生策略

### 3.2 运行学生策略

```
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16
```

https://github.com/user-attachments/assets/82a5cecb-ffbf-4a46-8504-79188a147c40

### 3.3 评估学生策略

```
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 
```

## 在艾萨克实验室中部署的方法

[Screencast from 2025년 08월 20일 18시 55분 01초.webm](https://github.com/user-attachments/assets/4fb1ba4b-1780-49b0-a739-bff0b95d9b66)

### 4.1 部署教师策略

```
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 
```

### 4.2 部署学生策略

```
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 
```

## 测试模块

```
cd parkour_test/ ## 你可以在这里测试你的模块
```

## 可视化控制（跑酷视口相机控制器）

```
按 1 或 2：进入环境
按 8：相机向前
按 4：相机向左
按 6：相机向右
按 5：相机向后
按 0：使用自由相机（可使用鼠标）
按 1：不使用自由相机（默认）
```

## 如何进行仿真到仿真或仿真到现实的部署

这是未来的工作，我会尽快开放这个仓库

* [x] 仿真到仿真：从艾萨克实验室到 MuJoCo
* [ ] 仿真到现实：从艾萨克实验室到现实世界

查看此[仓库](https://github.com/CAI23sbP/go2_parkour_deploy)

### 待办事项列表

* [x] 开放训练教师模型的代码
* [x] 开放训练蒸馏的代码
* [x] 通过演示开放在艾萨克实验室中部署策略的代码：代码参考[网站](https://isaac-sim.github.io/IsaacLab/main/source/overview/showroom.html)
* [x] 开放通过仿真到仿真（MuJoCo）部署策略的代码
* [ ] 开放在现实世界中部署策略的代码

## 引用

如果你将此代码用于你的研究，你**必须**引用以下论文：

```
@article{cheng2023parkour,
title={Extreme Parkour with Legged Robots},
author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
journal={arXiv preprint arXiv:2309.14341},
year={2023}
}
```

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```

```
Copyright (c) 2025, Sangbaek Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software …

The use of this software in academic or scientific publications requires
explicit citation of the following repository:

https://github.com/CAI23sbP/Isaaclab_Parkour
```

## 联系我们

```
sbp0783@hanyang.ac.kr
```