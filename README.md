# Isaaclab Parkour 项目

基于艾萨克实验室的跑酷运动

基础模型：[极限跑酷](https://extreme-parkour.github.io/)

https://github.com/user-attachments/assets/aa9f7ece-83c1-404f-be50-6ae6a3ba3530

## 安装方法

```bash
cd IsaacLab                                   # 进入艾萨克实验室工作区
git clone https://github.com/CAI23sbP/Isaaclab_Parkour.git
cd Isaaclab_Parkour
pip install -e .                              # 安装根包

cd parkour_tasks
pip install --no-build-isolation -e .         # 避免网络获取；toml 文件已包含在内
# 如果之前装过旧版本失败，可先卸载再重装：
# pip uninstall -y parkour_tasks && pip install --no-build-isolation -e .
```

## 清理缓存
```bash
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
```

## 训练策略的方法

```bash
conda activate isaaclab
```

### Galileo 训练流程（自动跳→混合）

- 课程逻辑：`layout="auto"` 先在跳跃列练习低杆跳（jump_train），平均 terrain_level 达阈值后自动切换到竞赛布局（20/30/40/50），并以小概率回到 jump_train 防遗忘。
- 奖励：跳跃奖励权重高（3.0），钻爬惩罚加大（-2.0），引导先学跳跃再学策略切换。
- 教师建议开启特权杆高观测（默认），学生保持非特权。

**单机单卡示例**
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_Parkour
LOG_RUN_NAME=galileo_teacher python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-Parkour-Teacher-v0 \
  --num_envs 8192 --max_iterations 50000 --run_name auto --headless

LOG_RUN_NAME=galileo_student python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-Parkour-Student-v0 \
  --num_envs 4096 --max_iterations 50000 --run_name auto --headless
```

**多卡分布式（4 卡示例）**
```bash
cd /home/lz/Project/IsaacLab/Isaaclab_Parkour


# 清残留
pkill -f isaaclab.python.kit || true
pkill -f torchrun || true

# 环境变量（禁止 P2P/IB/SHM，指定网卡，收窄 NCCL）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_SOCKET_IFNAME=ens3f3
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=INFO   # 或直接 unset

# 可选：为了更清晰堆栈，定位时再加
# export CUDA_LAUNCH_BLOCKING=1


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/rsl_rl/train.py \
  --task Isaac-Galileo-Parkour-Teacher-v0 \
  --distributed --num_envs 2048 --max_iterations 50000 \
  --run_name debug-4g-safe --device cuda:0
```
- `LOG_RUN_NAME` 环境变量可固定日志目录名，日志路径为 `logs/rsl_rl/<experiment>/<LOG_RUN_NAME>_<run_name>`。
- `--num_envs` 为每卡环境数，按显存调整。

**Play 可视化**
```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-Parkour-Teacher-Play-v0 \
  --checkpoint logs/rsl_rl/<exp>/<run>/checkpoints/ckpt_<iter>.pt \
  --num_envs 16 --video_length 500
```
- Play 默认 `layout="competition"` 展示 20/30/40/50 固定栏杆；可在 cfg 中切换为 `fixed`（5/10/20/30/40/50 档）或 `auto` 复现训练逻辑。
- `--task` 参数需要使用已注册的 Gym 环境名（例如 `Isaac-Galileo-Parkour-Teacher-v0`），不要传入 cfg 文件路径；可运行 `python list_envs.py` 查看当前可用的环境 ID。

## 运行策略的方法

### 2.1 预训练教师策略

通过此[链接](https://drive.google.com/file/d/1JtGzwkBixDHUWD_npz2Codc82tsaec_w/view?usp=sharing)下载教师策略

### 2.2 可视化教师策略

```bash
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --num_envs 16

# Galileo 教师（四栏杆满难度）
python scripts/rsl_rl/play.py --task Isaac-Galileo-Parkour-Teacher-Play-v0 --num_envs 16 --enable_cameras
```

[Screencast from 2025년 08월 16일 12시 43분 38초.webm](https://github.com/user-attachments/assets/ff1f58db-2439-449c-b596-5a047c526f1f)

### 2.3 可视化教师策略

```bash
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Eval-v0 

# Galileo 教师评估
python scripts/rsl_rl/evaluation.py --task Isaac-Galileo-Parkour-Teacher-Play-v0 --num_envs 16 --enable_cameras
```

### 3.1 预训练学生策略

通过此[链接](https://drive.google.com/file/d/1qter_3JZgbBcpUnTmTrexKnle7sUpDVe/view?usp=sharing)下载学生策略

### 3.2 运行学生策略

```bash
python scripts/rsl_rl/play.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --num_envs 16

# Galileo 学生（深度视觉）
python scripts/rsl_rl/play.py --task Isaac-Galileo-Parkour-Student-Play-v0 --num_envs 16 --enable_cameras
```

https://github.com/user-attachments/assets/82a5cecb-ffbf-4a46-8504-79188a147c40

### 3.3 评估学生策略

```bash
python scripts/rsl_rl/evaluation.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Eval-v0 

# Galileo 学生评估
python scripts/rsl_rl/evaluation.py --task Isaac-Galileo-Parkour-Student-Play-v0 --num_envs 16 --enable_cameras
```

## 在 Isaac Lab 中部署的方法

[Screencast from 2025년 08월 20일 18시 55분 01초.webm](https://github.com/user-attachments/assets/4fb1ba4b-1780-49b0-a739-bff0b95d9b66)

### 4.1 部署教师策略

```bash
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 

# Galileo 部署（教师）
python scripts/rsl_rl/demo.py --task Isaac-Galileo-Parkour-Teacher-Play-v0 --num_envs 8 --enable_cameras
```

### 4.2 部署学生策略

```bash
python scripts/rsl_rl/demo.py --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 

# Galileo 部署（学生）
python scripts/rsl_rl/demo.py --task Isaac-Galileo-Parkour-Student-Play-v0 --num_envs 8 --enable_cameras
```

## 测试模块

```bash
cd parkour_test/ ## 你可以在这里测试你的模块
```

## 可视化控制（跑酷视口相机控制器）

```bash
按 1 或 2：进入环境
按 8：相机向前
按 4：相机向左
按 6：相机向右
按 5：相机向后
按 0：使用自由相机（可使用鼠标）
按 1：不使用自由相机（默认）
```

### 5.1 使用 tensorboard  

```bash
cd /home/lz/Project/IsaacLab/Isaaclab_Parkour
tensorboard --logdir logs --port 6006
```

# git Using

```bash
# 1. 撤销最近一次 commit (把大文件退回到暂存区)
git reset --soft HEAD~1

# 2. 运行清理脚本 (把大文件从 git 追踪中移除)
./clean_git.sh

# 3. 重新提交 (这次就不包含大文件了)
git commit -m "chore: remove large files and retry"

# 4. 推送
git push
```
## 多卡训练常见故障与应急环境变量

### 症状与根因
- 报错：`CUDA error: an illegal memory access was encountered` 出现在第一次 NCCL 广播阶段，多数与服务器 IOMMU 开启或 PCIe 链路不平衡（x8 带宽）导致 P2P 受限有关。
- 表现：NCCL 构建通信环卡死/崩溃；P2P 带宽/延迟测试很低；单卡正常，多卡失败。

### 现有的“安全模式”环境变量（已验证可让 4 卡跑通）
```bash
export NCCL_P2P_DISABLE=1          # 禁用 GPU 间 P2P，绕过 IOMMU/PCIe 受限导致的非法访存
export NCCL_IB_DISABLE=1           # 禁用 InfiniBand，强制走 socket
export NCCL_SHM_DISABLE=1          # 禁用 NCCL 共享内存通道，避免 SHM 权限/容量问题
export NCCL_SOCKET_IFNAME=ens3f3   # 指定网卡接口，避免 NCCL 绑定错误接口
export NCCL_ALGO=Ring              # 固定 Ring 算法，减少拓扑自适应带来的不稳定
export NCCL_PROTO=Simple           # 使用简单协议，降低对链路质量的要求
export NCCL_MIN_NCHANNELS=1        # 减少并行通道数，降低带宽需求
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误上报，避免 NCCL 挂死
export OMP_NUM_THREADS=1           # 限制每进程 CPU 线程数，防止过度占用
export TORCH_DISTRIBUTED_DEBUG=INFO  # 控制 NCCL/分布式日志量，过大可 unset
```

### 推荐启动示例（4 卡）
```bash
# 导出上述变量后运行
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/rsl_rl/train.py \
  --task Isaac-Galileo-Parkour-Teacher-v0 \
  --distributed --num_envs 2048 --max_iterations 50000 \
  --run_name debug-4g-safe --device cuda:0
```

### 如果仍报错的排查顺序
1) 进一步减小 `--num_envs`（例如 1024），或先用 2 卡/3 卡避开带宽差的槽位。  
2) 临时加 `export CUDA_LAUNCH_BLOCKING=1` 获取更清晰堆栈（会变慢）。  
3) 仍不行时，保持上述变量，抓取首个报错前后的日志定位。  

### 长期最优方案（需运维/BIOS）
- 关闭 BIOS 的 IOMMU/VT-d（或 grub 内核参数 `intel_iommu=off`），重启后恢复 P2P。
- 确保 GPU 插在 x16 槽，或至少多卡训练挑同一 NUMA、带宽对称的卡。
- 关闭 IOMMU 后可移除 `NCCL_P2P_DISABLE/NCCL_IB_DISABLE/NCCL_SHM_DISABLE`，保留 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`，再逐步放大 `--num_envs` 以获得最高吞吐。

### 如何更高效利用显卡
- 在安全模式下，逐步提升 `--num_envs`，观察每卡显存与吞吐，找到最优点。  
- 若 rank0 占用更高，可减少可视化或让主进程不做额外负载。  
- 关闭 IOMMU 后，恢复 P2P 并移除禁用变量，吞吐和延迟会显著改善。


可视化窗口移动/锁视角说明（Isaac Sim 常用操作）：

- 按 Tab 回到 viewport；按 Alt+LMB 旋转、Alt+MMB 平移、Alt+RMB/滚轮 缩放。
- 开启自由相机：Viewport 工具条 → “Orbit/Fly” 模式；或按 F 聚焦选中物体，再用鼠标/滚轮导航。
- 若脚本频繁刷新导致视角跳回，可在播放前手动选择 “Perspective” 相机并锁定到自由模式，再开始模拟；必要时关闭 HUD/窗口自动对焦（Viewport 右上角菜单）。
- 在 play.py 运行时可用 --enable_cameras 保持传感器工作，GUI 模式下按 Space 暂停/继续，方便调整视角。



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
