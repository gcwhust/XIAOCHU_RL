# XiaoChu RL (小楚机器人强化学习)

本项目基于 Isaac Lab 和 RSL-RL，用于训练小楚（XiaoChu）双足机器人的强化学习 locomotion 任务。

## 📦 环境配置与安装

### 1. 安装项目依赖
在项目根目录下执行以下命令安装依赖：

```bash
# 使用清华源进行安装
python -m pip install -e exts/bipedal_locomotion/ --index-url https://pypi.org/simple   --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或者使用环境变量方式
PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple python -m pip install -e exts/bipedal_locomotion/ -vvv
```

### 2. 配置环境变量
请根据你本地 Isaac Sim 的安装路径修改以下路径：

```bash
# Isaac Sim 根目录
export ISAACSIM_PATH="/home/user/IsaacSim/_build/linux-x86_64/release"

# Isaac Sim python 可执行文件
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# 将 IsaacSim target-deps python bin 添加到 PATH
export PATH="/home/user/IsaacSim/_build/target-deps/python/bin:$PATH"

# 设置库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## 🚀 快速开始

### 训练模型
使用 `train.py` 脚本开始训练：

```bash
python scripts/train.py --task=Isaac-XC-Blind-Flat-v0 --headless --video
```
*   `--task`: 指定任务名称 (Isaac-XC-Blind-Flat-v0)
*   `--headless`: 无头模式运行（不显示 GUI），适合服务器训练
*   `--video`: 记录训练过程视频

### 评估模型
使用 `play.py` 脚本加载训练好的模型进行演示：

```bash
python scripts/play.py --task=Isaac-XC-Blind-Flat-Play-v0 --video --video_length 500 --headless
```

### 监控训练进度
使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir logs
```

---

## ⚙️ 实验配置概览

*   **环境名称**: `Isaac-XC-Blind-Flat-v0`
*   **机器人**: Xiaochu 双足机器人
*   **算法**: PPO (Proximal Policy Optimization)

### 1. 训练参数 (RSL-RL)
*   **总迭代次数**: 10,000
*   **每环境步数**: 24
*   **环境数量**: 4096
*   **网络架构**:
    *   Policy: [512, 256, 128] (ELU激活)
    *   Critic: [512, 256, 128] (ELU激活)
    *   Encoder: [256, 128] (输出维度 3)

### 2. 机器人配置
*   **执行器**:
    *   **腿部**: PD控制 (Stiffness=40.0, Damping=2.5)
    *   **轮子**: PD控制 (Stiffness=0.0, Damping=0.8)
*   **观测空间**:
    *   **Policy (含噪声)**: 基座角速度, 重力投影, 关节位置/速度, 上次动作
    *   **Critic (无噪声)**: 包含更详细的物理状态信息（线速度, 力矩, 接触力等）
    *   **History**: 历史长度 10 帧
*   **动作空间**:
    *   关节位置 (Offset=0, Scale=0.25)
    *   关节速度 (轮子)

### 3. 奖励设计 (Rewards)
详细权重请参考代码配置。

*   **正向奖励**:
    *   `keep_balance`: 保持平衡
    *   `rew_lin_vel_xy`: 跟踪目标线速度 (XY平面)
    *   `rew_ang_vel_z`: 跟踪目标角速度 (Z轴)
    *   `rew_leg_symmetry`: 腿部动作对称性
*   **负向奖励 (惩罚)**:
    *   `stand_still`: 静止不动
    *   `pen_lin_vel_z`: Z轴剧烈晃动
    *   `undesired_contacts`: 非脚部接触地面
    *   `pen_action_rate` / `pen_action_smoothness`: 动作不平滑
    *   `pen_base_height`: 基座高度异常

### 4. 域随机化 (Domain Randomization)
为了提高 Sim2Real 的鲁棒性，环境中启用了多种随机化：
*   **物理属性**: 摩擦力, 弹性系数, 机器人质量, 质心位置
*   **动力学**: 关节刚度/阻尼, 执行器增益
*   **干扰**: 随机推力 (Push robot)
*   **初始化**: 随机初始位置和关节状态
