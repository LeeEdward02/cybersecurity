# 网络安全博弈论仿真项目 | Network Security Game Theory Simulation

## 项目简介 | Project Overview

本项目是一个基于博弈论的网络防御合作演化仿真平台，用于研究网络拓扑结构和反馈机制对网络安全防御合作演化的影响。项目模拟了网络防御者在不同攻击场景下的合作行为，探索集体防御策略的形成和演化规律。

This project is a game theory-based simulation platform for studying the evolution of cooperative network defense under different attack scenarios. It investigates how network topology and feedback mechanisms influence the evolution of defensive cooperation in cybersecurity contexts.

## 研究背景 | Research Context

本项目基于论文 **"The impact of cyber-attacks on cybersecurity investment game model"** (网络攻击对网络安全投资博弈模型的影响) 进行实现，重点关注：
- 网络结构和自适应攻击行为对集体防御合作演化的影响
- 系统变化参数下防御合作的出现和持续条件
- 网络安全中的群体行为和策略演化规律

### 论文核心思想
该论文研究在网络安全背景下，通过公共物品博弈和攻防博弈的组合模型，分析网络攻击如何影响防御者的投资决策和合作行为。论文探讨了不同网络拓扑结构下集体防御的演化动态，以及反馈机制对策略演化的影响。


## 核心特性 | Key Features

- 🎯 **多层次建模**: 结合局部群体互动（公共物品博弈）与个体防御决策
- 🔄 **自适应智能体**: 防御者和攻击者都能根据表现调整策略
- 🌐 **灵活网络拓扑**: 支持多种网络结构研究拓扑效应
- 📊 **全面数据收集**: 追踪合作率、成功指标和策略演化
- 🧩 **模块化设计**: 易于扩展新实验和网络类型

## 项目结构 | Project Structure

```
├── core/                           # 核心模块
│   ├── agents.py                   # 智能体定义（防御者、攻击者）
│   ├── games.py                    # 博弈模型（公共物品博弈、攻防博弈）
│   ├── simulation.py               # 仿真框架基类
│   ├── evolution.py                # 策略演化机制
│   ├── topology.py                 # 网络拓扑结构
│   └── recorder.py                 # 数据记录与可视化
├── experiments/                    # 实验模块
│   ├── exp1/                       # 实验1：无反馈机制
│   ├── exp2/                       # 实验2：有反馈机制
│   └── exp3/                       # 实验3：网络拓扑效应
├── main.py                         # 主程序入口
└── README.md                       # 项目说明文档
```

## 安装要求 | Requirements

- Python 3.11+
- NumPy
- Matplotlib
- NetworkX

## 快速开始 | Quick Start

### 运行主程序 | Run Main Program

```bash
python main.py
```

程序将显示实验选择菜单：
```
选择实验：1=无反馈，2=有反馈，3=网络拓扑效应
请输入实验编号：
```

### 实验说明 | Experiment Descriptions

#### 实验1：无反馈机制 | Experiment 1: No Feedback
- 研究无自适应攻击行为下的合作演化动态
- 改变增强因子(r)寻找关键合作阈值
- 演示外部攻击概率如何促进防御合作

#### 实验2：有反馈机制 | Experiment 2: With Feedback
- 实现基于局部和全局成功率的自适应攻击概率
- 研究攻击者如何学习和调整策略
- 探索防御合作与攻击策略的协同演化

#### 实验3：网络拓扑效应 | Experiment 3: Network Topology Effects
- 比较不同网络结构下的合作水平
- 研究格子网络、小世界网络、无标度网络和随机网络对集体防御的影响
- 分析网络结构与合作稳定性的关系

## 核心模块说明 | Core Modules

### 智能体 | Agents
- **Defender**: 网络节点，可选择合作('C')或背叛('D')，具有累积收益
- **Attacker**: 战略实体，根据成功率自适应调整攻击概率
- **Agent**: 所有智能体的基类，包含ID、策略和收益追踪

### 博弈模型 | Game Models
- **公共物品博弈(PGG)**: 建模集体防御投资，防御者选择合作（投资安全）或背叛（搭便车）
- **攻防博弈**: 建模个体攻防互动，包含战略收益和成功率参数

### 策略演化 | Strategy Evolution
- **Fermi更新规则**: 基于收益差异的概率性策略模仿
- 温度参数(K)控制策略变化的随机性

### 网络拓扑 | Network Topology
支持多种网络结构：
- 规则格子网络 (Lattice)
- 小世界网络 (Small-World)
- 无标度网络 (Scale-Free)
- 随机网络 (Random)

## 使用示例 | Usage Examples

### 运行特定实验
```python
# 导入实验模块
from experiments.exp1 import exp1_no_feedback
from experiments.exp2 import exp2_with_feedback
from experiments.exp3 import exp3_fig7

# 运行实验
exp1_no_feedback.run_exp1()
exp2_with_feedback.run_exp2()
exp3_fig7.main()
```

### 自定义仿真参数
```python
from core.simulation import CyberSecuritySimulation
from core.topology import NetworkTopology

# 创建自定义网络
topology = NetworkTopology.create_network('small_world', n=100, k=4, p=0.1)

# 运行仿真
sim = CyberSecuritySimulation(topology, r=3.0, K=0.1)
sim.run_simulation(steps=1000)
```

## 参数配置 | Parameter Configuration

### 关键参数
- `r`: 公共物品博弈的增强因子
- `K`: Fermi更新的温度参数
- `attack_prob`: 外部攻击概率
- `network_type`: 网络类型 ('lattice', 'small_world', 'scale_free', 'random')

### 实验参数范围
- 增强因子 r: 1.0 - 5.0
- 温度参数 K: 0.1 - 10.0
- 网络规模: 100 - 1000 节点
- 仿真步数: 1000 - 10000 步

## 输出结果 | Output Results

仿真程序会生成以下输出：
- 📈 合作率随时间变化图
- 📊 攻击成功率统计图
- 🎯 攻击概率演化图
- 📋 策略分布数据


## ⚠️ 重要声明 | Important Notice

**实验结果差异说明**: 本项目的仿真实验结果与原始论文中的结果存在较大差异。尽管我们已对代码逻辑进行了详细检查，但在实现层面未发现明显的逻辑错误。可能的原因包括：

1. **参数理解差异**: 对论文中某些参数的具体设置和取值范围可能存在理解偏差
2. **初始化条件**: 网络初始化、策略分布或初始条件可能与论文设定不完全一致
3. **随机性影响**: 仿真过程中的随机性可能导致结果偏离，需要更多次独立运行
4. **算法实现细节**: 某些算法的具体实现细节可能与论文原始描述存在细微差别

**当前状态**: 代码逻辑结构完整，各模块功能正常，能够正常运行并产生仿真结果。建议进一步研究论文原文，对比关键参数设置，或联系论文作者获取更多实现细节。

**实验建议**:
- 增加仿真重复次数以提高结果稳定性
- 系统性调整关键参数进行敏感性分析
- 对比不同初始化条件对结果的影响

## 贡献指南 | Contributing

欢迎提交问题和改进建议！
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request


