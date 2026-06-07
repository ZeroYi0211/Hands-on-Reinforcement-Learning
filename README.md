# 《动手学强化学习》课程学习笔记

这个仓库用于记录我学习张伟楠老师《动手学强化学习》课程和教材时的代码复现、实验脚本和课程资料整理。

学习目标不是简单保存官方 notebook，而是随着课程推进，把关键算法和案例重新写成可以独立运行的 `.py` 脚本，逐步形成一套自己的强化学习实验代码。

## 学习方式

我计划按章节推进，每一章大致遵循下面的节奏：

1. 阅读教材和课程网站内容，先理解问题背景和核心公式。
2. 对照官方 notebook 跑通原始实验。
3. 在 `scripts/` 中用纯 `.py` 文件重新实现核心代码。
4. 修改少量超参数或实验设置，观察结果变化。
5. 将重复出现的工具函数再整理到公共模块中。

当前阶段以“理解算法”为主，所以代码会尽量保持直观，不会过早抽象成复杂框架。

## 目录结构

```text
.
├── notebooks/          # 官方 notebook 和辅助工具代码
├── scripts/            # 自己改写和复现的 Python 脚本
├── slides/             # 课程 slides
├── requirements.txt    # 官方依赖版本
└── README.md
```

其中 `scripts/` 是后续学习的主要工作区。每个脚本尽量做到可以独立运行，例如：

```powershell
D:\miniforge3\envs\hrl\python.exe scripts\ch02_bandit.py
```

## 环境配置

本项目使用的 Conda 虚拟环境名为 `hrl`。依赖版本主要参考官方 GitHub 配置，尤其保留旧版 Gym：

```text
gym==0.18.3
```

旧版 Gym 的接口和课程 notebook 保持一致，例如：

```python
state = env.reset()
next_state, reward, done, info = env.step(action)
```

这能减少学习早期的 API 迁移干扰，把注意力放在强化学习算法本身。

安装依赖：

```powershell
conda activate hrl
pip install -r requirements.txt
```

如果 `conda run` 在 Windows 中文输出下出现编码问题，可以直接调用环境里的 Python：

```powershell
D:\miniforge3\envs\hrl\python.exe scripts\ch02_bandit.py
```

## 已开始复现的内容

| 章节 | 主题 | 脚本 |
| --- | --- | --- |
| 第 2 章 | 多臂老虎机问题 | `scripts/ch02_bandit.py` |

后续计划继续按章节补充：

- 第 3 章：马尔可夫决策过程
- 第 4 章：动态规划算法
- 第 5 章：时序差分算法
- 第 6 章：Dyna-Q 算法
- 第 7 章：DQN 算法

## 代码风格约定

每个章节脚本尽量采用统一结构：

```python
"""
Chapter N: Topic

Goal:
- ...
"""

def run_experiment():
    ...


def main():
    ...


if __name__ == "__main__":
    main()
```

原则：

- 一个脚本优先对应一个章节或一个核心算法。
- 优先手写核心更新逻辑，而不是直接复用黑盒工具函数。
- 保持脚本可独立运行，不依赖 notebook 的隐藏状态。
- 等重复代码足够明显后，再抽取到 `scripts/rl_utils.py`。

## 说明

本仓库仅用于个人学习和复现。课程资料、slides 和教材内容的版权归原作者或发布方所有。
