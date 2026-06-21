# DREAMPlaceFPGA 学习笔记

本仓库记录了对 DREAMPlaceFPGA 开源 GPU 加速 FPGA 布局器的逐日学习过程。

## 学习进度

| 天数 | 内容 | 核心文件 |
|------|------|---------|
| Day 1 | 入口与配置系统 | `Placer.py`, `Params.py`, `paramsFPGA.json` |
| Day 2 | 设计数据库 | `PlaceDB.py`, `ops/place_io/` (C++ pybind11) |
| Day 3 | 数据封装与算子体系 | `BasicPlace.py`, `PlaceObj.py`, `ops/move_boundary/` |
| Day 4 | 非线性求解器 | `NonLinearPlace.py`, `NesterovAcceleratedGradientOptimizer.py` |
| Day 5 | 时序驱动布局 | `Timer.py`, `timing_graph.py`, `timing.py` |

## 项目简介

DREAMPlaceFPGA 是 UT Austin 开发的开源 GPU 加速 FPGA 布局器，基于 PyTorch 深度学习框架，利用 GPU 并行计算加速大规模异构 FPGA 的全局布局和合法化。

- 原始仓库: [rachelselinar/DREAMPlaceFPGA](https://github.com/rachelselinar/DREAMPlaceFPGA)
- 论文: ASP-DAC 2022 / ISPD 2023 / FCCM 2024
