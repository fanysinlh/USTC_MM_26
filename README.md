# 中国科学技术大学《数学建模》课程作业 - 2026年春

## 0. 课程简介

本仓库是中国科学技术大学本科课程《数学建模》（MATH001139）的课程作业仓库。

- 课程主页：[Mathematical Modeling 2026](http://staff.ustc.edu.cn/~renjiec/mm2026/)
- 任课教师：[陈仁杰](http://staff.ustc.edu.cn/~renjiec/)，[刘雨萌](https://lym29.github.io/)
- 助教：
  - 杨萱泽
  - 张雨阳
  - 邓博文

课程作业为两周一次，当周周三课上布置，次周周日早上截止提交。提交方式：[bb系统](https://www.bb.ustc.edu.cn/)

## 1. 课程作业

本课程将通过 4 次小作业和 1 次大作业，来介绍常见的数学建模问题和方法。通过这些作业，大家会学习到：
- 图和网络模型
- 矩阵分解与图像处理
- 数据拟合模型
- 微分方程模型

作业文档：
- [作业1：图与网络模型](./hw_1/README.md)
- [作业2：矩阵模型](./hw_2/README.md)
- [作业3：数据拟合模型](./hw_3/README.md)
- [作业4：微分方程模型](./hw_4/README.md)

## 2. 相关软件

### 2.1 Git

Git 是最常用的版本控制工具，可以记录代码/文档的修改历史，方便回滚、协作与提交作业。课程作业建议使用 Git 管理本仓库的代码与报告。

相关链接：[官网](https://git-scm.com/)，[下载链接](https://git-scm.com/download/win)，[菜鸟 Git 教程](https://www.runoob.com/git/git-tutorial.html) ，[廖雪峰 Git 教程](https://www.liaoxuefeng.com/wiki/896043488029600) 

### 2.2 C++ 程序编译

部分作业中提供了 C++ 实现框架（位于各题的 `code_template_cpp/` 目录），但 C++ 程序需要先编译才能运行。本节介绍如何使用 CMake 对本作业中的 C++ 程序进行构建和编译，详见 [C++ 程序编译说明](./BUILD_CPP.md)。

相关链接：[官网](https://cmake.org/), [官方教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)，[官方文档](https://cmake.org/documentation/)，[适合初学者的 CMakeTutorial](https://github.com/BrightXiaoHan/CMakeTutorial)

### 2.3 Miniforge

Miniforge 是一个轻量级的 Conda 发行版（默认使用社区维护的 `conda-forge` 软件源），可用于在 Windows/macOS/Linux 上创建彼此隔离的 Python 环境，便于统一课程作业的依赖版本，避免“系统 Python/包版本冲突”。安装后你会得到 `conda` 与更快的 `mamba`（若未自带，可后续安装）。

相关链接：[官网](https://conda-forge.org/miniforge/)，[官方 GitHub 仓库](https://github.com/conda-forge/miniforge)

## 3. 辅助资料

- `MATLAB` 相关资源：[MATLAB 官方文档](https://ww2.mathworks.cn/help/matlab/index.html?s_tid=hc_panel), [MATLAB 教程](https://www.cainiaojc.com/matlab/matlab-tutorial.html)
- `CPP` 相关资源：[CPP Reference](https://en.cppreference.com/w/), [CPP 教程](https://www.runoob.com/cplusplus/cpp-tutorial.html)
- `Python` 相关资源：[Python 入门](https://github.com/walter201230/Python), [OpenCV-Python Docs](https://codec.wang/docs/opencv)
- `Mathematica` 相关资源：[Mathematica 官网](https://www.wolfram.com/mathematica/), [WolframAlpha 在线计算平台](https://www.wolframalpha.com/)
- `国产科学计算软件 北太天元` 相关资源：[下载链接](https://www.baltamatica.com/download.html)
- `数学建模相关资源`：[全国大学生数学建模竞赛](https://www.mcm.edu.cn/) （注：含有往届赛事的赛题和优秀解答，可供参考）, [美国大学生数学建模竞赛](https://www.comap.com/contests/mcm-icm)