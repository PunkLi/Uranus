# Uranus
[![Build Status](https://travis-ci.org/hackath/Uranus.svg?branch=master)](https://travis-ci.org/hackath/Uranus)
[![License: MIT](./docs/license_badge.svg)](./LICENSE)

A tiny library for solving math and optimization problem.

## Why Uranus?

为了致敬1801年9月高斯用数学方法预测并发现了谷神星（Ceres），谷歌将自己开发的数学库命名为[ceres-solver](https://github.com/ceres-solver/ceres-solver)。

> Ceres Solver is an open source C++ library for modeling and solving large, complicated optimization problems. It is a feature rich, mature and performant library which has been used in production at Google since 2010. Ceres Solver can solve two kinds of problems.
>1. Non-linear Least Squares problems with bounds constraints.
>2. General unconstrained optimization problems.

我们希望实现一个小型数学库作为`AI5281L`软件工程这门课程的大作业，同时希望该库能够用于解决本学期其他学院选修课程的相关问题：

- `AI5276L`最优化及其应用
- `AI5278L`模式识别

我们的项目以“海王星（Uranus）”为名，因为`Uranus`也是一颗“被发现”的星体。

> 海王星是太阳系中距离太阳最远的行星，在1846年9月23日被发现，是唯一利用数学预测而非有计划的观测发现的行星。天文学家利用天王星轨道的摄动推测出海王星的存在与可能的位置。

## Dependencies
- 代码构建：https://cmake.org/
- 线性代数：http://eigen.tuxfamily.org/index.php?title=Main_Page
- 文档生成：http://www.doxygen.org/
- 日志系统：https://github.com/google/glog
- 数据存储：https://developers.google.com/protocol-buffers/
- 单元测试：https://github.com/google/googletest.git
- 集成测试：https://travis-ci.org/

## License

The MIT License (MIT)

Copyright (c) 2018 Artificial Intelligence College, Xidian university

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.