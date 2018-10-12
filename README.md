# Uranus
为了致敬1801年9月高斯用数学方法预测并发现了谷神星（Ceres）。谷歌开发了[ceres-solver](https://github.com/ceres-solver/ceres-solver)。

我们想像`ceres-solver`那样，实现一个简单的数学库，目前只用于：
- `AI5281L`软件工程大作业
- `AI5276L`最优化及其应用大作业
- `AI5278L`模式识别大作业

我们的项目以“海王星（Uranus）”为名，因为`Uranus`也是一颗“被发现”的星体。

## Dependencies
- 代码构建：[CMake](https://cmake.org/)
- 线性代数：[Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- 软件测试：[googetest](https://github.com/google/googletest.git)
- 文档生成：[doxygen](http://www.doxygen.org/)
- 数据存储：[Protocol](https://developers.google.com/protocol-buffers/)

## To-do List
该数学库的实现标准是能够正常使用并解决模式识别(AI5278L)的课程的大作业。

## License

The MIT License (MIT)

Copyright (c) 2018 Artificial Intelligence College, Xidian university

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.