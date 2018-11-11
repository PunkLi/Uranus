// The MIT License (MIT)

// Copyright (c) 2018 li chunpeng, Xidian university

// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#ifndef _URANUS_UN_CONSTRAINED_HPP_
#define _URANUS_UN_CONSTRAINED_HPP_

// 常见无约束优化算法
// 直接方法
// 间接方法

#include <iostream>
#include <Eigen/Dense>
using namespace std;

void Linear_search()
{
	double x_k;
	double t;
	double P_k;
	double x_kk = x_k + t * P_k;   // 迭代

	Eigen::Matrix<double, 3, 3> Jacobian;

	// if (Jacobian.transpose()*P_k == 0);  // 要满足的条件

	// minf(x_k + t*P_k)
}

// 函数
double fucntion(Eigen::Matrix<double, 2, 1>& x)
{
	return x(0)*x(0) + 4 * x(1)*x(1);
}

// 传入一个点，计算该点的梯度
Eigen::Matrix<double, 2, 1> gradient(Eigen::Matrix<double, 2, 1>& x)
{
	Eigen::Matrix<double, 2, 2> Jacobian;
	Jacobian << 2, 0, 0, 8;
	return Jacobian * x;
}

double varepsilon = 0.001;

template<int Dim>
void steepest_descent()
{

	// set 初始点
	Eigen::Matrix<double, 2, 1> var_x;
	var_x << 1, 1;

	// step1 求函数
	double y = fucntion(var_x);、

	do {
		// step2 求梯度
		Eigen::Matrix<double, 2, 1> P_k = - gradient(var_x); // p_k = -g , 负梯度方向

		// step3 求步长 ls(X0, -go)
		(x_0 + t_k * P_k) ^ T P_k = 0;
		solver(t_k);                      // 解这个方程

		var_x = var_x + t_k * P_k;        // update x

	} while (g.norm() > varepsilon);      // 梯度的模 > 0.001

	// 打印结果
	double result = fucntion(var_x);// f = x_1 ^2 + 4* x_2 ^2
}

#endif