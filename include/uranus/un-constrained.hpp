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

template<int Dim> struct problem
{
	using namespace uranus;
	SquareMatrix<Dim> matirx_Jacobian_;  // 一阶求导矩阵
	
	uranus::Vector<Dim> gradient(const uranus::Vector<Dim>& var_x)
	{
		return matirx_Jacobian_ * var_x;
	}

	uranus::Vector<Dim> constrained(uranus::Vector<Dim> var_x)
	{
		// 算式；
	}
};

/**
 * @brief 使用秩为2的模拟牛顿法, BFGS
 * @param 初始点 var_x,是一个向量，可包含多维
 * @param 终止条件 delta, 默认 0.001
 * @param 是否打印中间结果
 * @return 求解得到的最优点 
 */
template<int Dim>
uranus::Vector<Dim> BFGS(uranus::Vector<Dim> var_x, 
						 const double delta = 0.001, 
						 bool visual)
{
	uranus::Vector<Dim> s_k;
	uranus::Vector<Dim> y_k;
	uranus::SquareMatrix<Dim> H_k;
	H_k = MatrixXf::Identity(Dim,Dim);             // init 单位矩阵
	
	while(s_k < delta)  
	{
		s_k = -H_k * Jacobian(var_x);              // s_k = var_x2 - var_x1
		y_k = Jacobian(var_x);
		var_x = var_x - H_k * Jacobian(var_x);     
		y_k = Jacobian(var_x) - y_k;               // y_k = \delta f_{k} - \delta f_{k-1}
										      	   // update H_k
    	H_k = H_k
		- H_k * y_k * s_k.transpose()
		+ s_k * y_k.transpose() * H_k / s_k.transpose() * y_k 
		+ s_k * s_k.transpose() / (s_k.transpose() * y_k) 
		* (1 + y_k.transpose() * H_k * y_k / s_k.transpose() * y_k);
	}
	return var_x; // result
}
/**
 * @brief 使用BFGS的外点法
 * @param 初始点 var_x,是一个向量，可包含多维
 * @param 终止条件 delta, 默认 0.001
 * @param 是否打印中间结果
 * @return 求解得到的最优点 
 */
template<int Dim>
void External_point_method(uranus::Vector<Dim> var_x, 
						   const double delta = 0.001, 
						   const double varepsilon = 0.001, 
						   const double C = 10;)
{
	double M_k = 1;      // init M_k > 0	
	do{
		M_k = C * M_k;   // update M_k
		BFGS(var_x);
	}
	while(M_k* h > varepsilon);
}

void init_problem()
{
	problem<Dim> f;	  // f = x_1^2 + 4x_2^2

	f.matirx_Jacobian_ << 2, 0, 0, 8; 

	uranus::Vector<2> x0; 
	x0 << 1, 1;                   // 初始点

	uranus::Vector<2> result = BFGS<2>(x0); // 用BFGS求解
}

#endif