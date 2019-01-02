// The MIT License (MIT)

// Copyright (c) 2018 li chunpeng, Xidian university

// Permission is hereby granted, free of charge, to any person obtaining sum copy of
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
#include "Matrix.hpp"

using namespace std;

template<int Dim> struct problem
{
	uranus::SquareMatrix<Dim> matirx_Jacobian_;  // 一阶求导矩阵
	
	uranus::Vector<Dim>& gradient(uranus::Vector<Dim>& var_x)
	{
		var_x = matirx_Jacobian_ * var_x;
		return var_x;
	}
};

template<int Dim> double myval(uranus::Vector<Dim> s_k)
{
	double sum = 0;
	for(int i=0;i<Dim;++i)
	{
		sum = sum + pow(s_k(i),2);
	}
	return sqrt(sum);
}
/**
 * @brief 使用秩为2的模拟牛顿法, BFGS
 * @param 求解的问题f
 * @param 初始点 var_x,是一个向量，可包含多维
 * @param 终止条件 delta, 默认 0.001
 * @param 是否打印中间结果
 * @return 求解得到的最优点 
 */
template<int Dim>
uranus::Vector<Dim> BFGS(problem<Dim> f,
						 uranus::Vector<Dim> var_x, 
						 const double delta = 0.001, 
						 bool visual = false)
{
	uranus::Vector<Dim> s_k;
	uranus::Vector<Dim> y_k;
	uranus::SquareMatrix<Dim> H_k;
	uranus::SquareMatrix<Dim> E_k1;
	uranus::SquareMatrix<Dim> E_k2;

	// init 单位矩阵
	for (int i = 0;i < Dim; ++i)
		for (int j = 0;j < Dim; ++j)
			if ( i == j) H_k(i,j) = 1;
			else H_k(i,j) = 0;
	
	while(myval<Dim>(s_k) < delta)  
	{
		s_k = -H_k * f.gradient(var_x);           // s_k = var_x2 - var_x1
		y_k = f.gradient(var_x);
		var_x = var_x - H_k * f.gradient(var_x);     
		y_k = f.gradient(var_x) - y_k;            // y_k = \delta f_{k} - \delta f_{k-1}
	    
		
		E_k1  = (s_k * s_k.transpose() - H_k * y_k * s_k.transpose() - s_k * y_k.transpose() * H_k)
		      / (s_k.transpose() * y_k);
		E_k2 = s_k * s_k.transpose();

		double k21 = y_k.transpose() * H_k * y_k;
		double k22 = pow((s_k.transpose() * y_k),2);

		H_k = H_k + E_k1 + E_k2 * k21/k22;    // update H_k

		if(visual)
		{
			cout << "s_k:\n" << s_k << "\n" 
				 << "y_k:\n" << y_k << "\n"
				 << "H_k:\n" << H_k << "\n";
 		}
<<<<<<< HEAD
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
						   const double C = 10
						   bool visual = false)
{
	double M_k = 0.1;                          // init M_k > 0	
	do{
		M_k = C * M_k;                         // update M_k
		f.Jacobian_Matrix(1,1) = 2 + M_k;      // update J mat

		var_x = BFGS(f, var_x, delta, visual); // min f
	}
	while(M_k * pow(var_x(1)-1), 2) > delta);

}
=======
		 cin.get();
	}
	return var_x; // result
}
>>>>>>> 37ff8f24c79fe0c3ddf40c99b90e94e3d8d106be

#endif