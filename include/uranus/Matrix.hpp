// The MIT License (MIT)
//
// Copyright (c) 2015 Markus Herb
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/**
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-11
 */
#ifndef _URANUS_MATRIX_HPP_
#define _URANUS_MATRIX_HPP_

#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
namespace uranus
{
	const int Dynamic = Eigen::Dynamic;

	/**
	 * @class lcp::Matrix
	 * @brief Template type for matrices
	 * @param T The numeric scalar type
	 * @param rows The number of rows
	 * @param cols The number of columns
	 */
	template<int rows, int cols>
	using Matrix = Eigen::Matrix<double, rows, cols>;

	/**
	 * @brief Template type for vectors
	 * @param T The numeric scalar type
	 * @param N The vector dimension
	 */
	template<int N>
	using Vector = Eigen::Matrix<double, N, 1>;
	
	/**
	 * @class lcp::SquareMatrix
	 * @brief Template type representing a square matrix
	 * @param T The numeric scalar type
	 * @param N The dimensionality of the Matrix
	 */
	template<int N>
	using SquareMatrix = Eigen::Matrix<double, N, N>;

	/**
	 * @class uranus::setZeroMat
	 * @brief set all element zero for vector or SquareMatrix
	 * @param Type the input template
	 * @param N The dimensionality of vector or SquareMatrix
	 */
	template<typename Type, int N>
	inline bool setZero(Type& rhs)
	{
		if (std::is_same<Type, Vector<N>>::value)
		{
			for (int i = 0; i < N; ++i) rhs(i) = 0;
			return true;
		}
		else if (std::is_same<Type, SquareMatrix<N>>::value)
		{
			for (int i = 0; i < N*N; ++i) rhs(i) = 0;
			return true;
		}
		else
			return false;
	}

	/*Eigen::MatrixXcd evals = eigen_solver.eigenvectors();        // 特征值
	Eigen::MatrixXf::Index evalsMax;
	evalsReal.rowwise().sum().maxCoeff(&evalsMax);//得到最大特征值的位置

	Eigen::MatrixXf::Index evalsMin;
	evalsReal.rowwise().sum().minCoeff(&evalsMin);//得到最大特征值的位置

	uranus::Vector<Dim> q; //得到对应特征向量

	q << evals.real()(0, evalsMax), 
		 evals.real()(1, evalsMax), 
		 evals.real()(2, evalsMax), 
		 evals.real()(3, evalsMax);*/

	/*Eigen::MatrixXcd evals = eigen_solver.eigenvalues(); // 伪特征值
	cout << "\n eigen = " << evals << "\n";
	Eigen::MatrixXd evalsReal = evals.real();//获取特征值实数部分
	cout << "\n eigen = " << evalsReal << "\n";*/
}

#endif // _URANUS_MATRIX_HPP_
