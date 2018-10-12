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
#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <cmath>
#include <Eigen/Dense>

namespace uranus 
{
	const int Dynamic = Eigen::Dynamic;

	/**
	 * @class uranus::Matrix
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
	 * @class uranus::SquareMatrix
	 * @brief Template type representing a square matrix
	 * @param T The numeric scalar type
	 * @param N The dimensionality of the Matrix
	 */
	template<int N>
	using SquareMatrix = Eigen::Matrix<double, N, N>;
	
}
#endif
