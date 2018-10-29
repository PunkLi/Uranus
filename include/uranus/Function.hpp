// The MIT License (MIT)
//
// Copyright (c) 2018 li chunpeng, Xidian Universty
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

#ifndef _URANUS_FUNCTION_HPP_
#define _URANUS_FUNCTION_HPP_

#include "Matrix.hpp"

namespace uranus {
    /**
	 * @class uranus::Function
	 * @brief Template type representing a Polynomial function
	 * @param Dim The dimensionality of the Variable
	 */
	template<int Dim> class Function
	{
	public:
		Function(double* coeff, double* exp, double const_num);
		Vector<Dim> Solve_by_Newton(Vector<Dim> x0) { return x0 - Hessian_inv() * Jacobian(x0);}
		SquareMatrix<Dim> Jacobian_Matrix() { return matirx_Jacobian_; }
		SquareMatrix<Dim> Hessian_Matrix() { return matirx_Hessian_; }

	protected:
		Vector<Dim> Jacobian(uranus::Vector<Dim> x0) { return matirx_Jacobian_ * x0; }
		SquareMatrix<Dim> Hessian_inv() { return matirx_Hessian_.inverse(); }

	private:
		double* coeff_num_;
		double* exp_num_;
		SquareMatrix<Dim> matirx_Jacobian_;  // 一阶求导矩阵
		SquareMatrix<Dim> matirx_Hessian_;   // 二阶求导矩阵
	};

	template<int Dim>
	Function<Dim>::Function(double* coeff, 
							double* exp, 
							double const_num) 
	:coeff_num_(coeff), exp_num_(exp)
	{
		for (int idx = 0; idx < Dim; ++idx)
		{
			for (int jdx = 0; jdx < Dim; ++jdx)
			{
				if (idx == jdx)
				{
					matirx_Jacobian_(jdx, idx) = coeff_num_[idx] * exp_num_[idx];
					exp_num_[idx]--;
				}
				else
					matirx_Jacobian_(jdx, idx) = 0;
			}
		}

		for (int idx = 0; idx < Dim; ++idx)
		{
			for (int jdx = 0; jdx < Dim; ++jdx)
			{
				if (exp_num_[idx])
					matirx_Hessian_(jdx, idx) = matirx_Jacobian_(jdx, idx) * exp_num_[idx];
				else
					matirx_Hessian_(jdx, idx) = matirx_Jacobian_(jdx, idx);
			}
			if (exp_num_[idx]) exp_num_[idx]--;
		}
	}

}

#endif // _URANUS_FUNCTION_HPP_