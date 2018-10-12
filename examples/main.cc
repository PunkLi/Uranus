/**
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-11
 */

#include <iostream>
#include "uranus/Matrix.hpp"
#include "uranus/Function.hpp"

int main(int argc, char *argv[])
{
	uranus::Vector<2> x0; 
	x0 << 1, 1;                   // 初始点

	double coeff_num[] = { 1,4 }; // 系数项
	double exp_num[] = { 2,2 };   // 指数项
	double const_num = 0;         // 常数项
	
	// 二维 x1，x2
	uranus::Function<2> func(coeff_num, exp_num, const_num); 
	
	std::cout << "一阶求导：\n" << func.Jacobian_Matrix() << std::endl;

	std::cout << "二阶求导: \n" << func.Hessian_Matrix() << std::endl;

	uranus::Vector<2> x1 = func.Solve_by_Newton(x0);

	std::cout << "x1= \n" << x1; 

	std::cin.get();

	return EXIT_SUCCESS;
}