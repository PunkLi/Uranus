/**
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-04
 */
#include <gtest/gtest.h>
#include "uranus/Function.hpp"

TEST(Function, Newton) {
    
    uranus::Vector<2> x0; 
	x0 << 1, 1;                   // 初始点

	double coeff_num[] = { 1,4 }; // 系数项
	double exp_num[] = { 2,2 };   // 指数项
	double const_num = 0;         // 常数项
	
	uranus::Function<2> func(coeff_num, exp_num, const_num); 
	uranus::Vector<2> x1 = func.Solve_by_Newton(x0);

    uranus::Vector<2> except_x;   // 期望结果
	x0 << 0, 0;

	EXPECT_EQ(except_x, x1);
}