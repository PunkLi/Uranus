/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Matrix.hpp"
#include "uranus/Tensor.hpp"

int main(int argc, char *argv[])
{
	using namespace std;

	std::string path = "../data/UCI-Iris/iris.data";
	uranus::Data_Wrapper wrapper(path); // 读文件
	uranus::Tensor data(wrapper);

	auto& x_set1 = data.tensor[0];
	auto& x_set2 = data.tensor[1];
	auto& x_set3 = data.tensor[2];

	constexpr int feature_rows = 4;;

	uranus::Vector<feature_rows> mean_1;
	for (int i = 0; i < feature_rows; ++i)mean_1(i) = 0;

	uranus::Vector<feature_rows> mean_2;
	for (int i = 0; i < feature_rows; ++i)mean_2(i) = 0;

	uranus::Vector<feature_rows> mean_3;
	for (int i = 0; i < feature_rows; ++i)mean_3(i) = 0;

	uranus::SquareMatrix<feature_rows> Si_1;
	for (int i = 0; i < feature_rows*feature_rows; ++i)Si_1(i) = 0;

	uranus::SquareMatrix<feature_rows> Si_2;
	for (int i = 0; i < feature_rows*feature_rows; ++i)Si_2(i) = 0;

	uranus::SquareMatrix<feature_rows> Si_3;
	for (int i = 0; i < feature_rows*feature_rows; ++i)Si_3(i) = 0;

	// step1 均值向量
	for (int i = 0; i < uranus::class_1; ++i) mean_1 += x_set1[i];
	mean_1 = mean_1 / uranus::class_1;
	for (int i = 0; i < uranus::class_2; ++i) mean_2 += x_set2[i];
	mean_2 = mean_2 / uranus::class_2;
	for (int i = 0; i < uranus::class_3; ++i) mean_3 += x_set3[i];
	mean_3 = mean_3 / uranus::class_3;

	// step2 类内离散度矩阵
	for (int i = 0; i < uranus::class_1; ++i)
		Si_1 += (x_set1[i] - mean_1)*(x_set1[i] - mean_1).transpose();
	for (int i = 0; i < uranus::class_2; ++i)
		Si_2 += (x_set2[i] - mean_2)*(x_set2[i] - mean_2).transpose();
	for (int i = 0; i < uranus::class_3; ++i)
		Si_3 += (x_set3[i] - mean_3)*(x_set3[i] - mean_3).transpose();

	cout << "Si_1=\n" << Si_1 << endl << endl;
	cout << "Si_2=\n" << Si_2 << endl << endl;
	cout << "Si_3=\n" << Si_3 << endl << endl;
	// step3
	// 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
	uranus::SquareMatrix<feature_rows> Sw = Si_1 + Si_2 + Si_3;
	cout << "Sw=\n" << Sw << endl << endl;

	// step4
	// 样本类间离散度矩阵SB
	//uranus::SquareMatrix<feature_rows> Sb = (mean_1 - mean_2) * (mean_1 - mean_2).transpose();
	uranus::SquareMatrix<feature_rows> Sb_12 = (mean_1 - mean_2) * (mean_1 - mean_2).transpose();
	uranus::SquareMatrix<feature_rows> Sb_13 = (mean_1 - mean_3) * (mean_1 - mean_3).transpose();
	uranus::SquareMatrix<feature_rows> Sb_23 = (mean_2 - mean_3) * (mean_2 - mean_3).transpose();

	// step5
	// Fisher准则函数 -- 最佳投影方向
	// uranus::SquareMatrix<feature_rows> Jw = Sb*Sw.inverse();
	// w* = \argmax J(w)

	// uranus::Vector<feature_rows> argW = Sw.inverse()*(mean_1 - mean_2);
	uranus::Vector<feature_rows> argW_12 = Sw.inverse()*(mean_1 - mean_2);
	uranus::Vector<feature_rows> argW_13 = Sw.inverse()*(mean_1 - mean_3);
	uranus::Vector<feature_rows> argW_23 = Sw.inverse()*(mean_2 - mean_3);
	cout << "argW_12=\n" << argW_12 << endl << endl;
	cout << "argW_13=\n" << argW_13 << endl << endl;
	cout << "argW_23=\n" << argW_23 << endl << endl;

	// step6求阈值 W0 
	// uranus::Vector<1> W0 = argW.transpose()*mean_1 / 2 + argW.transpose()*mean_2 / 2;
	uranus::Vector<1> W0_12 = argW_12.transpose()*mean_1 / 2 + argW_12.transpose()*mean_2 / 2;
	uranus::Vector<1> W0_13 = argW_13.transpose()*mean_1 / 2 + argW_13.transpose()*mean_3 / 2;
	uranus::Vector<1> W0_23 = argW_23.transpose()*mean_2 / 2 + argW_23.transpose()*mean_3 / 2;

	// step7线性变换
	std::vector<uranus::Vector<1>> D1(uranus::class_1);
	std::vector<uranus::Vector<1>> D2(uranus::class_2);
	std::vector<uranus::Vector<1>> D3(uranus::class_3);

	// 1-2分类
	cout << "1-2分类" << endl;
	cout << "D1=" << endl;
	for (int i = 0; i < uranus::class_1; ++i)
	{
		D1[i] = argW_12.transpose()*x_set1[i];
		cout << D1[i] << endl;
	}
	cout << "Wo_12=\n" << W0_12 << endl << endl;
	cout << "D2=" << endl;
	for (int i = 0; i < uranus::class_2; ++i)
	{
		D2[i] = argW_12.transpose()*x_set2[i];
		cout << D2[i] << endl;
	}
	// 1-3分类
	cout << "1-3分类" << endl;
	cout << "D1=" << endl;
	for (int i = 0; i < uranus::class_1; ++i)
	{
		D1[i] = argW_13.transpose()*x_set1[i];
		cout << D1[i] << endl;
	}
	cout << "Wo_13=\n" << W0_13 << endl << endl;
	cout << "D3=" << endl;
	for (int i = 0; i < uranus::class_3; ++i)
	{
		D3[i] = argW_13.transpose()*x_set3[i];
		cout << D3[i] << endl;
	}
	// 2-3分类
	cout << "2-3分类" << endl;
	cout << "D2=" << endl;
	for (int i = 0; i < uranus::class_2; ++i)
	{
		D2[i] = argW_23.transpose()*x_set2[i];
		cout << D2[i] << endl;
	}
	cout << "Wo_23=\n" << W0_23 << endl << endl;
	cout << "D3=" << endl;
	for (int i = 0; i < uranus::class_3; ++i)
	{
		D3[i] = argW_23.transpose()*x_set3[i];
		cout << D3[i] << endl;
	}
	return EXIT_SUCCESS;
}