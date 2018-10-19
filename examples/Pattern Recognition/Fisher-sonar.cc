/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"

int main(int argc, char *argv[])
{
	using namespace std;

	std::string path = "../data/UCI-Iris/sonar.all-data";
	constexpr int feature_rows = 60;
	std::vector<int> class_ = { 97,111 };
	uranus::Data_Wrapper<feature_rows> wrapper(path, class_);
	uranus::Tensor<feature_rows> data(wrapper, class_);

	auto& x_set1 = data.tensor[0];
	auto& x_set2 = data.tensor[1];

	uranus::Vector<feature_rows> mean_1;
	for (int i = 0; i < feature_rows; ++i)mean_1(i) = 0;

	uranus::Vector<feature_rows> mean_2;
	for (int i = 0; i < feature_rows; ++i)mean_2(i) = 0;
	
	uranus::SquareMatrix<feature_rows> Si_1;
	for (int i = 0; i < feature_rows*feature_rows; ++i)Si_1(i) = 0;

	uranus::SquareMatrix<feature_rows> Si_2;
	for (int i = 0; i < feature_rows*feature_rows; ++i)Si_2(i) = 0;
	
	// step1 均值向量
	for (int i = 0; i < class_[0]; ++i) mean_1 += x_set1[i];
	mean_1 = mean_1 / class_[0];
	for (int i = 0; i < class_[1]; ++i) mean_2 += x_set2[i];
	mean_2 = mean_2 / class_[1];
	
	// step2 类内离散度矩阵
	for (int i = 0; i < class_[0]; ++i)
		Si_1 += (x_set1[i] - mean_1)*(x_set1[i] - mean_1).transpose();
	for (int i = 0; i < class_[1]; ++i)
		Si_2 += (x_set2[i] - mean_2)*(x_set2[i] - mean_2).transpose();

	//cout << "Si_1=\n" << Si_1 << endl << endl;
	//cout << "Si_2=\n" << Si_2 << endl << endl;
	
	// step3
	// 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
	uranus::SquareMatrix<feature_rows> Sw = Si_1 + Si_2;
	cout << "Sw=\n" << Sw << endl << endl;
	
	// step4
	// 样本类间离散度矩阵SB
	uranus::SquareMatrix<feature_rows> Sb = (mean_1 - mean_2) * (mean_1 - mean_2).transpose();

	// step5
	// Fisher准则函数 -- 最佳投影方向
	// uranus::SquareMatrix<feature_rows> Jw = Sb*Sw.inverse();
	// w* = \argmax J(w)

	uranus::Vector<feature_rows> argW = Sw.inverse()*(mean_1 - mean_2);
	cout << "argW=\n" << argW << endl << endl;

	// step6求阈值 W0 
	uranus::Vector<1> W0 = argW.transpose()*mean_1 / 2 + argW.transpose()*mean_2 / 2;
	cout << "Wo=\n" << W0 << endl << endl;

	// step7线性变换
	std::vector<uranus::Vector<1>> D1(class_[0]);
	std::vector<uranus::Vector<1>> D2(class_[1]);

	for (int i = 0; i < class_[0]; ++i)
	{
		D1[i] = argW.transpose()*x_set1[i];
		cout << D1[i] << endl;
	}

	for (int i = 0; i < class_[1]; ++i)
	{
		D2[i] = argW.transpose()*x_set2[i];
		cout << D2[i] << endl;
	}
	return EXIT_SUCCESS;
}