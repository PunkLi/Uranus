/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Matrix.hpp"
#include "uranus/Function.hpp"

int main(int argc, char *argv[])
{
	using namespace std;
	const int batch_size = 10;

	int size = 0;
	int size2 = 0;
	int size3 = 0;
	constexpr int Dim = 4;
	std::vector<uranus::Vector<Dim>> x_set1(50);  // 50组，4维数，Iris-setosa
	std::vector<uranus::Vector<Dim>> x_set2(50);  // 50组，4维数，Iris-versicolor
	std::vector<uranus::Vector<Dim>> x_set3(50);  // 50组，4维数，Iris-virginica

	x_set1[size++] << 5.1,3.5,1.4,0.2; // Iris-setosa
	x_set1[size++] << 4.9,3.0,1.4,0.2; // Iris-setosa
	x_set1[size++] << 4.7,3.2,1.3,0.2; // Iris-setosa
	x_set1[size++] << 4.6,3.1,1.5,0.2; // Iris-setosa
	x_set1[size++] << 5.0,3.6,1.4,0.2; // Iris-setosa
	x_set1[size++] << 5.4,3.9,1.7,0.4; // Iris-setosa
	x_set1[size++] << 4.6,3.4,1.4,0.3; // Iris-setosa
	x_set1[size++] << 5.0,3.4,1.5,0.2; // Iris-setosa
	x_set1[size++] << 4.4,2.9,1.4,0.2; // Iris-setosa
	x_set1[size++] << 4.9,3.1,1.5,0.1; // Iris-setosa
	x_set1[size++] << 5.4,3.7,1.5,0.2; // Iris-setosa
	x_set1[size++] << 4.8,3.4,1.6,0.2; // Iris-setosa
	x_set1[size++] << 4.8,3.0,1.4,0.1; // Iris-setosa
	x_set1[size++] << 4.3,3.0,1.1,0.1; // Iris-setosa
	x_set1[size++] << 5.8,4.0,1.2,0.2; // Iris-setosa
	x_set1[size++] << 5.7,4.4,1.5,0.4; // Iris-setosa
	x_set1[size++] << 5.4,3.9,1.3,0.4; // Iris-setosa
	x_set1[size++] << 5.1,3.5,1.4,0.3; // Iris-setosa
	x_set1[size++] << 5.7,3.8,1.7,0.3; // Iris-setosa
	x_set1[size++] << 5.1,3.8,1.5,0.3; // Iris-setosa
	x_set1[size++] << 5.4,3.4,1.7,0.2; // Iris-setosa
	x_set1[size++] << 5.1,3.7,1.5,0.4; // Iris-setosa
	x_set1[size++] << 4.6,3.6,1.0,0.2; // Iris-setosa
	x_set1[size++] << 5.1,3.3,1.7,0.5; // Iris-setosa
	x_set1[size++] << 4.8,3.4,1.9,0.2; // Iris-setosa
	x_set1[size++] << 5.0,3.0,1.6,0.2; // Iris-setosa
	x_set1[size++] << 5.0,3.4,1.6,0.4; // Iris-setosa
	x_set1[size++] << 5.2,3.5,1.5,0.2; // Iris-setosa
	x_set1[size++] << 5.2,3.4,1.4,0.2; // Iris-setosa
	x_set1[size++] << 4.7,3.2,1.6,0.2; // Iris-setosa
	x_set1[size++] << 4.8,3.1,1.6,0.2; // Iris-setosa
	x_set1[size++] << 5.4,3.4,1.5,0.4; // Iris-setosa
	x_set1[size++] << 5.2,4.1,1.5,0.1; // Iris-setosa
	x_set1[size++] << 5.5,4.2,1.4,0.2; // Iris-setosa
	x_set1[size++] << 4.9,3.1,1.5,0.1; // Iris-setosa
	x_set1[size++] << 5.0,3.2,1.2,0.2; // Iris-setosa
	x_set1[size++] << 5.5,3.5,1.3,0.2; // Iris-setosa
	x_set1[size++] << 4.9,3.1,1.5,0.1; // Iris-setosa
	x_set1[size++] << 4.4,3.0,1.3,0.2; // Iris-setosa
	x_set1[size++] << 5.1,3.4,1.5,0.2; // Iris-setosa
	x_set1[size++] << 5.0,3.5,1.3,0.3; // Iris-setosa
	x_set1[size++] << 4.5,2.3,1.3,0.3; // Iris-setosa
	x_set1[size++] << 4.4,3.2,1.3,0.2; // Iris-setosa
	x_set1[size++] << 5.0,3.5,1.6,0.6; // Iris-setosa
	x_set1[size++] << 5.1,3.8,1.9,0.4; // Iris-setosa
	x_set1[size++] << 4.8,3.0,1.4,0.3; // Iris-setosa
	x_set1[size++] << 5.1,3.8,1.6,0.2; // Iris-setosa
	x_set1[size++] << 4.6,3.2,1.4,0.2; // Iris-setosa
	x_set1[size++] << 5.3,3.7,1.5,0.2; // Iris-setosa
	x_set1[size++] << 5.0,3.3,1.4,0.2; // Iris-setosa

	x_set2[size2++] << 7.0,3.2,4.7,1.4; // Iris-versicolor
	x_set2[size2++] << 6.4,3.2,4.5,1.5; // Iris-versicolor
	x_set2[size2++] << 6.9,3.1,4.9,1.5; // Iris-versicolor
	x_set2[size2++] << 5.5,2.3,4.0,1.3; // Iris-versicolor
	x_set2[size2++] << 6.5,2.8,4.6,1.5; // Iris-versicolor
	x_set2[size2++] << 5.7,2.8,4.5,1.3; // Iris-versicolor
	x_set2[size2++] << 6.3,3.3,4.7,1.6; // Iris-versicolor
	x_set2[size2++] << 4.9,2.4,3.3,1.0; // Iris-versicolor
	x_set2[size2++] << 6.6,2.9,4.6,1.3; // Iris-versicolor
	x_set2[size2++] << 5.2,2.7,3.9,1.4; // Iris-versicolor
	x_set2[size2++] << 5.0,2.0,3.5,1.0; // Iris-versicolor
	x_set2[size2++] << 5.9,3.0,4.2,1.5; // Iris-versicolor
	x_set2[size2++] << 6.0,2.2,4.0,1.0; // Iris-versicolor
	x_set2[size2++] << 6.1,2.9,4.7,1.4; // Iris-versicolor
	x_set2[size2++] << 5.6,2.9,3.6,1.3; // Iris-versicolor
	x_set2[size2++] << 6.7,3.1,4.4,1.4; // Iris-versicolor
	x_set2[size2++] << 5.6,3.0,4.5,1.5; // Iris-versicolor
	x_set2[size2++] << 5.8,2.7,4.1,1.0; // Iris-versicolor
	x_set2[size2++] << 6.2,2.2,4.5,1.5; // Iris-versicolor
	x_set2[size2++] << 5.6,2.5,3.9,1.1; // Iris-versicolor
	x_set2[size2++] << 5.9,3.2,4.8,1.8; // Iris-versicolor
	x_set2[size2++] << 6.1,2.8,4.0,1.3; // Iris-versicolor
	x_set2[size2++] << 6.3,2.5,4.9,1.5; // Iris-versicolor
	x_set2[size2++] << 6.1,2.8,4.7,1.2; // Iris-versicolor
	x_set2[size2++] << 6.4,2.9,4.3,1.3; // Iris-versicolor
	x_set2[size2++] << 6.6,3.0,4.4,1.4; // Iris-versicolor
	x_set2[size2++] << 6.8,2.8,4.8,1.4; // Iris-versicolor
	x_set2[size2++] << 6.7,3.0,5.0,1.7; // Iris-versicolor
	x_set2[size2++] << 6.0,2.9,4.5,1.5; // Iris-versicolor
	x_set2[size2++] << 5.7,2.6,3.5,1.0; // Iris-versicolor
	x_set2[size2++] << 5.5,2.4,3.8,1.1; // Iris-versicolor
	x_set2[size2++] << 5.5,2.4,3.7,1.0; // Iris-versicolor
	x_set2[size2++] << 5.8,2.7,3.9,1.2; // Iris-versicolor
	x_set2[size2++] << 6.0,2.7,5.1,1.6; // Iris-versicolor
	x_set2[size2++] << 5.4,3.0,4.5,1.5; // Iris-versicolor
	x_set2[size2++] << 6.0,3.4,4.5,1.6; // Iris-versicolor
	x_set2[size2++] << 6.7,3.1,4.7,1.5; // Iris-versicolor
	x_set2[size2++] << 6.3,2.3,4.4,1.3; // Iris-versicolor
	x_set2[size2++] << 5.6,3.0,4.1,1.3; // Iris-versicolor
	x_set2[size2++] << 5.5,2.5,4.0,1.3; // Iris-versicolor
	x_set2[size2++] << 5.5,2.6,4.4,1.2; // Iris-versicolor
	x_set2[size2++] << 6.1,3.0,4.6,1.4; // Iris-versicolor
	x_set2[size2++] << 5.8,2.6,4.0,1.2; // Iris-versicolor
	x_set2[size2++] << 5.0,2.3,3.3,1.0; // Iris-versicolor
	x_set2[size2++] << 5.6,2.7,4.2,1.3; // Iris-versicolor
	x_set2[size2++] << 5.7,3.0,4.2,1.2; // Iris-versicolor
	x_set2[size2++] << 5.7,2.9,4.2,1.3; // Iris-versicolor
	x_set2[size2++] << 6.2,2.9,4.3,1.3; // Iris-versicolor
	x_set2[size2++] << 5.1,2.5,3.0,1.1; // Iris-versicolor
	x_set2[size2++] << 5.7,2.8,4.1,1.3; // Iris-versicolor

	x_set3[size3++] << 6.3,3.3,6.0,2.5; // Iris-virginica
	x_set3[size3++] << 5.8,2.7,5.1,1.9; // Iris-virginica
	x_set3[size3++] << 7.1,3.0,5.9,2.1; // Iris-virginica
	x_set3[size3++] << 6.3,2.9,5.6,1.8; // Iris-virginica
	x_set3[size3++] << 6.5,3.0,5.8,2.2; // Iris-virginica
	x_set3[size3++] << 7.6,3.0,6.6,2.1; // Iris-virginica
	x_set3[size3++] << 4.9,2.5,4.5,1.7; // Iris-virginica
	x_set3[size3++] << 7.3,2.9,6.3,1.8; // Iris-virginica
	x_set3[size3++] << 6.7,2.5,5.8,1.8; // Iris-virginica
	x_set3[size3++] << 7.2,3.6,6.1,2.5; // Iris-virginica
	x_set3[size3++] << 6.5,3.2,5.1,2.0; // Iris-virginica
	x_set3[size3++] << 6.4,2.7,5.3,1.9; // Iris-virginica
	x_set3[size3++] << 6.8,3.0,5.5,2.1; // Iris-virginica
	x_set3[size3++] << 5.7,2.5,5.0,2.0; // Iris-virginica
	x_set3[size3++] << 5.8,2.8,5.1,2.4; // Iris-virginica
	x_set3[size3++] << 6.4,3.2,5.3,2.3; // Iris-virginica
	x_set3[size3++] << 6.5,3.0,5.5,1.8; // Iris-virginica
	x_set3[size3++] << 7.7,3.8,6.7,2.2; // Iris-virginica
	x_set3[size3++] << 7.7,2.6,6.9,2.3; // Iris-virginica
	x_set3[size3++] << 6.0,2.2,5.0,1.5; // Iris-virginica
	x_set3[size3++] << 6.9,3.2,5.7,2.3; // Iris-virginica
	x_set3[size3++] << 5.6,2.8,4.9,2.0; // Iris-virginica
	x_set3[size3++] << 7.7,2.8,6.7,2.0; // Iris-virginica
	x_set3[size3++] << 6.3,2.7,4.9,1.8; // Iris-virginica
	x_set3[size3++] << 6.7,3.3,5.7,2.1; // Iris-virginica
	x_set3[size3++] << 7.2,3.2,6.0,1.8; // Iris-virginica
	x_set3[size3++] << 6.2,2.8,4.8,1.8; // Iris-virginica
	x_set3[size3++] << 6.1,3.0,4.9,1.8; // Iris-virginica
	x_set3[size3++] << 6.4,2.8,5.6,2.1; // Iris-virginica
	x_set3[size3++] << 7.2,3.0,5.8,1.6; // Iris-virginica
	x_set3[size3++] << 7.4,2.8,6.1,1.9; // Iris-virginica
	x_set3[size3++] << 7.9,3.8,6.4,2.0; // Iris-virginica
	x_set3[size3++] << 6.4,2.8,5.6,2.2; // Iris-virginica
	x_set3[size3++] << 6.3,2.8,5.1,1.5; // Iris-virginica
	x_set3[size3++] << 6.1,2.6,5.6,1.4; // Iris-virginica
	x_set3[size3++] << 7.7,3.0,6.1,2.3; // Iris-virginica
	x_set3[size3++] << 6.3,3.4,5.6,2.4; // Iris-virginica
	x_set3[size3++] << 6.4,3.1,5.5,1.8; // Iris-virginica
	x_set3[size3++] << 6.0,3.0,4.8,1.8; // Iris-virginica
	x_set3[size3++] << 6.9,3.1,5.4,2.1; // Iris-virginica
	x_set3[size3++] << 6.7,3.1,5.6,2.4; // Iris-virginica
	x_set3[size3++] << 6.9,3.1,5.1,2.3; // Iris-virginica
	x_set3[size3++] << 5.8,2.7,5.1,1.9; // Iris-virginica
	x_set3[size3++] << 6.8,3.2,5.9,2.3; // Iris-virginica
	x_set3[size3++] << 6.7,3.3,5.7,2.5; // Iris-virginica
	x_set3[size3++] << 6.7,3.0,5.2,2.3; // Iris-virginica
	x_set3[size3++] << 6.3,2.5,5.0,1.9; // Iris-virginica
	x_set3[size3++] << 6.5,3.0,5.2,2.0; // Iris-virginica
	x_set3[size3++] << 6.2,3.4,5.4,2.3; // Iris-virginica
	x_set3[size3++] << 5.9,3.0,5.1,1.8; // Iris-virginica

	uranus::Vector<Dim> mean_1;
	for (int i = 0; i < Dim; ++i)mean_1(i) = 0;

	uranus::Vector<Dim> mean_2;
	for (int i = 0; i < Dim; ++i)mean_2(i) = 0;

	uranus::Vector<Dim> mean_3;
	for (int i = 0; i < Dim; ++i)mean_3(i) = 0;
	
	uranus::SquareMatrix<Dim> Si_1;
	for (int i = 0; i < Dim*Dim; ++i)Si_1(i) = 0;

	uranus::SquareMatrix<Dim> Si_2;
	for (int i = 0; i < Dim*Dim; ++i)Si_2(i) = 0;

	uranus::SquareMatrix<Dim> Si_3;
	for (int i = 0; i < Dim*Dim; ++i)Si_3(i) = 0;
	
	// step1 均值向量
	for (int i = 0; i < size; ++i) mean_1 += x_set1[i];
	mean_1 = mean_1 / size;
	for (int i = 0; i < size2; ++i) mean_2 += x_set2[i];
	mean_2 = mean_2 / size2;
	for (int i = 0; i < size3; ++i) mean_3 += x_set3[i];
	mean_3 = mean_3 / size3;

	// step2 类内离散度矩阵
	for (int i = 0; i < size; ++i)
		Si_1 += (x_set1[i] - mean_1)*(x_set1[i] - mean_1).transpose();
	for (int i = 0; i < size2; ++i)
		Si_2 += (x_set2[i] - mean_2)*(x_set2[i] - mean_2).transpose();
	for (int i = 0; i < size3; ++i)
		Si_3 += (x_set3[i] - mean_3)*(x_set3[i] - mean_3).transpose();

	//cout << "Si_1=\n" << Si_1 << endl << endl;
	//cout << "Si_2=\n" << Si_2 << endl << endl;
	//cout << "Si_3=\n" << Si_3 << endl << endl;
	// step3
	// 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
	uranus::SquareMatrix<Dim> Sw = Si_1 + Si_2 + Si_3;
	cout << "Sw=\n" << Sw << endl << endl;
	
	// step4
	// 样本类间离散度矩阵SB
	//uranus::SquareMatrix<Dim> Sb = (mean_1 - mean_2) * (mean_1 - mean_2).transpose();
	uranus::SquareMatrix<Dim> Sb_12 = (mean_1 - mean_2) * (mean_1 - mean_2).transpose();
	uranus::SquareMatrix<Dim> Sb_13 = (mean_1 - mean_3) * (mean_1 - mean_3).transpose();
	uranus::SquareMatrix<Dim> Sb_23 = (mean_2 - mean_3) * (mean_2 - mean_3).transpose();

	// step5
	// Fisher准则函数 -- 最佳投影方向
	// uranus::SquareMatrix<Dim> Jw = Sb*Sw.inverse();
	// w* = \argmax J(w)

	// uranus::Vector<Dim> argW = Sw.inverse()*(mean_1 - mean_2);
	uranus::Vector<Dim> argW_12 = Sw.inverse()*(mean_1 - mean_2);
	uranus::Vector<Dim> argW_13 = Sw.inverse()*(mean_1 - mean_3);
	uranus::Vector<Dim> argW_23 = Sw.inverse()*(mean_2 - mean_3);
	cout << "argW_12=\n" << argW_12 << endl << endl;
	cout << "argW_13=\n" << argW_13 << endl << endl;
	cout << "argW_23=\n" << argW_23 << endl << endl;

	// step6求阈值 W0 
	// uranus::Vector<1> W0 = argW.transpose()*mean_1 / 2 + argW.transpose()*mean_2 / 2;
	uranus::Vector<1> W0_12 = argW_12.transpose()*mean_1 / 2 + argW_12.transpose()*mean_2 / 2;
	uranus::Vector<1> W0_13 = argW_13.transpose()*mean_1 / 2 + argW_13.transpose()*mean_3 / 2;
	uranus::Vector<1> W0_23 = argW_23.transpose()*mean_2 / 2 + argW_23.transpose()*mean_3 / 2;

	// step7线性变换
	std::vector<uranus::Vector<1>> D1(size);
	std::vector<uranus::Vector<1>> D2(size2);
	std::vector<uranus::Vector<1>> D3(size3);

	// 1-2分类
	cout << "1-2分类" << endl;
	cout << "D1=" << endl;
	for (int i = 0; i < size; ++i)
	{
		D1[i] = argW_12.transpose()*x_set1[i];
		cout << D1[i] << endl;
	}
	cout << "Wo_12=\n" << W0_12 << endl << endl;
	cout << "D2=" << endl;
	for (int i = 0; i < size2; ++i)
	{
		D2[i] = argW_12.transpose()*x_set2[i];
		cout << D2[i] << endl;
	}
	// 1-3分类
	cout << "1-3分类" << endl;
	cout << "D1=" << endl;
	for (int i = 0; i < size; ++i)
	{
		D1[i] = argW_13.transpose()*x_set1[i];
		cout << D1[i] << endl;
	}
	cout << "Wo_13=\n" << W0_13 << endl << endl;
	cout << "D3=" << endl;
	for (int i = 0; i < size3; ++i)
	{
		D3[i] = argW_13.transpose()*x_set3[i];
		cout << D3[i] << endl;
	}
	// 2-3分类
	cout << "2-3分类" << endl;
	cout << "D2=" << endl;
	for (int i = 0; i < size; ++i)
	{
		D2[i] = argW_23.transpose()*x_set2[i];
		cout << D2[i] << endl;
	}
	cout << "Wo_23=\n" << W0_23 << endl << endl;
	cout << "D3=" << endl;
	for (int i = 0; i < size3; ++i)
	{
		D3[i] = argW_23.transpose()*x_set3[i];
		cout << D3[i] << endl;
	}
	return EXIT_SUCCESS;
}