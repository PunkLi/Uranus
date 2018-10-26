/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"
#include "Fisher.h"

constexpr int feature_rows = 60;
constexpr int Dim = feature_rows;

std::vector<int> data_class = { 97,111 };

std::string path = "../data/sonar.all-data";

std::vector<uranus::Vector<feature_rows>> mean;
uranus::SquareMatrix<feature_rows> Si_1;
uranus::SquareMatrix<feature_rows> Si_2;
uranus::SquareMatrix<feature_rows> Sw;

int main(int argc, char *argv[])
{
	using namespace std;
	const int batch_size = 10;

	using namespace std;
	using sample_set = uranus::Tensor<feature_rows>::sample_set;
	using tensor = uranus::Tensor<feature_rows>::TensorType;

	uranus::Data_Wrapper<feature_rows> wrapper(path, data_class);
	uranus::Tensor<feature_rows> data(wrapper, data_class);

	uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_1);
	uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_2);
	
	tensor tensor_x1 = data.k_fold_crossValidation<2>(0, true);
	tensor tensor_x2 = data.k_fold_crossValidation<2>(1, true);

	sample_set train_x1 = tensor_x1[0];
	sample_set train_x2 = tensor_x2[0];

	sample_set test_x1 = tensor_x1[1];
	sample_set test_x2 = tensor_x2[1];

	// step1 均值向量
	auto mean_0 = data.get_mean(train_x1);
	auto mean_1 = data.get_mean(train_x2);

	// step2 类内离散度矩阵
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < train_x1.size(); ++i)
			Si_1 += (train_x1[i] - mean_0)*(train_x1[i] - mean_0).transpose();
		//cout << "Si_1=\n" << Si_1 << endl << endl;
#pragma omp for
		for (int i = 0; i < train_x2.size(); ++i)
			Si_2 += (train_x2[i] - mean_1)*(train_x2[i] - mean_1).transpose();
		//cout << "Si_2=\n" << Si_2 << endl << endl;
	}

	// step3
	// 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
	uranus::SquareMatrix<Dim> Sw = Si_1 + Si_2;
	//cout << "Sw=\n" << Sw << endl << endl;

	// step4
	// 样本类间离散度矩阵SB
	init_Sb_(Dim, mean_0, mean_1);
	//cout << "Sb=\n" << Sb_(mean_0, mean_1) << endl << endl;

	// step5
	// Fisher准则函数 -- 最佳投影方向
	// uranus::SquareMatrix<Dim> Jw = Sb*Sw.inverse();
	// w* = \argmax J(w)
	init_argW_(Dim, mean_0, mean_1);
	//cout << "argW=\n" << argW_(mean_0, mean_1) << endl << endl;

	// step6求阈值 W0 
	init_W0_(Dim, mean_0, mean_1);
	//cout << "Wo=\n" << W0_(mean_0, mean_1) << endl << endl;

	// step7线性变换
	std::vector<uranus::Vector<1>> D1(test_x1.size());
	std::vector<uranus::Vector<1>> D2(test_x2.size());

	for (int i = 0; i < test_x1.size(); ++i)
	{
		D1[i] = argW_(mean_0, mean_1).transpose()*test_x1[i];
		//cout << D1[i] << endl;
	}
	//cout << "\n";
	for (int i = 0; i < test_x2.size(); ++i)
	{
		D2[i] = argW_(mean_0, mean_1).transpose()*test_x2[i];
		//cout << D2[i] << endl;
	}

	Evaluation(W0_(mean_0, mean_1), D1);
	Evaluation(W0_(mean_0, mean_1), D2);

	return EXIT_SUCCESS;
}