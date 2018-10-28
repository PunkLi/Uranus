/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"
#include "Fisher.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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
	using sample_set = uranus::Tensor<feature_rows>::sample_set;
	using tensor = uranus::Tensor<feature_rows>::TensorType;

	constexpr double E = 2.718282;
	constexpr double In_omega = 97.0 / 111.0;
	constexpr double c1 = 97.0 / 208;
	constexpr double c2 = 111.0 / 208;
	uranus::Data_Wrapper<feature_rows> wrapper(path, data_class);
	uranus::Tensor<feature_rows> data(wrapper, data_class);

	uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_1);
	uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_2);

	tensor tensor_x1;
	tensor tensor_x2;

	sample_set train_x1, train_x2, test_x1, test_x2;
	uranus::Vector<feature_rows> mean_0, mean_1;
	uranus::SquareMatrix<Dim> Sw;
	uranus::SquareMatrix<Dim> Sw_bayes;
	

	constexpr int K = 2;
	for (int try_n = 0; try_n < 1; ++try_n)
	{
		double Sw_norm = 0;
		for (int step = 0; step < 3; ++step)
		{
			tensor tensor_x1 = data.k_fold_crossValidation<K>(0);
			tensor tensor_x2 = data.k_fold_crossValidation<K>(1);
			train_x1 = tensor_x1[0];
			train_x2 = tensor_x2[0];
			test_x1 = tensor_x1[1];
			test_x2 = tensor_x2[1];

// step1 均值向量
			mean_0 = data.get_mean(train_x1);
			mean_1 = data.get_mean(train_x2);

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
			Sw = Si_1 + Si_2;
			Sw_bayes = c1 * Si_1 + c2 * Si_2;
			//cout << "Sw=\n" << Sw << endl << endl;

			double cond = (Sw.norm())*(Sw.inverse().norm());

			if (Sw.norm() > Sw_norm) Sw_norm = Sw.norm();

			cout << "norm: " << Sw_norm << "\t cond number: " << cond << endl;
		}
		std::vector<uranus::Vector<1>> D1(test_x1.size());
		std::vector<uranus::Vector<1>> D2(test_x2.size());
		cout << "\n";

// step4
		// 样本类间离散度矩阵SB
		init_Sb_(Dim, mean_0, mean_1);

		//cout << "Sb=\n" << Sb_(mean_0, mean_1) << endl << endl;
		
		uranus::SquareMatrix<Dim> A = Sw.inverse()*Sb_(mean_0, mean_1);

		Eigen::EigenSolver<uranus::SquareMatrix<Dim>> eigen_solver(A);
		Eigen::VectorXcd evals = eigen_solver.eigenvalues();
		//cout << "\n eigen = " << evals << "\n";
		Eigen::VectorXd evalsReal = evals.real(); // 获取特征值实数部分
		Eigen::VectorXd evalsimag = evals.imag();

		//cout << "\n eigen real = \n" << evalsReal << "\n";
		//cout << "\n eigen imag = \n" << evalsimag << "\n";

		std::vector<double> eigenval;

		for (int i = 0; i < Dim; ++i)
			if (evalsimag(i) == 0) eigenval.push_back(evalsReal(i));

		std::sort(eigenval.begin(), eigenval.end(), [](double lhs, double rhs) {return lhs > rhs; });
		//cout << "real:" << eigenval.size() << endl;;
		//for (int i = 0; i < eigenval.size(); ++i)
		//	cout << eigenval[i] << endl;

		Eigen::MatrixXcd eigenvect = eigen_solver.eigenvectors(); 
		Eigen::MatrixXd eigenvectReal = eigenvect.real();
		Eigen::MatrixXf::Index eigenvectMax;
		eigenvectReal.rowwise().sum().maxCoeff(&eigenvectMax);
		uranus::Vector<Dim> q;
		for (int i = 0; i < Dim; ++i)
			q(i) = eigenvectReal(i, eigenvectMax); // [60x1]
		
// step5
		// Fisher准则函数 -- 最佳投影方向
		// uranus::SquareMatrix<Dim> Jw = Sb*Sw.inverse();
		// w* = \argmax J(w)

		// 方法一
		init_argW_(Dim, mean_0, mean_1);
		uranus::Vector<feature_rows> argW_bayes = Sw.inverse()*(mean_0 - mean_1);
		//cout << "argW=\n" << argW_(mean_0, mean_1) << endl << endl;

// step6求阈值 W0 
		// 传统判别
		init_W0_(Dim, mean_0, mean_1);
		cout << "Wo=\n" << W0_(mean_0, mean_1) << endl << endl;
		// bayes判别
		double Inw = log(In_omega) / log(E);
		//cout << "Inw:" << Inw << "\n";
		uranus::Vector<1> Wo = (mean_0 + mean_1).transpose()*Sw.inverse()*(mean_0 - mean_1)
			- static_cast<uranus::Vector<1>>(Inw);
		cout << "Bayes W0 = " << Wo << endl;

// step7线性变换
		for (int i = 0; i < test_x1.size(); ++i)
			D1[i] = argW_(mean_0, mean_1).transpose()*test_x1[i];

		for (int i = 0; i < test_x2.size(); ++i)
			D2[i] = argW_(mean_0, mean_1).transpose()*test_x2[i];

		cout << "\n传统投影 传统判别:\n";
		Evaluation(W0_(mean_0, mean_1), D1);
		Evaluation(W0_(mean_0, mean_1), D2);

		cout << "\n传统投影 贝叶斯判别:\n";
		Evaluation(Wo, D1);
		Evaluation(Wo, D2);
		cout << "\n---------------------------------\n";
		for (int i = 0; i < test_x1.size(); ++i)
			D1[i] = argW_bayes.transpose()*test_x1[i];

		for (int i = 0; i < test_x2.size(); ++i)
			D2[i] = argW_bayes.transpose()*test_x2[i];

		cout << "\n贝叶斯投影 传统判别:\n";
		Evaluation(W0_(mean_0, mean_1), D1);
		Evaluation(W0_(mean_0, mean_1), D2);
		
		cout << "\n贝叶斯投影 贝叶斯判别:\n";
		Evaluation(Wo, D1);
		Evaluation(Wo, D2);
		cout << "\n---------------------------------\n";
		for (int i = 0; i < test_x1.size(); ++i)
			D1[i] = q.transpose()*test_x1[i];

		for (int i = 0; i < test_x2.size(); ++i)
			D2[i] = q.transpose()*test_x2[i];

		cout << "\n特征值投影 传统判别:\n";
		Evaluation(W0_(mean_0, mean_1), D1);
		Evaluation(W0_(mean_0, mean_1), D2);

		cout << "\n特征值投影 贝叶斯判别:\n";
		Evaluation(Wo, D1);
		Evaluation(Wo, D2);
	}
	return EXIT_SUCCESS;
}