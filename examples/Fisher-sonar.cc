/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"
#include "Fisher.h"

constexpr int Dim = 60;

std::vector<int> data_class = { 97,111 };

std::string path = "../data/sonar.all-data";

uranus::SquareMatrix<Dim> Si_1;
uranus::SquareMatrix<Dim> Si_2;
uranus::SquareMatrix<Dim> Sw;

int main(int argc, char *argv[])
{
	using namespace std;
	using sample_set = uranus::Tensor<Dim>::sample_set;
	using tensor = uranus::Tensor<Dim>::TensorType;

	constexpr double E = 2.718282;
	constexpr double In_omega = 97.0 / 111.0;
	constexpr double c1 = 97.0 / 208;
	constexpr double c2 = 111.0 / 208;
	uranus::Data_Wrapper<Dim> wrapper(path, data_class);
	uranus::Tensor<Dim> data(wrapper, data_class);

	uranus::setZero<uranus::SquareMatrix<Dim>, Dim>(Si_1);
	uranus::setZero<uranus::SquareMatrix<Dim>, Dim>(Si_2);

	tensor tensor_x1;
	tensor tensor_x2;

	sample_set train_x1, train_x2, test_x1, test_x2;
	uranus::Vector<Dim> mean_0, mean_1;
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
		Eigen::MatrixXcd eigenvect = eigen_solver.eigenvectors();

		std::vector<std::pair<int, std::complex<double>>> eigenval; //  i, val[i] 
		for (int i = 0; i < Dim; ++i)
			eigenval.push_back(make_pair(i, evals(i)));

		std::sort(
			eigenval.begin(),
			eigenval.end(),
			[](std::pair<int, std::complex<double>> lhs,
				std::pair<int, std::complex<double>> rhs)
		{
			return lhs.second.real() > rhs.second.real();
		}
		);
		constexpr int dim = 1;

		Eigen::Matrix<std::complex < double>, Dim, dim> eigen_Wc;

		for (int i = 0; i < 1; ++i)
			eigen_Wc.col(i) = eigenvect.col(eigenval[i].first);

		//cout << "\n eigen =\n" << evals << "\n";
		//cout << "\n eigenvector =\n" << eigenvect << "\n";
		//cout << "\n eigen_Wc =\n" << eigen_Wc << "\n";

		Eigen::Matrix<double, Dim, dim> eigen_W = eigen_Wc.real(); // 投影矩阵

		//cout << "\n eigen_W =\n" << eigen_W << "\n";

// step5
		// Fisher准则函数 -- 最佳投影方向
		// uranus::SquareMatrix<Dim> Jw = Sb*Sw.inverse();
		// w* = \argmax J(w)

		// 方法一
		init_argW_(Dim, mean_0, mean_1);
		uranus::Vector<Dim> argW_bayes = Sw.inverse()*(mean_0 - mean_1);
		//cout << "argW=\n" << argW_(mean_0, mean_1) << endl << endl;

// step6求阈值 W0 
		// 传统判别
		init_W0_(Dim, mean_0, mean_1);
		cout << "Wo=\n" << W0_(mean_0, mean_1) << endl << endl;
		// bayes判别
		double Inw = log(In_omega) / log(E);

		uranus::Vector<1> _InW_;
		_InW_ << Inw;
		//cout << "Inw:" << Inw << "\n";
		uranus::Vector<1> Wo = (mean_0 + mean_1).transpose()*Sw.inverse()*(mean_0 - mean_1) - _InW_;

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
			D1[i] = eigen_W.transpose()*test_x1[i];

		for (int i = 0; i < test_x2.size(); ++i)
			D2[i] = eigen_W.transpose()*test_x2[i];

		cout << "\n特征值投影 传统判别:\n";
		Evaluation(W0_(mean_0, mean_1), D1);
		Evaluation(W0_(mean_0, mean_1), D2);

		cout << "\n特征值投影 贝叶斯判别:\n";
		Evaluation(Wo, D1);
		Evaluation(Wo, D2);
	}

	return EXIT_SUCCESS;
}