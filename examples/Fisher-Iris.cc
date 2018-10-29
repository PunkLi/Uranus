/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"
#include "Fisher.h"

constexpr int feature_rows = 4;
constexpr int Dim = feature_rows;

std::vector<int> data_class = { 50,50,50 };

std::string path = "../data/iris.data";

std::vector<uranus::Vector<feature_rows>> mean;
uranus::SquareMatrix<feature_rows> Si_1;
uranus::SquareMatrix<feature_rows> Si_2;
uranus::SquareMatrix<feature_rows> Si_3;
uranus::SquareMatrix<feature_rows> Sw;

int main(int argc, char *argv[])
{
	using namespace std;

	using sample_set = uranus::Tensor<feature_rows>::sample_set;
	using tensor = uranus::Tensor<feature_rows>::TensorType;

	uranus::Data_Wrapper<feature_rows> wrapper(path, data_class);
	uranus::Tensor<feature_rows> data(wrapper, data_class);

	uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_1);
	uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_2);
	uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_3);

	tensor tensor_x1, tensor_x2, tensor_x3;
	uranus::SquareMatrix<Dim> Sw;
	uranus::SquareMatrix<Dim> Sw_Bayes;

	sample_set train_x1, train_x2, train_x3, test_x1, test_x2, test_x3;
	uranus::Vector<Dim> mean_total, mean_0, mean_1, mean_2;
	uranus::Vector<Dim> mean_t1;// = data.get_mean(test_x1);
	uranus::Vector<Dim> mean_t2;// = data.get_mean(test_x2);
	uranus::Vector<Dim> mean_t3;// = data.get_mean(test_x3);

	double Sw_norm = 0;
	for (int try_n = 0; try_n < 10; ++try_n)
	{
		for (int step = 0; step < 1; ++step)
		{
			tensor_x1 = data.k_fold_crossValidation<2>(0);
			tensor_x2 = data.k_fold_crossValidation<2>(1);
			tensor_x3 = data.k_fold_crossValidation<2>(2);

			train_x1 = tensor_x1[0];
			train_x2 = tensor_x2[0];
			train_x3 = tensor_x3[0];

			test_x1 = tensor_x1[1];
			test_x2 = tensor_x2[1];
			test_x3 = tensor_x3[1];

			// step1 均值向量
			mean_0 = data.get_mean(train_x1);
			mean_1 = data.get_mean(train_x2);
			mean_2 = data.get_mean(train_x3);

			mean_t1 = data.get_mean(test_x1);
			mean_t2 = data.get_mean(test_x2);
			mean_t3 = data.get_mean(test_x3);

			mean_total = mean_0 / 3 + mean_1 / 3 + mean_2 / 3;

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
#pragma omp for
				for (int i = 0; i < train_x3.size(); ++i)
					Si_3 += (train_x3[i] - mean_2)*(train_x3[i] - mean_2).transpose();
				//cout << "Si_2=\n" << Si_2 << endl << endl;
			}

			// step3
			// 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
			Sw = Si_1 + Si_2 + Si_3;
			//cout << "Sw=\n" << Sw << endl << endl;
			const double c1 = train_x1.size() / (train_x1.size() + train_x2.size() + train_x3.size());
			const double c2 = train_x2.size() / (train_x1.size() + train_x2.size() + train_x3.size());
			const double c3 = train_x3.size() / (train_x1.size() + train_x2.size() + train_x3.size());
			Sw_Bayes = c1 * Si_1 + c2 * Si_2 + c3 * Si_3;
			//cout << "Sw=\n" << Sw << endl << endl;

			double cond = (Sw.norm())*(Sw.inverse().norm());

			if (Sw.norm() > Sw_norm) Sw_norm = Sw.norm();

			cout << "norm: " << Sw_norm << "\t cond number: " << cond << endl;
		}

		// step4
		// 样本类间离散度矩阵SB
		init_Sb_(Dim, mean_0, mean_1);
		init_Sb_(Dim, mean_0, mean_2);
		init_Sb_(Dim, mean_1, mean_2);
		//cout << "Sb=\n" << Sb_(mean_0, mean_1) << endl << endl;

		uranus::SquareMatrix<Dim> Sb 
			= (mean_0 - mean_total)*(mean_0 - mean_total).transpose() * 1 / 3
			+ (mean_0 - mean_total)*(mean_0 - mean_total).transpose() * 1 / 3
			+ (mean_0 - mean_total)*(mean_0 - mean_total).transpose() * 1 / 3;

		// step5
		// Fisher准则函数 -- 最佳投影方向
		// uranus::SquareMatrix<Dim> Jw = Sb*Sw.inverse();
		// w* = \argmax J(w)
		init_argW_(Dim, mean_0, mean_1);
		init_argW_(Dim, mean_0, mean_2);
		init_argW_(Dim, mean_1, mean_2);

		// 方法2：特征值
		uranus::SquareMatrix<Dim> A = Sw.inverse()*Sb;

		Eigen::EigenSolver<uranus::SquareMatrix<Dim>> eigen_solver(A);

		Eigen::VectorXcd evals = eigen_solver.eigenvalues();
		Eigen::MatrixXcd eigenvect = eigen_solver.eigenvectors();

		std::vector<std::pair<int,std::complex<double>>> eigenval; //  i, val[i] 
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

		Eigen::Matrix<std::complex < double>, 4, 2> eigen_Wc;

		for (int i = 0; i < 2; ++i)
			eigen_Wc.col(i) = eigenvect.col(eigenval[i].first);

		cout << "\n eigen =\n" << evals << "\n";
		cout << "\n eigenvector =\n" << eigenvect << "\n";
		cout << "\n eigen_Wc =\n" << eigen_Wc << "\n";

		constexpr int dim = 2;
		Eigen::Matrix<double, Dim, dim> eigen_W = eigen_Wc.real(); // 投影矩阵

		cout << "\n eigen_W =\n" << eigen_W << "\n";
		
		std::vector<uranus::Vector<dim>> DD1(test_x1.size());
		std::vector<uranus::Vector<dim>> DD2(test_x2.size());
		std::vector<uranus::Vector<dim>> DD3(test_x3.size());

// 再来一次Fisher
		// 投影后的均值
		uranus::Vector<dim> DD1_mean = eigen_W.transpose()*mean_t1,
							DD2_mean = eigen_W.transpose()*mean_t1,
							DD3_mean = eigen_W.transpose()*mean_t1;
		
		// 投影后的各元素
		for (int i = 0; i < test_x1.size(); ++i) DD1[i] = eigen_W.transpose()*test_x1[i];
		for (int i = 0; i < test_x2.size(); ++i) DD2[i] = eigen_W.transpose()*test_x2[i];
		for (int i = 0; i < test_x3.size(); ++i) DD3[i] = eigen_W.transpose()*test_x3[i];
		
		// init
		uranus::SquareMatrix<dim> plane_Si_1, plane_Si_2, plane_Si_3;
		// init
		uranus::setZero< uranus::SquareMatrix<dim>, dim>(plane_Si_1);
		uranus::setZero< uranus::SquareMatrix<dim>, dim>(plane_Si_2);
		uranus::setZero< uranus::SquareMatrix<dim>, dim>(plane_Si_3);
#pragma omp parallel
		{
#pragma omp for
			for (int i = 0; i < DD1.size(); ++i)
				plane_Si_1 += (DD1[i] - DD1_mean)*(DD1[i] - DD1_mean).transpose();
			//cout << "Si_1=\n" << Si_1 << endl << endl;
#pragma omp for
			for (int i = 0; i < DD2.size(); ++i)
				plane_Si_2 += (DD2[i] - DD2_mean)*(DD2[i] - DD2_mean).transpose();
			//cout << "Si_2=\n" << Si_2 << endl << endl;
#pragma omp for
			for (int i = 0; i < DD3.size(); ++i)
				plane_Si_3 += (DD3[i] - DD3_mean)*(DD3[i] - DD3_mean).transpose();
			//cout << "Si_2=\n" << Si_2 << endl << endl;
		}
		uranus::SquareMatrix<dim> plane_Sw = plane_Si_1 + plane_Si_2 + plane_Si_3; 
		const double pc1 = DD1.size() / (DD1.size() + DD2.size() + DD3.size());
		const double pc2 = DD2.size() / (DD1.size() + DD2.size() + DD3.size());
		const double pc3 = DD3.size() / (DD1.size() + DD2.size() + DD3.size());
		uranus::SquareMatrix<dim> plane_Sw_Bayes = pc1 * plane_Si_1 + pc2 * plane_Si_2 + pc3 * plane_Si_3;  // 贝叶斯
		double cond = (plane_Sw.norm())*(plane_Sw.inverse().norm());
		cout << "\ncond number: " << cond << endl;

		uranus::Vector<dim> plane_argW_12 = plane_Sw.inverse()*(DD1_mean - DD2_mean),
							plane_argW_13 = plane_Sw.inverse()*(DD1_mean - DD3_mean),
							plane_argW_23 = plane_Sw.inverse()*(DD2_mean - DD3_mean);

		uranus::Vector<1> plane_W0_12 = plane_argW_12.transpose()*(DD1_mean) / 2
									  + plane_argW_12.transpose()*(DD2_mean) / 2,

						  plane_W0_13 = plane_argW_13.transpose()*(DD1_mean) / 2
							 		  + plane_argW_13.transpose()*(DD3_mean) / 2,

						  plane_W0_23 = plane_argW_23.transpose()*(DD2_mean) / 2
									  + plane_argW_23.transpose()*(DD3_mean) / 2; // 考虑一下贝叶斯

		double p = 0.0, n = 0.0;
		for (int i = 0; i < DD1.size(); ++i)
		{
			bool res = Multi_Discriminant<dim>(plane_argW_12, plane_argW_13, plane_W0_12, plane_W0_13, DD1[i]);
			if (res)p++;
			else n++;
		}
		cout << "\n 准确率：" << "p = " << p << "\t n = " << n << "\t size = " << DD1.size() << "\n";

		p = 0.0, n = 0.0;
		for (int i = 0; i < DD2.size(); ++i)
		{
			bool res = Multi_Discriminant<dim>(plane_argW_12, plane_argW_23, plane_W0_12, plane_W0_23, DD2[i]);
			if (res)p++;
			else n++;
		}
		cout << "\n 准确率：" << "p = " << p << "\t n = " << n << "\t size = " << DD2.size() << "\n";

		p = 0.0, n = 0.0;
		for (int i = 0; i < DD3.size(); ++i)
		{
			bool res = Multi_Discriminant<dim>(plane_argW_13, plane_argW_23, plane_W0_13, plane_W0_23, DD3[i]);
			if (res)p++;
			else n++;
		}
		cout << "\n 准确率：" << "p = " << p << "\t n = " << n << "\t size = " << DD3.size() << "\n";
	}
	return EXIT_SUCCESS;
}