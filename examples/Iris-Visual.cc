/**
 * 模式识别 Fisher
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "Tensor.hpp"
#include <opencv2/opencv.hpp>
#include "Fisher.h"

#define SHOW_TRAIN_SET
#define SHOW_TEST_SET

template < class T>
std::string ConvertToString(T value)
{
	std::stringstream ss;
	ss << value;
	return ss.str();
}

constexpr int feature_rows = 4;
constexpr int Dim = feature_rows;
constexpr int dim = 2;

std::vector<int> data_class = { 50,50,50 };

std::string path = "../data/iris.data";

std::vector<uranus::Vector<feature_rows>> mean;
uranus::SquareMatrix<feature_rows> Si_1;
uranus::SquareMatrix<feature_rows> Si_2;
uranus::SquareMatrix<feature_rows> Si_3;
uranus::SquareMatrix<feature_rows> Sw;

int main(int argc, char *argv[])
{
	// init
	using namespace std;
	using sample_set = uranus::Tensor<feature_rows>::sample_set;
	using tensor = uranus::Tensor<feature_rows>::TensorType;

	uranus::Data_Wrapper<feature_rows> wrapper(path, data_class);
	uranus::Tensor<feature_rows> data(wrapper, data_class);

	tensor tensor_x1, tensor_x2, tensor_x3;
	uranus::SquareMatrix<Dim> Sw;
	uranus::SquareMatrix<Dim> Sw_Bayes;

	sample_set train_x1, train_x2, train_x3, test_x1, test_x2, test_x3;
	uranus::Vector<Dim> mean_total, mean_0, mean_1, mean_2;
	uranus::Vector<Dim> mean_t1;// = data.get_mean(test_x1);
	uranus::Vector<Dim> mean_t2;// = data.get_mean(test_x2);
	uranus::Vector<Dim> mean_t3;// = data.get_mean(test_x3);

	cv::Mat draw_cond_pic = cv::Mat(480, 640, CV_8UC3);
	constexpr int step_size = 30;
	constexpr int draw_cond_x_scale = 640 / step_size;

	cv::Mat draw_train_set = cv::Mat(960, 1280, CV_8UC3);
	cv::Mat draw_test_set = cv::Mat(960, 1280, CV_8UC3);

	cv::Point2f zero_pt;
	zero_pt.x = 640;
	zero_pt.y = 480;
	constexpr int draw_set_scale = 20;
	cv::line(draw_train_set, cv::Point(0, zero_pt.y), cv::Point(1280, zero_pt.y), cv::Scalar(255, 255, 255), 2); // x
	cv::line(draw_train_set, cv::Point(zero_pt.x, 0), cv::Point(zero_pt.x, 960), cv::Scalar(255, 255, 255), 2); // y

	Eigen::Matrix<double, Dim, dim> eigen_W;   // 特征值投影
	uranus::SquareMatrix<Dim> Sb;              // 类间离散度矩阵

	cv::Point2f norm_pt_arr[step_size];
	cv::Point2f cond_pt_arr[step_size];

	double balance = 0;

	// trian
	for (int step = 0; step < step_size; ++step)
	{
		draw_train_set.copyTo(draw_test_set);

		tensor_x1 = data.k_fold_crossValidation<2>(0);
		tensor_x2 = data.k_fold_crossValidation<2>(1);
		tensor_x3 = data.k_fold_crossValidation<2>(2);

		train_x1 = tensor_x1[0];
		train_x2 = tensor_x2[0];
		train_x3 = tensor_x3[0];

		// step1 均值向量
		mean_0 = data.get_mean(train_x1);
		mean_1 = data.get_mean(train_x2);
		mean_2 = data.get_mean(train_x3);

		mean_total = mean_0 / 3 + mean_1 / 3 + mean_2 / 3;  // train mean

		// step2 类内离散度矩阵
		uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_1);
		uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_2);
		uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_3);
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

		Sw = Si_1 + Si_2 + Si_3;     // 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
		//cout << "Sw=\n" << Sw << endl << endl;
		//const double c1 = train_x1.size() / (train_x1.size() + train_x2.size() + train_x3.size());
		//const double c2 = train_x2.size() / (train_x1.size() + train_x2.size() + train_x3.size());
		//const double c3 = train_x3.size() / (train_x1.size() + train_x2.size() + train_x3.size());
		//Sw_Bayes = c1 * Si_1 + c2 * Si_2 + c3 * Si_3; // 带贝叶斯的 Sw
		//cout << "Sw=\n" << Sw << endl << endl;

		double cond = (Sw.norm())*(Sw.inverse().norm());  // 类内离散度矩阵条件数

		double _b = abs(Sw.norm() - cond);
		//if (_b > balance)
		//	balance = _b;

		cout << "norm: " << Sw.norm() << "\t cond number: " << cond << endl;

		// visual
		cv::Point2f norm_pt;
		norm_pt.x = step * draw_cond_x_scale;
		norm_pt.y = (480 - Sw.norm());
		norm_pt_arr[step] = norm_pt;
		cout << "norm_pt" << norm_pt << endl;

		cv::Point2f cond_pt;
		cond_pt.x = step * draw_cond_x_scale;
		cond_pt.y = (480 - cond);
		cond_pt_arr[step] = cond_pt;
		cout << "cond_pt" << cond_pt << endl;

		cv::circle(draw_cond_pic, cond_pt, 5, cv::Scalar(0, 0, 255), 5);
		std::string cond_str = ConvertToString(cond);
		putText(draw_cond_pic, cond_str, cond_pt, CV_FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(0, 0, 255), 1);
		if (step) cv::line(draw_cond_pic, cond_pt_arr[step], cond_pt_arr[step - 1], cv::Scalar(0, 0, 255), 3);

		cv::circle(draw_cond_pic, norm_pt, 5, cv::Scalar(255, 0, 0), 5);
		std::string norm_str = ConvertToString(Sw.norm());
		putText(draw_cond_pic, norm_str, norm_pt, CV_FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(0, 0, 255), 1);
		if (step) cv::line(draw_cond_pic, norm_pt_arr[step], norm_pt_arr[step - 1], cv::Scalar(255, 0, 0), 3);

		cv::imshow("Sw (R:cond number  B: Norm)", draw_cond_pic);

		Sb = (mean_0 - mean_total)*(mean_0 - mean_total).transpose() * 1 / 3
			+ (mean_1 - mean_total)*(mean_1 - mean_total).transpose() * 1 / 3
			+ (mean_2 - mean_total)*(mean_2 - mean_total).transpose() * 1 / 3;

		// 特征值
		uranus::SquareMatrix<Dim> A = Sw.inverse()*Sb;

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

		Eigen::Matrix<std::complex < double>, 4, 2> eigen_Wc;

		for (int i = 0; i < 2; ++i)
			eigen_Wc.col(i) = eigenvect.col(eigenval[i].first);

		cout << "\n eigen =\n" << evals << "\n";
		cout << "\n eigenvector =\n" << eigenvect << "\n";
		cout << "\n eigen_Wc =\n" << eigen_Wc << "\n";
		eigen_W = eigen_Wc.real(); // 投影矩阵

		cout << "\n eigen_W =\n" << eigen_W << "\n";

		std::vector<uranus::Vector<dim>> train_pt1(train_x1.size());
		std::vector<uranus::Vector<dim>> train_pt2(train_x2.size());
		std::vector<uranus::Vector<dim>> train_pt3(train_x3.size());

		test_x1 = tensor_x1[1];
		test_x2 = tensor_x2[1];
		test_x3 = tensor_x3[1];
		std::vector<uranus::Vector<dim>> test_pt1(test_x1.size());
		std::vector<uranus::Vector<dim>> test_pt2(test_x2.size());
		std::vector<uranus::Vector<dim>> test_pt3(test_x3.size());

		// 再来一次Fisher
		mean_t1 = data.get_mean(test_x1);
		mean_t2 = data.get_mean(test_x2);
		mean_t3 = data.get_mean(test_x3);
		uranus::Vector<dim> DD1_mean = eigen_W.transpose()*mean_t1,
			DD2_mean = eigen_W.transpose()*mean_t2,
			DD3_mean = eigen_W.transpose()*mean_t3;

		// 投影后的各元素
		for (int i = 0; i < train_x1.size(); ++i) train_pt1[i] = eigen_W.transpose()*train_x1[i];
		for (int i = 0; i < train_x2.size(); ++i) train_pt2[i] = eigen_W.transpose()*train_x2[i];
		for (int i = 0; i < train_x3.size(); ++i) train_pt3[i] = eigen_W.transpose()*train_x3[i];
		for (int i = 0; i < test_x1.size(); ++i) test_pt1[i] = eigen_W.transpose()*test_x1[i];
		for (int i = 0; i < test_x2.size(); ++i) test_pt2[i] = eigen_W.transpose()*test_x2[i];
		for (int i = 0; i < test_x3.size(); ++i) test_pt3[i] = eigen_W.transpose()*test_x3[i];
#ifdef SHOW_DATA
		cout << "\n"; for (int i = 0; i < test_x1.size(); ++i) cout << DD1[i](0) << "\n";
		cout << "\n"; for (int i = 0; i < test_x1.size(); ++i) cout << DD1[i](1) << "\n";
		cout << "\n"; for (int i = 0; i < test_x2.size(); ++i) cout << DD2[i](0) << "\n";
		cout << "\n"; for (int i = 0; i < test_x2.size(); ++i) cout << DD2[i](1) << "\n";
		cout << "\n"; for (int i = 0; i < test_x3.size(); ++i) cout << DD3[i](0) << "\n";
		cout << "\n"; for (int i = 0; i < test_x3.size(); ++i) cout << DD3[i](1) << "\n";
#endif
#pragma omp parallel
		{
#ifdef SHOW_TRAIN_SET
#pragma omp for
			for (int i = 0; i < train_x1.size(); ++i) {
				cv::Point2f pt;
				pt.x = zero_pt.x + train_pt1[i](0) * 4 * draw_set_scale;
				pt.y = zero_pt.y - train_pt1[i](1) * 3 * draw_set_scale;
				cv::circle(draw_test_set, pt, 2, cv::Scalar(155, 0, 0), 2);
			}
#pragma omp for
			for (int i = 0; i < train_x2.size(); ++i) {
				cv::Point2f pt;
				pt.x = zero_pt.x + train_pt2[i](0) * 4 * draw_set_scale;
				pt.y = zero_pt.y - train_pt2[i](1) * 3 * draw_set_scale;
				cv::circle(draw_test_set, pt, 2, cv::Scalar(0, 155, 0), 2);
			}
#pragma omp for
			for (int i = 0; i < train_x3.size(); ++i) {
				cv::Point2f pt;
				pt.x = zero_pt.x + train_pt3[i](0) * 4 * draw_set_scale;
				pt.y = zero_pt.y - train_pt3[i](1) * 3 * draw_set_scale;
				cv::circle(draw_test_set, pt, 2, cv::Scalar(0, 0, 155), 2);
			}
#endif // SHOW_TRAIN_SET
#ifdef SHOW_TEST_SET
#pragma omp for
			for (int i = 0; i < test_x1.size(); ++i) {
				cv::Point2f pt;
				pt.x = zero_pt.x + test_pt1[i](0) * 4 * draw_set_scale;
				pt.y = zero_pt.y - test_pt1[i](1) * 3 * draw_set_scale;
				cv::circle(draw_test_set, pt, 2, cv::Scalar(255, 0, 0), 2);
			}
#pragma omp for
			for (int i = 0; i < test_x2.size(); ++i) {
				cv::Point2f pt;
				pt.x = zero_pt.x + test_pt2[i](0) * 4 * draw_set_scale;
				pt.y = zero_pt.y - test_pt2[i](1) * 3 * draw_set_scale;
				cv::circle(draw_test_set, pt, 2, cv::Scalar(0, 255, 0), 2);
			}
#pragma omp for
			for (int i = 0; i < test_x3.size(); ++i) {
				cv::Point2f pt;
				pt.x = zero_pt.x + test_pt3[i](0) * 4 * draw_set_scale;
				pt.y = zero_pt.y - test_pt3[i](1) * 3 * draw_set_scale;
				cv::circle(draw_test_set, pt, 2, cv::Scalar(0, 0, 255), 2);
			}
#endif // SHOW_TEST_SET
		}

		cv::imshow("draw_set : B1 G2 R3", draw_test_set);
		// init
		uranus::SquareMatrix<dim> plane_Si_1, plane_Si_2, plane_Si_3;
		// init
		uranus::setZero< uranus::SquareMatrix<dim>, dim>(plane_Si_1);
		uranus::setZero< uranus::SquareMatrix<dim>, dim>(plane_Si_2);
		uranus::setZero< uranus::SquareMatrix<dim>, dim>(plane_Si_3);
#pragma omp parallel
		{
#pragma omp for
			for (int i = 0; i < test_pt1.size(); ++i)
				plane_Si_1 += (test_pt1[i] - DD1_mean)*(test_pt1[i] - DD1_mean).transpose();
			//cout << "Si_1=\n" << Si_1 << endl << endl;
#pragma omp for
			for (int i = 0; i < test_pt2.size(); ++i)
				plane_Si_2 += (test_pt2[i] - DD2_mean)*(test_pt2[i] - DD2_mean).transpose();
			//cout << "Si_2=\n" << Si_2 << endl << endl;
#pragma omp for
			for (int i = 0; i < test_pt3.size(); ++i)
				plane_Si_3 += (test_pt3[i] - DD3_mean)*(test_pt3[i] - DD3_mean).transpose();
			//cout << "Si_2=\n" << Si_2 << endl << endl;
		}
		uranus::SquareMatrix<dim> plane_Sw = plane_Si_1 + plane_Si_2 + plane_Si_3;
		const double pc1 = test_pt1.size() / (test_pt1.size() + test_pt2.size() + test_pt3.size());
		const double pc2 = test_pt2.size() / (test_pt1.size() + test_pt2.size() + test_pt3.size());
		const double pc3 = test_pt3.size() / (test_pt1.size() + test_pt2.size() + test_pt3.size());
		uranus::SquareMatrix<dim> plane_Sw_Bayes = pc1 * plane_Si_1 + pc2 * plane_Si_2 + pc3 * plane_Si_3;  // 贝叶斯
		double plance_SW_cond = (plane_Sw.norm())*(plane_Sw.inverse().norm());
		cout << "\ncond number: " << plance_SW_cond << endl;

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
		double rate;
		for (int i = 0; i < test_pt1.size(); ++i)
		{
			bool res = Multi_Discriminant<dim>(plane_argW_12, plane_argW_13, plane_W0_12, plane_W0_13, test_pt1[i]);
			if (res)p++;
			else n++;
		}
		rate = p > n ? p / test_pt1.size() : n / test_pt1.size();
		cout << "\n 1准确率：" << rate * 100 << "%\t p = " << p << "\t n = " << n << "\t size = " << test_pt1.size() << "\n";

		p = 0.0, n = 0.0;
		for (int i = 0; i < test_pt2.size(); ++i)
		{
			bool res = Multi_Discriminant<dim>(plane_argW_12, plane_argW_23, plane_W0_12, plane_W0_23, test_pt2[i]);
			if (res)p++;
			else n++;
		}
		rate = p > n ? p / test_pt1.size() : n / test_pt1.size();
		cout << "\n 2准确率：" << rate * 100 << "%\t p = " << p << "\t n = " << n << "\t size = " << test_pt2.size() << "\n";

		p = 0.0, n = 0.0;
		for (int i = 0; i < test_pt3.size(); ++i)
		{
			bool res = Multi_Discriminant<dim>(plane_argW_13, plane_argW_23, plane_W0_13, plane_W0_23, test_pt3[i]);
			if (res)p++;
			else n++;
		}
		rate = p > n ? p / test_pt1.size() : n / test_pt1.size();
		cout << "\n 3准确率：" << rate * 100 << "%\t p = " << p << "\t n = " << n << "\t size = " << test_pt3.size() << "\n";
		char key = cv::waitKey(0);
		if (key == 'q') break;
	}

	cv::waitKey(0);

	return EXIT_SUCCESS;
}