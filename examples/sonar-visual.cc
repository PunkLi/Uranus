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

constexpr double E = 2.718282;

constexpr int feature_rows = 60;
constexpr int Dim = feature_rows;
constexpr int dim = 1;
std::vector<int> data_class = { 97,111 };

std::string path = "../data/sonar.all-data";

std::vector<uranus::Vector<feature_rows>> mean;
uranus::SquareMatrix<feature_rows> Si_1;
uranus::SquareMatrix<feature_rows> Si_2;
uranus::SquareMatrix<feature_rows> Sw;


int main(int argc, char *argv[])
{
	constexpr int step_size = 30;
	constexpr int draw_cond_x_scale = 640 / step_size;
	// init
	using namespace std;
	using sample_set = uranus::Tensor<feature_rows>::sample_set;
	using tensor = uranus::Tensor<feature_rows>::TensorType;

	uranus::Data_Wrapper<feature_rows> wrapper(path, data_class);
	uranus::Tensor<feature_rows> data(wrapper, data_class);

	tensor tensor_x1, tensor_x2;

	uranus::SquareMatrix<Dim> Sw;
	uranus::SquareMatrix<Dim> Sw_Bayes;
	uranus::SquareMatrix<Dim> Sb;              // 类间离散度矩阵
	uranus::SquareMatrix<Dim> Sb_Bayes;

	sample_set train_x1, train_x2, test_x1, test_x2;

	uranus::Vector<Dim> mean_total, mean_1, mean_2;
	uranus::Vector<Dim> mean_t1;// = data.get_mean(test_x1);
	uranus::Vector<Dim> mean_t2;// = data.get_mean(test_x2);

	cv::Mat draw_null = cv::Mat(960, 1280, CV_8UC3);
	cv::Mat draw_dim1 = cv::Mat(960, 1280, CV_8UC3);
	cv::Mat draw_dim2 = cv::Mat(960, 1280, CV_8UC3);
	cv::Mat draw_rate = cv::Mat(480, 640, CV_8UC3);

	cv::line(draw_rate, cv::Point(0, 40), cv::Point(640, 40), cv::Scalar(255, 0, 0), 1.5);       // 100
	cv::line(draw_rate, cv::Point(0, 80), cv::Point(640, 80), cv::Scalar(255, 255, 255), 1.5);   // 90
	cv::line(draw_rate, cv::Point(0, 120), cv::Point(640, 120), cv::Scalar(255, 255, 255), 1.5); // 80
	cv::line(draw_rate, cv::Point(0, 160), cv::Point(640, 160), cv::Scalar(255, 255, 255), 1.5); // 70
	cv::line(draw_rate, cv::Point(0, 200), cv::Point(640, 200), cv::Scalar(255, 255, 255), 1.5); // 60
	cv::line(draw_rate, cv::Point(0, 240), cv::Point(640, 240), cv::Scalar(100, 255, 255), 1.5); // 50
	cv::line(draw_rate, cv::Point(0, 280), cv::Point(640, 280), cv::Scalar(255, 255, 255), 1.5); // 40
	cv::line(draw_rate, cv::Point(0, 320), cv::Point(640, 320), cv::Scalar(255, 255, 255), 1.5); // 30
	cv::line(draw_rate, cv::Point(0, 360), cv::Point(640, 360), cv::Scalar(255, 255, 255), 1.5); // 20
	cv::line(draw_rate, cv::Point(0, 400), cv::Point(640, 400), cv::Scalar(255, 255, 255), 1.5); // 10
	cv::line(draw_rate, cv::Point(0, 440), cv::Point(640, 440), cv::Scalar(255, 0, 0), 2);       // 0


	cv::Point2f zero_pt;
	zero_pt.x = 640;
	zero_pt.y = 480;
	constexpr int draw_set_scale = 1000;
	cv::line(draw_null, cv::Point(0, zero_pt.y), cv::Point(1280, zero_pt.y), cv::Scalar(255, 255, 255), 2); // x
	cv::line(draw_null, cv::Point(zero_pt.x, 0), cv::Point(zero_pt.x, 960), cv::Scalar(255, 255, 255), 2); // y

	Eigen::Matrix<double, Dim, dim> eigen_W;   // 特征值投影
	
	cv::Point2f norm_pt_arr[step_size];
	cv::Point2f cond_pt_arr[step_size];

	double balance = 0;

	// draw rate
	cv::Point2f norm_norm_arr1[step_size];
	cv::Point2f norm_bayes_arr1[step_size];
	cv::Point2f bayes_norm_arr1[step_size];
	cv::Point2f bayes_bayes_arr1[step_size];
	cv::Point2f norm_norm_arr2[step_size];
	cv::Point2f norm_bayes_arr2[step_size];
	cv::Point2f bayes_norm_arr2[step_size];
	cv::Point2f bayes_bayes_arr2[step_size];
	// trian
	for (int step = 0; step < step_size; ++step)
	{
		draw_null.copyTo(draw_dim1);
		draw_null.copyTo(draw_dim2);

		tensor_x1 = data.k_fold_crossValidation<2>(0);
		tensor_x2 = data.k_fold_crossValidation<2>(1);

		train_x1 = tensor_x1[0];
		train_x2 = tensor_x2[0];

		// step1 均值向量
		mean_1 = data.get_mean(train_x1);
		mean_2 = data.get_mean(train_x2);

		mean_total = mean_1 / 2 + mean_2 / 2;  // train mean

		// step2 类内离散度矩阵
		uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_1);
		uranus::setZero<uranus::SquareMatrix<feature_rows>, feature_rows>(Si_2);
#pragma omp parallel
		{
#pragma omp for
			for (int i = 0; i < train_x1.size(); ++i)
				Si_1 += (train_x1[i] - mean_1)*(train_x1[i] - mean_1).transpose();
			//cout << "Si_1=\n" << Si_1 << endl << endl;
#pragma omp for
			for (int i = 0; i < train_x2.size(); ++i)
				Si_2 += (train_x2[i] - mean_2)*(train_x2[i] - mean_2).transpose();
			//cout << "Si_2=\n" << Si_2 << endl << endl;
		}
		Sw = Si_1 + Si_2;     // 总样本类内离散度矩阵Sw  对称半正定矩阵，而且当n>d时通常是非奇异的
		//cout << "Sw=\n" << Sw << endl << endl;
		const double train_frac = train_x1.size() + train_x2.size();
		const double c1 = train_x1.size() / train_frac;
		const double c2 = train_x2.size() / train_frac;
		Sw_Bayes = c1 * Si_1 + c2 * Si_2; // 带贝叶斯的 Sw
		//cout << "Sw=\n" << Sw << endl << endl;

		double cond = (Sw.norm())*(Sw.inverse().norm());  // 类内离散度矩阵条件数

		double _b = abs(Sw.norm() - cond);
		//if (_b > balance)
		//	balance = _b;

		cout << "norm: " << Sw.norm() << "\t cond number: " << cond << endl;

		// visual
		/*
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
		*/
		Sb = (mean_1 - mean_total)*(mean_1 - mean_total).transpose() * 1 / 2
			+ (mean_2 - mean_total)*(mean_2 - mean_total).transpose() * 1 / 2;

		Sb_Bayes = (mean_1 - mean_total)*(mean_1 - mean_total).transpose() * c1
			+ (mean_2 - mean_total)*(mean_2 - mean_total).transpose() * c2;

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

		// d维度
		Eigen::Matrix<std::complex<double>, Dim, dim> eigen_Wc; // 复数投影矩阵

		for (int i = 0; i < dim; ++i) eigen_Wc.col(i) = eigenvect.col(eigenval[i].first);

		eigen_W = eigen_Wc.real();								 // 实数投影矩阵

	
		/*
		// 再来一次Fisher
		mean_t1 = data.get_mean(test_x1);
		mean_t2 = data.get_mean(test_x2);
		uranus::Vector<dim> DD1_mean = eigen_W.transpose()*mean_t1,
			DD2_mean = eigen_W.transpose()*mean_t2;

		// 投影后的各元素
		for (int i = 0; i < train_x1.size(); ++i) train_pt1[i] = eigen_W.transpose()*train_x1[i];
		for (int i = 0; i < train_x2.size(); ++i) train_pt2[i] = eigen_W.transpose()*train_x2[i];

		for (int i = 0; i < test_x1.size(); ++i) test_pt1[i] = eigen_W.transpose()*test_x1[i];
		for (int i = 0; i < test_x2.size(); ++i) test_pt2[i] = eigen_W.transpose()*test_x2[i];

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
				cv::circle(draw_test_set, pt, 2, cv::Scalar(0, 0, 255), 2);
			}
#endif // SHOW_TEST_SET
		}

		cv::imshow("draw_set : B1 G2 R3", draw_test_set);
		char key = cv::waitKey(0);
		if (key == 'q') break;

		// init
		uranus::SquareMatrix<dim> plane_Si_1, plane_Si_2, plane_Si_3;
		// init
		uranus::setZero< uranus::SquareMatrix<dim>, dim>(plane_Si_1);
		uranus::setZero< uranus::SquareMatrix<dim>, dim>(plane_Si_2);
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
		}
		uranus::SquareMatrix<dim> plane_Sw = plane_Si_1 + plane_Si_2 + plane_Si_3;
		const double pc1 = test_pt1.size() / (test_pt1.size() + test_pt2.size());
		const double pc2 = test_pt2.size() / (test_pt1.size() + test_pt2.size());

		uranus::SquareMatrix<dim> plane_Sw_Bayes = pc1 * plane_Si_1 + pc2 * plane_Si_2;  // 贝叶斯

		double plance_SW_cond = (plane_Sw.norm())*(plane_Sw.inverse().norm());
		cout << "\ncond number: " << plance_SW_cond << endl;

		uranus::Vector<dim> plane_argW = plane_Sw.inverse()*(DD1_mean - DD2_mean);

		uranus::Vector<1> plane_W0 = plane_argW.transpose()*(DD1_mean) / 2
			+ plane_argW.transpose()*(DD2_mean) / 2;                            // 考虑一下贝叶斯

		*/

		double In_omega = train_x2.size() / train_x1.size();
		double Inw = log(In_omega) / log(E);
		uranus::Vector<1> _InW_;
		_InW_ << Inw;

		// 方法一
		uranus::Vector<Dim> argW       = Sw.inverse()*(mean_1 - mean_2);        // 贝叶斯
		uranus::Vector<Dim> argW_bayes = Sw_Bayes.inverse()*(mean_1 - mean_2);  // 贝叶斯

		// step6求阈值 W0 
		// 传统判别
		uranus::Vector<1> Wo = argW.transpose()*(mean_1) / 2.0 + argW.transpose()*(mean_2) / 2.0;
		cout << "Wo = " << Wo << endl;

		uranus::Vector<1> Wo_Bayes = (mean_1 + mean_2).transpose()*Sw.inverse()*(mean_1 - mean_2) - _InW_; // 判别准则Wo
		cout << "Bayes W0 = " << Wo_Bayes << endl;
		uranus::Vector<1> Wo_BBayes = (mean_1 + mean_2).transpose()*Sw_Bayes.inverse()*(mean_1 - mean_2) - _InW_; // 判别准则Wo
		cout << "BBayes W0 = " << Wo_Bayes << endl;

		test_x1 = tensor_x1[1];
		test_x2 = tensor_x2[1];
		std::vector<uranus::Vector<1>> D1(test_x1.size());
		std::vector<uranus::Vector<1>> D2(test_x2.size());

		// 传统投影到一维
		for (int i = 0; i < test_x1.size(); ++i) D1[i] = argW.transpose()*test_x1[i];
		for (int i = 0; i < test_x2.size(); ++i) D2[i] = argW.transpose()*test_x2[i];

		for (int i = 0; i < test_x1.size(); ++i) {
			cv::Point2f pt;
			pt.x = zero_pt.x + D1[i](0) *4 * 100;
			pt.y = zero_pt.y - 400;
			cv::circle(draw_dim1, pt, 2, cv::Scalar(0, 0, 100), 2);
		}
		for (int i = 0; i < test_x2.size(); ++i) {
			cv::Point2f pt;
			pt.x = zero_pt.x + D2[i](0) *4 * 100;
			pt.y = zero_pt.y - 350;
			cv::circle(draw_dim1, pt, 2, cv::Scalar(100, 0, 0), 2);
		}
		cout << "\n传统投影 传统判别:\n";
		double norm_norm_rate1 = Evaluation(Wo, D1);
		Evaluation(Wo, D2);
		
		cv::line(draw_dim1, 
			cv::Point(zero_pt.x + Wo(0) * 4 * 100, zero_pt.y - 500),
			cv::Point(zero_pt.x + Wo(0) * 4 * 100, zero_pt.y - 300), 
			cv::Scalar(0, 155, 0), 2);
		
		cout << "\n传统投影 贝叶斯判别:\n";
		Evaluation(Wo_Bayes, D1);
		Evaluation(Wo_Bayes, D2);
		cout << "\n---------------------------------\n";

		// 贝叶斯投影到一维
		for (int i = 0; i < test_x1.size(); ++i) D1[i] = argW_bayes.transpose()*test_x1[i];
		for (int i = 0; i < test_x2.size(); ++i) D2[i] = argW_bayes.transpose()*test_x2[i];

		for (int i = 0; i < test_x1.size(); ++i) {
			cv::Point2f pt;
			pt.x = zero_pt.x + D1[i](0) * 4 * 100;
			pt.y = zero_pt.y - 250;
			cv::circle(draw_dim1, pt, 2, cv::Scalar(0, 0, 195), 2);
		}
		for (int i = 0; i < test_x2.size(); ++i) {
			cv::Point2f pt;
			pt.x = zero_pt.x + D2[i](0) * 4 * 100;
			pt.y = zero_pt.y - 200;
			cv::circle(draw_dim1, pt, 2, cv::Scalar(195, 0, 0), 2);
		}

		cout << "\n贝叶斯投影 传统判别:\n";
		Evaluation(Wo, D1);
		Evaluation(Wo, D2);
	
		cout << "\n贝叶斯投影 贝叶斯判别:\n";
		double bayes_norm_rate1 = Evaluation(Wo_Bayes, D1);
		Evaluation(Wo_Bayes, D2);

		cv::line(draw_dim1,
			cv::Point(zero_pt.x + Wo_Bayes(0) * 4 * 100, zero_pt.y - 300),
			cv::Point(zero_pt.x + Wo_Bayes(0) * 4 * 100, zero_pt.y - 150),
			cv::Scalar(0, 155, 0), 2);

		cout << "\n---------------------------------\n";

		// 特征值投影到一维
		uranus::Vector<1> Wo_eigen = eigen_W.transpose()*(mean_1)*c1 + argW.transpose()*(mean_2)*c2;

		for (int i = 0; i < test_x1.size(); ++i) D1[i] = eigen_W.transpose()*test_x1[i];
		for (int i = 0; i < test_x2.size(); ++i) D2[i] = eigen_W.transpose()*test_x2[i];

		for (int i = 0; i < test_x1.size(); ++i) {
			cv::Point2f pt;
			pt.x = zero_pt.x + D1[i](0) * 4 * 400;
			pt.y = zero_pt.y - 100;
			cv::circle(draw_dim1, pt, 2, cv::Scalar(0, 0, 255), 2);
		}
		for (int i = 0; i < test_x2.size(); ++i) {
			cv::Point2f pt;
			pt.x = zero_pt.x + D2[i](0) * 4 * 1000;
			pt.y = zero_pt.y - 50;
			cv::circle(draw_dim1, pt, 2, cv::Scalar(255, 0, 0), 2);
		}
		cv::line(draw_dim1,
			cv::Point(zero_pt.x + Wo_eigen(0) * 4 * 1000, zero_pt.y - 150),
			cv::Point(zero_pt.x + Wo_eigen(0) * 4 * 1000, zero_pt.y),
			cv::Scalar(0, 155, 0), 2);

// 往二维投影
		Eigen::Matrix<std::complex<double>, Dim, 2> eigen_Wc2; // 复数投影矩阵
		for (int i = 0; i < 2; ++i) eigen_Wc2.col(i) = eigenvect.col(eigenval[i].first);
		Eigen::Matrix<double, Dim, 2> eigen_W2 = eigen_Wc2.real();						      // 实数投影矩阵
		 
		std::vector<uranus::Vector<2>> plane_x1(test_x1.size());
		std::vector<uranus::Vector<2>> plane_x2(test_x2.size());
		for (int i = 0; i < test_x1.size(); ++i) plane_x1[i] = eigen_W2.transpose()*test_x1[i];
		for (int i = 0; i < test_x2.size(); ++i) plane_x2[i] = eigen_W2.transpose()*test_x2[i];

		for (int i = 0; i < test_x1.size(); ++i) {
			cv::Point2f pt;
			pt.x = zero_pt.x + plane_x1[i](0) * 4 * draw_set_scale;
			pt.y = zero_pt.y - plane_x1[i](1) * 3 * draw_set_scale;
			cv::circle(draw_dim2, pt, 2, cv::Scalar(0, 0, 255), 2);
		}
		for (int i = 0; i < test_x2.size(); ++i) {
			cv::Point2f pt;
			pt.x = zero_pt.x + plane_x2[i](0) * 4 *draw_set_scale;
			pt.y = zero_pt.y - plane_x2[i](1) * 3 *draw_set_scale;
			cv::circle(draw_dim2, pt, 2, cv::Scalar(255, 0, 0), 2);
		}

		//uranus::Vector<2> Wo_eigen2 = eigen_W2.transpose()*(mean_1)*c1 + argW.transpose()*(mean_2)*c2;
		//cv::circle(draw_dim2, 
		//	cv::Point(Wo_eigen2(0) * 4 * draw_set_scale, Wo_eigen2(1) * 3 * draw_set_scale),
		//	2, cv::Scalar(255, 0, 0), 2);

		// visual
		cv::Point2f norm_norm_pt1;
		norm_norm_pt1.x = step * draw_cond_x_scale;
		norm_norm_pt1.y = (440 - norm_norm_rate1 * 400);
		norm_norm_arr1[step] = norm_norm_pt1;
		cv::circle(draw_rate, norm_norm_pt1, 2, cv::Scalar(0, 0, 255), 2);
		if (step) cv::line(draw_rate, norm_norm_arr1[step], norm_norm_arr1[step - 1], cv::Scalar(0, 0, 255), 1.5);

		cv::Point2f bayes_norm_pt1;
		bayes_norm_pt1.x = step * draw_cond_x_scale;
		bayes_norm_pt1.y = (440 - bayes_norm_rate1 * 400);
		bayes_norm_arr1[step] = bayes_norm_pt1;
		cv::circle(draw_rate, bayes_norm_pt1, 2, cv::Scalar(255, 0, 0), 2);
		if (step) cv::line(draw_rate, bayes_norm_arr1[step], bayes_norm_arr1[step - 1], cv::Scalar(255, 0, 0), 1.5);

		cv::imshow("dim 1", draw_dim1);
		cv::imshow("dim 2", draw_dim2);
		cv::imshow("rate", draw_rate);

		char key = cv::waitKey(0);
		if (key == 'q') break;
	}
	cv::waitKey(0);
	return EXIT_SUCCESS;
}