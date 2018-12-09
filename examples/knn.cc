/**
 * 模式识别 Knn
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"

#define Iris

#ifdef Iris
constexpr int Dim = 4;
std::string path = "../data/iris.data";
std::vector<int> data_class = { 50,50,50 };
#else
constexpr int Dim = 60;
std::string path = "../data/sonar.all-data";
std::vector<int> data_class = { 97,111 };
#endif

using sample = uranus::Tensor<Dim>::sampleType;
using sample_set = uranus::Tensor<Dim>::sample_set;
using tensor = uranus::Tensor<Dim>::TensorType;
using CP = std::pair<sample, int>;
using KC = std::pair<double, int >;

constexpr int K_knn = 10; // knn

std::vector<int> knn_solver(std::vector<CP> vec_, sample_set& test_x, bool visual = false);
double Evaluation(std::vector<int> vec, int except);

int main(int argc, char *argv[])
{
	uranus::Data_Wrapper<Dim> wrapper(path, data_class);
	uranus::Tensor<Dim> data(wrapper, data_class);

	tensor tensor_x1 = data.k_fold_crossValidation<2>(0);
	tensor tensor_x2 = data.k_fold_crossValidation<2>(1);

	sample_set train_x1, train_x2, train_x3, test_x1, test_x2, test_x3;
	train_x1 = tensor_x1[0];
	train_x2 = tensor_x2[0];

	test_x1 = tensor_x1[1];
	test_x2 = tensor_x2[1];

	std::vector<CP> vec_;
	for (int i = 0; i < train_x1.size(); ++i) vec_.push_back(std::make_pair(train_x1[i], 1));
	for (int i = 0; i < train_x2.size(); ++i) vec_.push_back(std::make_pair(train_x2[i], 2));
#ifdef Iris	
	tensor tensor_x3 = data.k_fold_crossValidation<2>(2);
	train_x3 = tensor_x3[0];
	test_x3 = tensor_x3[1];
	for (int i = 0; i < train_x3.size(); ++i) vec_.push_back(std::make_pair(train_x3[i], 3));
#endif
	std::vector<int> sorce1 = knn_solver(vec_, test_x1);
	std::vector<int> sorce2 = knn_solver(vec_, test_x2);

	double c1 = Evaluation(sorce1, 1);
	double c2 = Evaluation(sorce2, 2);

	std::cout << "rate1:" << c1 << "\n";
	std::cout << "rate2:" << c2 << "\n";

#ifdef Iris
	std::vector<int> sorce3 = knn_solver(vec_, test_x3);
	double c3 = Evaluation(sorce3, 3);
	std::cout << "rate3:" << c3 << "\n";
#endif

	return EXIT_SUCCESS;
}

std::vector<int> knn_solver(std::vector<CP> vec_, sample_set& test_x, bool visual)
{
	std::vector<int> sorce_(3);

	for (size_t idx = 0; idx < test_x.size(); ++idx) // 对 test 1 的每个样本
	{
		std::vector<KC> vec_dis;
		sample x = test_x[idx];
		for (size_t i = 0; i < vec_.size(); ++i) // 计算全部 train 的距离
		{
			double res = uranus::Norm<sample, Dim>(x, vec_[i].first);
			vec_dis.push_back(std::make_pair(res, vec_[i].second));
		}
		// 距离排序
		std::sort(vec_dis.begin(), vec_dis.end(),
			[](const KC& lhs, const KC& rhs)
		{
			return lhs.first < rhs.first;
		});
#ifdef Iris
		std::vector<int> sorce(3);
		sorce[0] = 0;
		sorce[1] = 0;
		sorce[2] = 0;
#else
		std::vector<int> sorce(2);
		sorce[0] = 0;
		sorce[1] = 0;
#endif
		// 选前k个
		for (size_t i = 0; i < K_knn; ++i)
		{
			int class_ = vec_dis[i].second; // which

			if (class_ == 1) sorce[0]++;
			else if (class_ == 2) sorce[1]++;
#ifdef Iris
			else sorce[2]++;
#endif
		}
#ifdef Iris
		if (sorce[0] > sorce[1] && sorce[0] > sorce[2])
		{
			sorce_[0]++;
			if (visual) std::cout << "belong class 1 \n";
		}
		if (sorce[1] > sorce[0] && sorce[1] > sorce[2])
		{
			sorce_[1]++;
			if (visual) std::cout << "belong class 2 \n";
		}
		if (sorce[2] > sorce[0] && sorce[2] > sorce[1])
		{
			sorce_[2]++;
			if (visual) std::cout << "belong class 3 \n";
		}
#else
		if (sorce[0] > sorce[1])
		{
			sorce_[0]++;
			if (visual) std::cout << "belong class 1 \n";
		}
		if (sorce[1] > sorce[0])
		{
			sorce_[1]++;
			if (visual) std::cout << "belong class 2 \n";
		}
#endif
	}
	return sorce_;
}

double Evaluation(std::vector<int> vec, int except)
{
	double total = 0.0;
	for (size_t idx = 0; idx < vec.size(); ++idx)
	{
		total += vec[idx];
	}
	double rate = vec[except - 1] / total;
	return rate;
}