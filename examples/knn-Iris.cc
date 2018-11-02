/**
 * 模式识别 Knn
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"

constexpr int feature_rows = 4;
constexpr int Dim = feature_rows;

std::vector<int> data_class = { 50,50,50 };

std::string path = "../data/iris.data";

constexpr int K_knn = 10; // knn

int main(int argc, char *argv[])
{
	using namespace std;
	using sample = uranus::Tensor<Dim>::sampleType;
	using sample_set = uranus::Tensor<Dim>::sample_set;
	using tensor = uranus::Tensor<Dim>::TensorType;

	uranus::Data_Wrapper<Dim> wrapper(path, data_class);
	uranus::Tensor<Dim> data(wrapper, data_class);

	tensor tensor_x1 = data.k_fold_crossValidation<2>(0);
	tensor tensor_x2 = data.k_fold_crossValidation<2>(1);
	tensor tensor_x3 = data.k_fold_crossValidation<2>(2);

	sample_set train_x1, train_x2, train_x3, test_x1, test_x2, test_x3;
	train_x1 = tensor_x1[0];
	train_x2 = tensor_x2[0];
	train_x3 = tensor_x3[0];

	test_x1 = tensor_x1[1];
	test_x2 = tensor_x2[1];
	test_x3 = tensor_x3[1];

	using CP = pair<sample, int>;
	using KC = pair<double, int >;
	std::vector<CP> vec_;
	std::vector<KC> vec_dis;

	for (int i = 0; i < train_x1.size(); ++i) vec_.push_back(make_pair(train_x1[i], 1));
	for (int i = 0; i < train_x2.size(); ++i) vec_.push_back(make_pair(train_x2[i], 2));
	for (int i = 0; i < train_x3.size(); ++i) vec_.push_back(make_pair(train_x3[i], 3));

	for (size_t idx = 0; idx < test_x1.size(); ++idx) // 对 test 1 的每个样本
	{
		sample x = test_x1[idx];
		vec_dis.clear();
		for (size_t i = 0; i < vec_.size(); ++i) // 计算全部 train 的距离
		{
			double res = uranus::Norm<sample, Dim>(x, vec_[i].first);
			cout << res << "\n";
			vec_dis.push_back(make_pair(res, vec_[i].second));
		}
		// 距离排序
		std::sort(vec_dis.begin(), vec_dis.end(), 
			[](const KC& lhs,const KC& rhs) 
		{
			return lhs.first < rhs.first;
		});
		std::vector<int> sorce(3);
		sorce[0] = 0;
		sorce[1] = 0;
		sorce[2] = 0;
		// 选前k个
		for (size_t i = 0; i < K_knn; ++i)
		{
			int class_ = vec_dis[i].second; // which

			if (class_ == 1) sorce[0]++;
			else if (class_ == 2) sorce[1]++;
			else sorce[2]++;
		}
		if (sorce[0] > sorce[1] && sorce[0] > sorce[2]) std::cout << "belong class 1 \n";
		if (sorce[1] > sorce[0] && sorce[1] > sorce[2]) std::cout << "belong class 2 \n";
		if (sorce[2] > sorce[0] && sorce[2] > sorce[1]) std::cout << "belong class 3 \n";
	}


	for (size_t idx = 0; idx < test_x2.size(); ++idx) // 对 test 2 的每个样本
	{
		sample x = test_x2[idx];
		vec_dis.clear();
		for (size_t i = 0; i < vec_.size(); ++i) // 计算全部 train 的距离
		{
			double res = uranus::Norm<sample, Dim>(x, vec_[i].first);
			vec_dis.push_back(make_pair(res, vec_[i].second));
		}
		// 距离排序
		std::sort(vec_dis.begin(), vec_dis.end(),
			[](const KC& lhs, const KC& rhs)
		{
			return lhs.first < rhs.first;
		});
		std::vector<int> sorce(3);
		sorce[0] = 0;
		sorce[1] = 0;
		sorce[2] = 0;
		// 选前k个
		for (size_t i = 0; i < K_knn; ++i)
		{
			int class_ = vec_dis[i].second; // which

			if (class_ == 1) sorce[0]++;
			else if (class_ == 2) sorce[1]++;
			else sorce[2]++;
		}
		if (sorce[0] > sorce[1] && sorce[0] > sorce[2]) std::cout << "belong class 1 \n";
		if (sorce[1] > sorce[0] && sorce[1] > sorce[2]) std::cout << "belong class 2 \n";
		if (sorce[2] > sorce[0] && sorce[2] > sorce[1]) std::cout << "belong class 3 \n";
	}

	for (size_t idx = 0; idx < test_x3.size(); ++idx) // 对 test 2 的每个样本
	{
		sample x = test_x3[idx];
		vec_dis.clear();
		for (size_t i = 0; i < vec_.size(); ++i) // 计算全部 train 的距离
		{
			double res = uranus::Norm<sample, Dim>(x, vec_[i].first);
			vec_dis.push_back(make_pair(res, vec_[i].second));
		}
		// 距离排序
		std::sort(vec_dis.begin(), vec_dis.end(),
			[](const KC& lhs, const KC& rhs)
		{
			return lhs.first < rhs.first;
		});
		std::vector<int> sorce(3);
		sorce[0] = 0;
		sorce[1] = 0;
		sorce[2] = 0;
		// 选前k个
		for (size_t i = 0; i < K_knn; ++i)
		{
			int class_ = vec_dis[i].second; // which

			if (class_ == 1) sorce[0]++;
			else if (class_ == 2) sorce[1]++;
			else sorce[2]++;
		}
		if (sorce[0] > sorce[1] && sorce[0] > sorce[2]) std::cout << "belong class 1 \n";
		if (sorce[1] > sorce[0] && sorce[1] > sorce[2]) std::cout << "belong class 2 \n";
		if (sorce[2] > sorce[0] && sorce[2] > sorce[1]) std::cout << "belong class 3 \n";
	}
	return EXIT_SUCCESS;
}
