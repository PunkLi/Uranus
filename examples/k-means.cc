/**
 * 模式识别 Knn
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */

#include <iostream>
#include <vector>
#include "uranus/Tensor.hpp"

int uniform_intx(int a, int b)
{
	static std::default_random_engine e{std::random_device{}()};
	static std::uniform_int_distribution<int> u;
	return u(e, std::uniform_int_distribution<int>::param_type(a, b));
}

//#define Iris

#ifdef Iris
constexpr int Dim = 4;
std::string path = "../data/iris.data";
std::vector<int> data_class = {50, 50, 50};
#else
constexpr int Dim = 60;
std::string path = "../data/sonar.all-data";
std::vector<int> data_class = {97, 111};
#endif

using sample = uranus::Tensor<Dim>::sampleType;
using sample_set = uranus::Tensor<Dim>::sample_set;
using tensor = uranus::Tensor<Dim>::TensorType;
using CP = std::pair<sample, int>;

void kmeans_solver(std::vector<CP> vec_, bool visual = false);
double Evaluation(sample_set test_set, int except_C);

sample_set C1; // class1
sample_set C2; // class2
sample_set C3; // class3

sample C1_mean;
sample C2_mean;
sample C3_mean;

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

	// 产生带标签的数据
	std::vector<CP> vec_;
	for (int i = 0; i < train_x1.size(); ++i)
		vec_.push_back(std::make_pair(train_x1[i], 1));
	for (int i = 0; i < train_x2.size(); ++i)
		vec_.push_back(std::make_pair(train_x2[i], 2));
#ifdef Iris
	tensor tensor_x3 = data.k_fold_crossValidation<2>(2);
	train_x3 = tensor_x3[0];
	test_x3 = tensor_x3[1];
	for (int i = 0; i < train_x3.size(); ++i)
		vec_.push_back(std::make_pair(train_x3[i], 3));
#endif

	kmeans_solver(vec_); // kmeans 无监督、无标签

	Evaluation(test_x1, 1);
	Evaluation(test_x2, 2);
#ifdef Iris
	Evaluation(test_x3, 3);
#endif
	std::cout << "End";
	return EXIT_SUCCESS;
}

void kmeans_solver(std::vector<CP> vec_, bool visual)
{

	uranus::setZero<sample, Dim>(C1_mean);
	uranus::setZero<sample, Dim>(C2_mean);
	uranus::setZero<sample, Dim>(C3_mean);
	int mark = 0;

	// step1 -- Init
again:
	if (mark)
	{
		C1.clear(); // clear
		C2.clear();
		C3.clear();

		C1.push_back(C1_mean); // update means
		C2.push_back(C2_mean);
		C3.push_back(C3_mean);
	}
	else // first -- rand generate center
	{
		C1.push_back(vec_[uniform_intx(0, vec_.size())].first);
		C2.push_back(vec_[uniform_intx(0, vec_.size())].first); // 不匹配
#ifdef Iris
		C3.push_back(vec_[uniform_intx(0, vec_.size())].first);
#endif
	}

	// step2 -- 按照最小距离法则逐个将样本x划分到以聚类中
	for (size_t idx = 0; idx < vec_.size(); ++idx) // 对 vec_ 的每个样本
	{
		// Sonar
		double res1 = uranus::Norm<sample, Dim>(C1[0], vec_[idx].first);
		double res2 = uranus::Norm<sample, Dim>(C2[0], vec_[idx].first);
		if (res1 > res2)
			C2.push_back(vec_[idx].first);
		else
			C1.push_back(vec_[idx].first);
// Iris
#ifdef Iris
		double res3 = uranus::Norm<sample, Dim>(C3[0], vec_[idx].first);
		if (res3 <= res1 && res3 <= res2)
			C3.push_back(vec_[idx].first);
#endif
	}

	// step3 -- 重新计算k个类的聚类中心
	for (int i = 0; i < C1.size(); ++i)
		C1_mean += C1[i];
	for (int i = 0; i < C2.size(); ++i)
		C2_mean += C2[i];
	C1_mean = C1_mean / C1.size();
	C2_mean = C2_mean / C2.size();
#ifdef Iris
	for (int i = 0; i < C3.size(); ++i)
		C3_mean += C3[i];
	C3_mean = C3_mean / C3.size();
#endif

	if (C1_mean != C1[0] || C2_mean != C2[0] || C3_mean != C3[0])
	{
		if (visual)
			std::cout << "C1 means:" << mark << "\n"
					  << C1_mean << std::endl;
		mark++;
		if (mark > 100)
			return; // 最多100次迭代，防止永不收敛
		goto again;
	}
	else
		return;
}
/**
 * @brief calu N dim Norm number
 * @param 测试集
 * @param 期望的类别
 */
double Evaluation(sample_set test_set, int except_C)
{
	double sorce1 = 0, sorce2 = 0, sorce3 = 0;
	double total = 0.0;
	for (size_t idx = 0; idx < test_set.size(); ++idx) // 对每一个测试集合的样本
	{
		double res1 = uranus::Norm<sample, Dim>(test_set[idx], C1_mean);
		double res2 = uranus::Norm<sample, Dim>(test_set[idx], C2_mean);
		if (res1 > res2)
			sorce1++;
		else
			sorce2++;
#ifdef Iris
		double res3 = uranus::Norm<sample, Dim>(test_set[idx], C3_mean);
		if (res3 <= res1 && res3 <= res2)
			sorce3++;
#endif
	}
	double rate;
	if (except_C == 1)
		rate = sorce1 / test_set.size();
	if (except_C == 2)
		rate = sorce2 / test_set.size();
	if (except_C == 3)
		rate = sorce3 / test_set.size();
	std::cout << except_C << " rate: " << rate << std::endl;
	return rate;
}