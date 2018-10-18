/**
 * author: li chunpeng  <lichunpeng@stu.xidian.edu.cn>
 * create: 2018-10-18
 */
#include <iostream>
#include <random>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "uranus/Matrix.hpp"

using DataSet = std::vector<
					std::pair<std::vector<double>,
							  std::string
				    	>
					>;

// 留着生成随机数，做交叉验证
int uniform_intx(int a, int b) {
	static std::default_random_engine e{ std::random_device{}() };
	static std::uniform_int_distribution<int> u;
	return u(e, std::uniform_int_distribution<int>::param_type(a, b));
}

std::string Trim(std::string& str)
{
	str.erase(0, str.find_first_not_of(" \t\r\n"));
	str.erase(str.find_last_not_of(" \t\r\n") + 1);
	return str;
}
/**	
 * @brief read data set from file ,such as *.csv & *.txt & *.data
 * @param file_name
 * @param feature_vec of feature
 * @param feature_vec of sample set
 * @param feature_vec of class, default as 2
 */
void readData(DataSet& buffer,
			  std::string& file_name, 
              int feature_num, 
              int sample_num,
              int class_num = 2)
{
	std::ifstream fin(file_name);
	std::string line;
	
	for (int index = 0; index < sample_num; ++index)
	{
		std::getline(fin, line);
		std::istringstream sin(line);
		std::vector<std::string> fields;
		std::string field;
		while (std::getline(sin, field, ',')) 
			fields.push_back(field);

		int idx = feature_num;
		std::vector<double> feature_vec(fields.size());
		for (int idx = 0; idx < feature_num; ++idx)
			feature_vec[idx] = std::stod(Trim(fields[idx]));
		std::string label = Trim(fields[idx]);

		buffer[index] = std::make_pair(feature_vec,label);
		/*
		std::cout << "处理之后的字符串：" 
			<< feature_vec[0] << "\t" 
			<< feature_vec[1] << "\t"
			<< feature_vec[2] << "\t"
			<< feature_vec[3] << "\t"
			<< std::endl;
		*/
		// To-do: 还需要多读一个label，还需要封装
	}
}

int main(int argc, char *argv[])
{	
	constexpr int feature = 4;
	constexpr int sample_num = 150;

	DataSet buffer(sample_num);

	std::string name = "iris.data";
	readData(buffer, name, feature, sample_num);

	// To-do : std::vector -> uranus::vector , step by step 
	/*
	std::vector<std::vector<uranus::Vector<feature>>> x_set;

	int class_id;

	for(int j = 0; j < feature; ++j)
		x_set[class_id][0](j) = buffer[0].first[j];
		
	for(int i = 1; i < sample_num; ++i)
	{
		for(int j = 0; j < feature; ++j)
		{
			if(buffer[i].second != buffer[i-1].second) class_id++;
			x_set[class_id][i](j) = buffer[i].first[j];
			// 第i类 / 类内 / 逐元素    第i类  / vector / 逐元素
		}
	}
	*/
	return EXIT_SUCCESS;
}