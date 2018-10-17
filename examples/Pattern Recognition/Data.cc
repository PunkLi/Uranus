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

// 留着生成随机数，做交叉验证
int uniform_intx(int a, int b) {
	static std::default_random_engine e{ std::random_device{}() };
	static std::uniform_int_distribution<int> u;
	return u(e, std::uniform_int_distribution<int>::param_type(a, b));
}

double Trim(std::string& str)
{
	str.erase(0, str.find_first_not_of(" \t\r\n"));
	str.erase(str.find_last_not_of(" \t\r\n") + 1);
	return std::stod(str);
}

/**	
 * @brief read data set from file ,such as *.csv & *.txt & *.data
 * @param file_name
 * @param number of feature
 * @param number of sample set
 * @param number of class, default as 2
 */
void readData(std::string& file_name, 
              int feature_num, 
              int sample_num,
              int class_num = 2)
{
	std::ifstream fin(file_name);
	std::string line;
	
	while (sample_num--)
	{
		std::getline(fin, line);
		std::istringstream sin(line);
		std::vector<std::string> fields;
		std::string field;
		while (std::getline(sin, field, ',')) 
			fields.push_back(field);

		int idx = feature_num;
		std::vector<double> number(fields.size());
		for (int idx = 0; idx < feature_num; ++idx)
			number[idx] = Trim(fields[idx]);

		std::cout << "处理之后的字符串：" 
			<< number[0] << "\t" 
			<< number[1] << "\t"
			<< number[2] << "\t"
			<< number[3] << "\t"
			<< std::endl;

		// To-do: 还需要多读一个label，还需要封装
	}
}

int main(int argc, char *argv[])
{	
	std::string name = "iris.data";
	readData(name, 4, 150);
	return EXIT_SUCCESS;
}