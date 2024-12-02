#ifndef UTILS_H
#define UTILS_H

/*
	Created by Qiulin Wu on 2024-12-02.
*/

#include <vector>
#include <ctime>
#include <random>
#include <map>
#include <bitset>
#include "../src/Distributed_graph.h"
namespace utils {

	// For noise sampling
	double feibonaqi_sum(int n);  
	double feibonaqi_n(float n, float n_front, int k_num);  

	// For reindexing
	unsigned int nextPowerOfTwo(unsigned int n);
	string toBinaryString(unsigned int n, int bitWidth);
	unsigned int toUnsignedInt(string binaryStr);
}
#endif //UTILS_H
