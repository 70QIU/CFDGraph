#include "Utils.h"
#include <algorithm>
#include <string>

/*
	Created by Qiulin Wu on 2024-12-02.
*/

double utils::feibonaqi_n(float n, float n_front, int k_num) {
	/**
	 * n:      The nth Fibonacci value
	 * n_front:The n - 1 th Fibonacci value
	 * k_num:  When n and n_front values are not passed, the value of the k_num item is returned 
	 * return: The n + 1 th Fibonacci value
	*/
	if (n && n_front)
		return n + n_front;

	else {
		double k1 = 1, k2 = 1;
		for (int i = 0; i < k_num - 2; i++) {
			float temp = k2;
			k2 = feibonaqi_n(k1, k2, 0);
			k1 = temp;
		}
		return k2;
	}
}

double utils::feibonaqi_sum(int n) {
	/**
	* n:      The sum of n-term Fibonacci numbers satisfying a certain relation
	* result: An array to hold the results
	*/
	if (n == 1)
		return 1.0;
	else if (n == 2)
		return 2.0;
	double k1 = 1, k2 = 1, start = 1, sum = 2;
	//First compute the sum of the first n terms starting from the first term
	for (int i = 0; i < n - 2; i++) {  
		double temp = k2;
		k2 = feibonaqi_n(k1, k2, 0);
		sum = k2 + sum;
		k1 = temp;
	}
	return sum;
}

// Calculate the next power of two
unsigned int utils::nextPowerOfTwo(unsigned int n) {
	if (n == 0)
		return 1;

	n--;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;

	return n + 1;
}

// Converts an unsigned integer to a binary string by the specified number of bits
string utils::toBinaryString(unsigned int n, int bitWidth) {
	std::string binaryStr;
	if (n == 0)
		return std::string(bitWidth, '0');  
	while (n > 0) {
		binaryStr = (n & 1 ? "1" : "0") + binaryStr;
		n >>= 1; 
	}
	unsigned int missingZeros = bitWidth > binaryStr.size() ? bitWidth - binaryStr.size() : 0;
	return std::string(missingZeros, '0') + binaryStr;
}

// Converts the string to an unsigned integer
unsigned int utils::toUnsignedInt(string binaryStr) {

	bitset<32> bitsetStr(binaryStr);
	unsigned int num = bitsetStr.to_ulong();
	return num;
}