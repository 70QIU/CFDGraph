#ifndef GEOGRAPHY_H
#define GEOGRAPHY_H
#include <string>
#include <iostream>
#include <fstream>
using namespace std;
#endif

/*
	Created by Qiulin Wu on 2024-12-02.
*/

//cluster different countries into a small number of cloud regions
extern ofstream fileResult;
enum Amazon_EC2_regions
{
	useast=0,
	uswest_oregon=1,
	uswest_northcalifornia=2,
	eu_ireland=3,
	ap_singapore=4,
	ap_tokyo=5,
	ap_sydney=6,
	saeast=7,
	sim1,
	sim2,
	sim3,
	sim4,
	sim5,
	sim6,
	sim7,
	sim8,
	sim9,
	sim10,
	sim11,
	sim12,
	//num_regions
	east_asia,
	jp_west,
	west_eu,
	east_us
};