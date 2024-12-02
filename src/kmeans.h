#ifndef KMEANS_H
#define KMEANS_H

#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <omp.h>

#endif //KMEANS_H

/*
	Created by Qiulin Wu on 2024-12-02.
*/

extern string graphfile; // file path
extern int numofdcs;
using namespace std;
class kmeans {
public:
	/*
		define data struct  
		x:	  rank value; 
		group:cluster id  
		ID:   Unique identifier
	*/
	typedef struct {
		float x;
		int group;
		int ID;
	} point_t, * point;

	typedef struct {
		double x;
		int count;
	} cent_t, * centroid;

	centroid cent;
	point pt;
	int n_cluster;
	int pcount;
	map<int, vector<int>> result;
	vector<double> centvalue;
	int DC_index;
	int iter_index;
	float MaxDiv;
	float MinDiv;
	
	kmeans(int ncluster, int tr) {
		n_cluster = ncluster;		
		cent = (cent_t*)malloc(sizeof(cent_t) * n_cluster);		
		for (int i = 0; i < n_cluster; i++)
		{
			vector<int> pointid;
			result[i] = pointid;
			centvalue.push_back(0);
		}
		DC_index = tr;
		//iter_index = iteration;
		MaxDiv = 0;
		MinDiv = 0;
	} 
	~kmeans() {
		delete cent;
		delete pt;
		for (int i = 0; i < n_cluster; i++) {
			vector<int>().swap(result[i]);
		}
		map<int, vector<int>>().swap(result);
	}
	// some init setting
	void set_pcount(int pc, int iteration) {
		pcount = pc;
		pt = (point_t*)malloc(sizeof(point_t) * pcount); 
		iter_index = iteration;
	}
	// initial
	void init() { 
		for (int i = 0; i < n_cluster; i++)
		{
			result[i].clear();
			centvalue[i] = 0;
		}
	}
	// Randomly returns a floating point number smaller than m
	float randf(float m)
	{
		return m * rand() / (RAND_MAX - 1.);  
	}
	// load data from files
	void load_points_fromfile(string filename)
	{
		point p;  
		std::ifstream graph_file(filename);     //Opens the file *filename in a read-only manner
		std::string line;
		std::getline(graph_file, line);  
		p = pt + 0;
		while (!graph_file.eof()) { 
			//std::cout << line << std::endl;
			std::stringstream strm(line);
			int vertex = 0;
			float value = 0;
			strm >> vertex;             //The vertex ID is the first row of data
			strm.ignore(1);             //ignore the empty row
			strm >> value;              //The rank value is the second row of data
			p->x = value;       
			p->ID = vertex;
			p++;
			std::getline(graph_file, line);
		}
	}
	// load data from data
	void load_points_fromdata(vector<pair<int, float>> data)
	{
		point p; 
		p = pt + 0;
		pcount = data.size(); 
		//map<int, float>::iterator iter;
		for (int i  = 0; i < pcount; i++)  
		{			
			p->x = data[i].second;
			p->ID = data[i].first;

			// clip the msg
			if (p->x > MaxDiv)
				p->x = MaxDiv;
			if (p->x < MinDiv)
				p->x = MinDiv;

			p++;
		}
	}
	// Calculate the distance between two points   
	inline float dist2(centroid a, point b)
	{
		//float x = a->x - b->x, y = a->y - b->y;
		//return x * x + y * y;
		float x = a->x - b->x;
		return x * x;
	}
	// Find the closest centroid
	inline int nearest(point p, int ncluster, float* d2)
	{
		int i, min_i;
		centroid c;
		float d, min_d;

		min_d = HUGE_VAL; 
		min_i = p->group;
		for (c = cent, i = 0; i < ncluster; i++, c++) {
			if (min_d > (d = dist2(c, p))) {   
				min_d = d; min_i = i;
			}
		}
		if (d2)*d2 = min_d; 
		return min_i;
	}
	// kmeans++ for initial
	void kpp()
	{
		int j;
		int i_cluster;
		float * d = (float*)malloc(sizeof(float) * pcount);
		double sum;

		point p;
		
		int randnum = rand() % pcount;
		cent[0].x = pt[randnum].x;  
		centvalue[0] = cent[0].x;
		for (i_cluster = 1; i_cluster < n_cluster; i_cluster++) { 
			sum = 0;
			for (j = 0, p = pt; j < pcount; j++, p++) {
				nearest(p, i_cluster, d + j); 
				sum += d[j];
			}
			sum = randf(sum); 
			for (j = 0, p = pt; j < pcount; j++, p++) {
				if ((sum -= d[j]) > 0  || find(centvalue.begin(), centvalue.end(), pt[j].x) != centvalue.end())
					continue;  
				cent[i_cluster].x = pt[j].x; 
				centvalue[i_cluster] = cent[i_cluster].x;
				break;
			}
		}
				       
	}
	// main step for kmeans
    void lloyd(int round_num, float sigma)
	{
		int i, j;
		int changed;
		point p;
		centroid c; 

		/* assign init grouping randomly */
		time_t start = time(NULL);
		/*
			If it is the first round, kmeans++ is enabled. 
			If it is not the first round, the centroid value of the previous round is used as the initial centroid
		*/
		if (iter_index == 0)  
			kpp(); 
		if (DC_index < numofdcs)
			cout << "DC" << DC_index << " init_centroids" <<endl;
		else
			cout << "DC" << DC_index - numofdcs << " init_centroids" << endl;
		for (int i = 0; i < n_cluster; i++)
		{
			cout << cent[i].x << " ";
		}
		cout << endl;

		time_t end = time(NULL);
		cout << "Kmeans init used " << end - start << "s" << endl;
		float sse = 0;
		vector<float> origins(n_cluster);

		for (int t = 0; t < round_num; t++)
		{
			time_t start_i = time(NULL);
			if (DC_index < numofdcs) 
				cout << "DC" << DC_index << "_out_round" << t << endl;
			else
				cout << "DC" << DC_index - numofdcs << "_in_round" << t << endl;

            #pragma omp parallel for 
			for (int j = 0; j < pcount; j++)
				pt[j].group = nearest(pt + j, n_cluster, 0); 

			/* group element for centroids are used as counters */
			for (c = cent, i = 0; i < n_cluster; i++, c++) {   
				c->count = 0; 
				c->x = 0;
			}
			for (j = 0, p = pt; j < pcount; j++, p++) {  
				c = cent + p->group;  
				c->count++;
				c->x += p->x; 
			}

			if (sigma != 0)
				cout << setw(15) << left << "count" << setw(15) << left << "n_count" << setw(15) << left << "sum" << setw(15) << left << "n_sum" << endl;
			// For the value of each center point, the average rank value is taken as the new center value
			for (c = cent, i = 0; i < n_cluster; i++, c++) {
				origins[i] = c->x / c->count;
				if (c->count == 0)
				{
					c->x /= c->count;
					cout << i << " have no points" << endl;
					continue;
				}
				float noise_count = laplace_generator(0, sigma);  // perturb count
				if (sigma != 0)
					cout << setw(15) << left << c->count << setw(15) << left << noise_count;
				c->count += noise_count;
				float noise_sum = laplace_generator(0, sigma);   // perturb sum
				if (sigma != 0)
					cout << setw(15) << left << c->x << setw(15) << left << noise_sum << endl;
				c->x += noise_sum;
				c->x /= c->count;

				// bound the value
				if (c->x < MinDiv)
					c->x = MinDiv;
				if (c->x > MaxDiv)
					c->x = MaxDiv;

			}
			time_t end_i = time(NULL);
			cout << "Kmeans round " << t << " used " << end_i - start_i << "s" << endl;
		}
		
		cout << setw(15) << left << "original" << setw(15) << left << "centroid" << setw(15) << left << "size" << endl;
		for (int i = 0; i < n_cluster; i++)
		{
			cout << setw(15) << left << origins[i] << setw(15) << left << cent[i].x << setw(15) << left << cent[i].count << endl;
		}
	}

	map<int, vector<int>> Kmeans(string filename, vector<pair<int, float>> data, float budget, float maxdiv, float mindiv)
	{
		MaxDiv = maxdiv;
		MinDiv = mindiv;
		if (filename != "")
			load_points_fromfile(filename); 
		else
			load_points_fromdata(data);    
		init();
		float sigma = 0;
		int round_num = 5; // kmeans iteration counts
		if (budget != 0)
			sigma = ((MaxDiv - MinDiv + 1) * round_num) / budget;  // (r + 1)t / bgt
		cout << "sigma: " << sigma << endl;
		
		lloyd(round_num, sigma);

		return result;
	}
	// Noisy Max
	int GetIndex(float epsilon, float msg_rank, vector<float>& dists, vector<float>& noises) {
		vector<float> C(n_cluster);
		float min_C_noise = std::numeric_limits<float>::max(); 
		int C_index = 0;
		float max_C = std::numeric_limits<float>::lowest(), min_C = std::numeric_limits<float>::max();
		for (int i = 0; i < n_cluster; i++) { 
			C[i] = fabs(msg_rank - cent[i].x);
			dists[i] = C[i];
			if (C[i] > max_C)
				max_C = C[i];
			if (C[i] < min_C)
				min_C = C[i];
		}
		float gs_C = max_C - min_C;
		for (int i = 0; i < n_cluster; i++) {  // noisy_min
			float noise = laplace_generator(0, 2 * gs_C / epsilon);
			noises[i] = noise;
			C[i] += noise;
			if (C[i] < min_C_noise)
			{
				min_C_noise = C[i];
				C_index = i;
			}

		}
		return C_index;
	}

	// generate laplace noise
	double rnd_generator(double lower, double upper) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(lower, upper);
		return dis(gen);
	}
	//Laplace distribution
	double laplace_generator(double mu, double sigma) {
		double rnd = rnd_generator(0, 1) - 0.5;
		double b = sigma / sqrt(2.0);
		float sign = -1;
		if (rnd == 0)
			sign = 0;
		else if (rnd > 0)
			sign = 1;
		rnd = mu - b * sign * log(1 - 2 * rnd * sign);
		return rnd;
	}
};
