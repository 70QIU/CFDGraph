#pragma once
#ifndef DISTRIBUTED_GRAPH_H
#define DISTRIBUTED_GRAPH_H

#include <vector>
#include <random>
#include <mutex>
#include <unordered_map>
#include "Geography.h"
#include <map>
#include <vector>
#include <algorithm>
#include <iomanip>

#define B 2.5 //clamping parameter
#define NLATENT 5 
#define LAMBDA 0.045

/*
	Created by Qiulin Wu on 2024-12-02.
*/

using  namespace std;
extern ofstream fileResult;
extern bool privacy;
#endif

enum Graph_type
{
	test,
	synthetic,
	livejournal,
	twitter,
	roadca
};

enum vertex_status{
	activated,
	deactivated
};

class VertexData{
public:

	float input_size;
	int send_dc_id;   	
	
	virtual float size_of() { return 0; }
	VertexData(){
		input_size = 10; 
	}

	VertexData& operator=(const VertexData& other)  {
		this->input_size = other.input_size;
        return *this;
    }
};
// For PageRank, Personalized PageRank
class PageRankVertexData:public VertexData{
public:
	/*for pagerank application*/
	float rank;
	float last_change;
	double delta; // delta for delta-based pagerank	
	PageRankVertexData(){
		input_size = 10; 
		rank = 0.0; last_change = 0.0; delta = 0; send_dc_id = -1;
	}
	float size_of(){
		return 0.000008;
	}
	PageRankVertexData& operator=(const PageRankVertexData& other)  {
		VertexData::operator= (other);
		this->rank = other.rank;
		this->last_change = other.last_change;
		this->delta = other.delta;
		this->send_dc_id = other.send_dc_id;
        return *this;
    }
};

class PersonalPageRankVertexData :public VertexData {
public:
	/*for personalized pagerank application*/
	float rank;
	float last_change;
	double delta; // delta for delta-based
	PersonalPageRankVertexData() {
		input_size = 10;
		rank = 0.0; last_change = 0.0; delta = 0; send_dc_id = -1;
	}
	float size_of() {
		return 0.000008;
	}
	PersonalPageRankVertexData& operator=(const PersonalPageRankVertexData& other) {
		VertexData::operator= (other);
		this->rank = other.rank;
		this->last_change = other.last_change;
		this->delta = other.delta;
		this->send_dc_id = other.send_dc_id;
		return *this;
	}
};

class HITSVertexData :public VertexData {
public:
	/*for HITS application*/
	double a;
	double h;
	double last_change_a;
	double last_change_h;
	double delta_a, delta_h; // delta for delta-based 
	HITSVertexData() {
		input_size = 10;
		a = 0.0;
		h = 0.0;
		last_change_a = 0.0;
		last_change_h = 0.0;
		delta_a = 0; delta_h = 0;
		send_dc_id = -1;
	}
	float size_of() {
		return 0.000008;
	}
	HITSVertexData& operator=(const HITSVertexData& other) {
		VertexData::operator= (other);
		this->a = other.a;
		this->h = other.h;
		this->last_change_a = other.last_change_a;
		this->last_change_h = other.last_change_h;
		this->delta_a = other.delta_a;
		this->delta_h = other.delta_h;
		this->send_dc_id = other.send_dc_id;
		return *this;
	}
};

class salsaVertexData :public VertexData {
public:
	/*for SALSA application*/
	double authority;
	double hub;
	double last_change_a;
	double last_change_h;
	double delta_a, delta_h; // delta for delta-based 
	salsaVertexData() {
		input_size = 10;
		authority = 0.0; last_change_a = 0.0; last_change_h = 0; hub = 0; delta_a = 0; delta_h = 0; send_dc_id = -1;
	}
	float size_of() {
		return 0.000008;
	}
	salsaVertexData& operator=(const salsaVertexData& other) {
		VertexData::operator= (other);
		this->authority = other.authority;
		this->hub = other.hub;
		this->last_change_h = other.last_change_h;
		this->last_change_a = other.last_change_a;
		this->delta_a = other.delta_a;
		this->delta_h = other.delta_h;
		this->send_dc_id = other.send_dc_id;
		return *this;
	}
};

// vertex data
class MyVertex{

public:
	/*identification*/
	int vertex_id;
	/*init location*/
	Amazon_EC2_regions location_id;
	VertexData* data;
	VertexData* accum;

	vector<int> in_neighbour;
	vector<int> out_neighbour;
	int inSize;
	int outSize;
	//only for PPR application
	bool isChoose;

	// for value application
	vector<VertexData*> messages_sendout_h;  
	vector<VertexData*> messages_sendin_a;  
	vector<double> msg_out_h_except; 
	vector<double> msg_in_a_except;
	std::vector<double> obs;

	vertex_status status;
	vertex_status next_status;


	// int expected_replicas;
	bool is_master;
	int master_location;
	int local_vid;


	MyVertex(){
		master_location = -1;
		status = (vertex_status)deactivated;
		next_status = (vertex_status)deactivated;
		inSize = 0;
		outSize = 0;
		//only for PPR application
		isChoose = false;
	} 

};

// For combiner
class Aggregator {
public:
	int DC_id; //send message to DC_id
	int vertex_id;
	int which_msg;
	VertexData* aggregated_data;
	float noise_budget;
	std::vector<int> v_list; //for DC-level aggregation
	vector<pair<unsigned int, int>> num_list;
	std::vector<int> source_list; //record the message send from which vertex
	Aggregator() {
		DC_id = vertex_id = -1;
		noise_budget = 0;
	}
};

// edge data
class EdgeData{
public:
	int edge_id;
	int start_node_id;
	int end_node_id;
	float weight;
	float obs;
	string edge_type;
	Amazon_EC2_regions assigned_location;
};

class Graph{
public:
	//MyVertex *myvertex;
	vector< MyVertex * > myvertex;
	vector<pair<int, int>> random_edges;
	map<pair<int,int>,EdgeData> edgedata; 
	int num_vertices;
	int num_edges;
	//only for the PPR application
	float num_choosed;
public:

	void GraphInit(int i){
		for (int num = 0; num < i; num++)
		{
			MyVertex* v = new MyVertex();
			myvertex.push_back(v);
		}
	}
	int source_edge(int i){
		return random_edges[i].first;
	}
	int target_edge(int i){
		return random_edges[i].second;
	}

	Graph(){
		random_edges.clear();
		num_edges = 0;
		num_vertices = 0;
		num_choosed = 0.0;
	}
};


class distributed_graph{
public:
	
	Graph_type graph_type;
	Graph* g;
	std::unordered_map<int,pair<int,int>> random_access_edges;
	
	int num_vertices;
	int num_edges;
	/*indicating at initilization stage
	* for SubgraphIsomorphism app only
	*/
	bool init_indicator = true; 
	int  get_in_nbrs_size(int v){
		return g->myvertex[v]->in_neighbour.size(); 
	}

	void  vectorSort() {
		for (int i = 0; i < num_vertices; i++) {
			sort(g->myvertex[i]->in_neighbour.begin(), g->myvertex[i]->in_neighbour.end());
			sort(g->myvertex[i]->out_neighbour.begin(), g->myvertex[i]->out_neighbour.end());
		}
	}
	std::vector<int> get_in_nbrs(int v){
		std::vector<int> in_brs;
		in_brs = g->myvertex[v]->in_neighbour;
		return in_brs;
	}

	int get_out_nbrs_size(int v){
		return g->myvertex[v]->out_neighbour.size();
	}

	std::vector<int> get_out_nbrs(int v){
		std::vector<int> out_nbrs;
		out_nbrs = g->myvertex[v]->out_neighbour;
		return out_nbrs;
	}

	void pdf2cdf(std::vector<double>& pdf) {
		double Z = 0;
		for (size_t i = 0; i < pdf.size(); ++i) Z += pdf[i];
		for (size_t i = 0; i < pdf.size(); ++i)
			pdf[i] = pdf[i] / Z + ((i>0) ? pdf[i - 1] : 0);
	}

	double rnd_generator(double lower, double upper){
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

	//Gamma distribution
	double gamma_generator(double alpha, double sigma, float count) {
		//std::random_device rd;
		std::mt19937 gen(count);
		std::gamma_distribution<> dis(alpha, sigma);

		return dis(gen);
	}
	/**
	* Generate a draw from a multinomial using a CDF.  This is
	* slightly more efficient since normalization is not required
	* and a binary search can be used.
	*/
	size_t multinomial_cdf(const std::vector<double>& cdf) {
		double rnd = rnd_generator(0,1);
		return 1;
		//return std::upper_bound(cdf.begin(), cdf.end(),rnd) - cdf.begin();
	}
	/**
	*   the line parser returns true if the line is parsed successfully and
	*	calls graph.add_vertex(...) or graph.add_edge(...)
	*/
	bool line_parser(const std::string& filename,int source,int target,float weight, float obs, string type);
	

	void load_from_file(char*, int, int);

	~distributed_graph(){delete g; g=NULL; random_access_edges.clear();}
};