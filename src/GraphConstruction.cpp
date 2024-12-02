#include "Distributed_graph.h"
#include <sstream>
#include <fstream>

#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include "../utils/Utils.h"
#include <string.h>
#include <omp.h>
extern int N_THREADS;
extern string MAP_GRAPH;
static omp_lock_t lock_;

/*
	Created by Qiulin Wu on 2024-12-02.
*/

extern int numofdcs;
extern string graphfile;
extern string assgin_mode;

// Add edges
bool distributed_graph::line_parser(const std::string& filename, int source, int target, float weight,float obs,string type) {
		
	//if the edge is not self-loop, add this edge into the graph
	if (source != target){

		(this->g)->myvertex[target]->in_neighbour.push_back(source);
		(this->g)->myvertex[target]->inSize++;
		(this->g)->myvertex[source]->out_neighbour.push_back(target);
		(this->g)->myvertex[source]->outSize++;
		(this->g)->num_edges++; 
	}
	return true;
}

// Checking if it's a number
bool isNum(string str)
{
	stringstream sin(str);
	double d;
	char c;
	if (!(sin >> d)){
		return false;
	}
	if (sin >> c){
		return false;
	}
	return true;
}

// Gets the format of the input data
char* input_format_specifier(char* filename, char* delimiter)
{
	string line;
	char* data;
	int count = 0;
	ifstream in(filename);
	bool cycle = true;
	char* format = new char[15];
	while (cycle)
	{
		getline(in, line);
		data = (char*)line.data();
		cout << data << endl; 

		char* token = strtok(data, delimiter);
		count = 0;
		while (token){
			token = strtok(NULL, delimiter);
			count++;
		}
		if (count == 1){
			//cout << "The file format is incorrect." << endl;
			return NULL;
		}
		char* first = strtok(data, delimiter);
		if (!isNum(first))
			continue;
		switch (count)
		{
		case 2:
			strcpy(format, "%u %u");
			format[5] = '\0';
			cycle = false;
			break;
		case 3:
			strcpy(format, "%u %u %f");
			format[8] = '\0';
			cycle = false;
			break;
		}
	}
	return format;
}

// load data from file
void distributed_graph::load_from_file(char* filename, int vertices, int edges){
	omp_init_lock(&lock_); 
	this->g = new Graph();
	std::cout << "load from file " << filename <<std::endl;
	//load vertices	

	this->num_vertices = vertices;
	this->num_edges = edges;
	std::vector<float> rndlocs;
	std::ifstream rndfile("rndlog");        //Opens the file rndlog in a read-only manner
	std::string rndline;
	
	//Read the contents of the rndlog file line by line and place them in the rndlocs vector	
	for (int i = 0; i < 1000000; i++) {
		std::getline(rndfile,rndline);      // getline(): Read a line from the document rndfile and store it in the rndline  
		std::stringstream strm(rndline);    // stringstream: Convert a string to a number
		float loc;
		strm >> loc;
		rndlocs.push_back(loc);
	}
	std::cout << "load from random file done" << std::endl;
	
	this->g->GraphInit(num_vertices);
	vector<int> DC_Vnum; 
	vector<int> DC_Vexp; 
	vector<float> percent; 
	
	percent = { 0.43,0.31,0.11,0.1,0.05 };
		
	int sum = 0;
	for (int i = 0; i < numofdcs; i++)
	{
		DC_Vnum.push_back(0);
		DC_Vexp.push_back(ceil(percent[i] * num_vertices));
		sum += DC_Vexp[i];
		cout << "DC_Vexp["<<i<<"]=" << DC_Vexp[i] << endl;
	}
	cout << "sum=" << sum << ", num_vertices=" << num_vertices << endl;
	if (sum < num_vertices)
	{
		DC_Vexp[numofdcs - 1] += (num_vertices - sum);
		cout << "DC_Vexp=" << DC_Vexp[numofdcs - 1] << endl;
	}
	// DC id
	vector<int> DC_ID(numofdcs);
	for (int i = 0; i < numofdcs; i++)
		DC_ID[i] = i;
	int numofdcs_tmp = numofdcs;

	for (int i = 0; i < num_vertices; ++i){
		g->myvertex[i]->vertex_id = i;
		g->myvertex[i]->isChoose = false; // For PPR
		int loc = std::floor(rndlocs[i % 1000000] * (float)numofdcs_tmp);
		loc = DC_ID[loc];
		DC_Vnum[loc] += 1;
		if (DC_Vnum[loc] >= DC_Vexp[loc]){
			numofdcs_tmp -= 1;
			vector<int>::iterator it = DC_ID.begin();
			for (int sb = 0; sb < DC_ID.size(); sb++){
				if (DC_ID[sb] == loc)
					break;
				it++;
			}
			DC_ID.erase(it); 
		}
		g->myvertex[i]->location_id = (Amazon_EC2_regions)loc;  //random assign DC
		(this->g)->num_vertices++;
	}
	printf("Load vertices done.\n");
	printf("Get ready for vertex mapping.\n");
	int ecount = 0;
	map<int, int> vertex_map;  // Used to map vertices, which may be non-contiguous in the original graph(eg: 0,2,4->0,1,2)
	map<int, int> has_maped;
	size_t map_index = 0;
	string datafile;
	if (MAP_GRAPH == "map")
	{
		datafile = filename;
		datafile += "_mapped";
		for (int i = 0; i < vertices; i++)
			vertex_map[i] = -1;
	}
	ofstream OutFile(datafile); 

	char str[50];
	time_t now = time(NULL);
	strftime(str, 50, "%x %X", localtime(&now));
	std::cout <<"Begin to graph construction at time: "<< str << std::endl;
	cout << "MAP_GRAPH=" << MAP_GRAPH << endl;
	/**/
	FILE* file_descriptor = fopen(filename, "r");
	size_t source = 0;
	size_t target = 0;
	float weight = 0.0;
	if (!file_descriptor)
	{
		cout << "Error : can't open the input file!" << endl;
		exit(-1);
	}
	char delimiter[] = " ,'\t'";
	char* format = input_format_specifier(filename, delimiter);
	if (!format)
	{
		cout << "input file format is error!" << endl;
	}

	int skip = 0;
	cout << "format=" << format << endl;
	if (strcmp(format, "%u %u") == 0) {
		char line[80];
		while (fscanf(file_descriptor, format, &source, &target) != 2) {
			fscanf(file_descriptor, "%[^\n]%*c", line); //»»ÐÐ
			cout << "skip line" << line << endl;
			skip += 1;
		}
	}
	if (strcmp(format, "%u %u %f") == 0) {
		char line[80];
		while (fscanf(file_descriptor, format, &source, &target, &weight) != 3) {
			fscanf(file_descriptor, "%[^\n]%*c", line); //»»ÐÐ
			cout << "skip line" << line << endl;
			skip += 1;
		}
	}
	cout << "skip " << skip << " lines" << endl;
	

	if (strcmp(format, "%u %u") == 0)
	{
		cout << "graph format: source->target" << endl;
		while (fscanf(file_descriptor, format, &source, &target) == 2)
		{
			if (MAP_GRAPH == "map")
			{
				if (has_maped.find(source) == has_maped.end())
				{
					vertex_map[map_index] = source;
					has_maped[source] = map_index;
					source = map_index;
					map_index++;
				}
				else {
					source = has_maped[source];
				}
				if (has_maped.find(target) == has_maped.end())
				{
					vertex_map[map_index] = target;
					has_maped[target] = map_index;
					target = map_index;
					map_index++;
				}
				else {
					target = has_maped[target];

				}
				OutFile << source << " " << target << std::endl;
			}
			line_parser(filename, source, target, 1.0, 0.0, "NONE");  //add edge
			ecount++;
			if (ecount % 1000000 == 0)
			{
				std::cout << "Thread " << omp_get_thread_num() << ": " << ecount << " edges inserted\n";
			}
		}
	}
	else if (strcmp(format, "%u %u %f") == 0)
	{
		cout << "graph format: source->target weight" << endl;
		while (fscanf(file_descriptor, format, &source, &target, &weight) == 3)
		{
			if (MAP_GRAPH == "map")
			{
				if (has_maped.find(source) == has_maped.end())
				{
					vertex_map[map_index] = source;
					has_maped[source] = map_index;
					source = map_index;
					map_index++;
				}
				else {
					source = has_maped[source];
				}
				if (has_maped.find(target) == has_maped.end())
				{
					vertex_map[map_index] = target;
					has_maped[target] = map_index;
					target = map_index;
					map_index++;
				}
				else {
					target = has_maped[target];
				}
				OutFile << source << " " << target << std::endl;
			}
			line_parser(filename, source, target, weight, 0.0, "NONE");  //add edge
			ecount++;
			if (ecount % 1000000 == 0)
			{
				std::cout << "Thread " << omp_get_thread_num() << ": " << ecount << " edges inserted\n";
			}
		}
	}
	else
	{
		cout << "graph fromat Error!!" << endl;
		cout << "format=" << format << endl;
	}
	OutFile.close();
	if (MAP_GRAPH == "map")
	{
		has_maped.clear();
		vertex_map.clear();
	}
	time_t now_ = time(NULL);
	strftime(str, 50, "%x %X", localtime(&now_));
	std::cout << "Finish graph construction at time: " << str << std::endl;
	
	printf("Vertex number mapping successful.\n");
	printf("Load edges done.\n");
	if (MAP_GRAPH == "map")
		exit(1);
	
	// 30% of the points are randomly selected as the source points of the PPR
	cout << "Begin choosing source points" << endl;
	string sourcefile = "./ppr_sources/" + graphfile + "_30_percent.txt";
	ifstream ppr_source(sourcefile);
	string source_line;	
	int testcount = 0;
	while (getline(ppr_source, source_line)) {	
		istringstream iss(source_line);  
		int source_point;
		iss >> source_point;
		g->myvertex[source_point]->isChoose = true; 
	}
	cout << "Finish choosing source points" << endl;

	//Output the num of across DC's edges
	long int across_num = 0;
	long int numedges = 0;
	long int numvertices = (this->g)->num_vertices;

	for (int vertex = 0; vertex < numvertices; vertex++) {
		int src = vertex;
		for (int nbr = 0; nbr < (this->g)->myvertex[vertex]->outSize; nbr++)
		{
			int tgt = (this->g)->myvertex[vertex]->out_neighbour[nbr];
			if ((this->g)->myvertex[tgt]->location_id != (this->g)->myvertex[src]->location_id)
			{
				across_num++;
			}
			numedges++;
		}
	}
	std::cout << "Total num of edges: " << numedges << ". Across DC's num of edges:  "  << across_num << std::endl;
	std::cout << "should have " << edges << " edges," << "added "<< ecount <<" edges." << std::endl;
	this->num_edges = ecount;
}
//end of load_from_file