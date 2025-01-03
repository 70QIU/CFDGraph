#include "src/Simulator.h"
#include <ctime>
#include <stdlib.h>  
#include <time.h>  
#include <string.h>
#include "src/kmeans.h"

/*
	Created by Qiulin Wu on 2024-12-02.
*/

int N_THREADS;
int numofdcs;
float noise_budget;
bool privacy;  
int AGG_NUM_withNoise = 0;

double PR_MAX = 500;  
double PR_MIN = 0.15;  
 
// for quantization
int bits;
int ncluster;
string MAP_GRAPH;

int BASELINE;
string graphfile;

// clip msg
float msg_upperbound;

// budget assignment
float cent_bgt_percent;

string algorithm;
string method;

int main(int argc, char* argv[])
{
	char str[50];
	time_t now = time(NULL);
	strftime(str, 50, "%x %X", localtime(&now));
	std::cout << str << std::endl;

	graphfile = argv[1];
	algorithm = argv[2];
	int numpoint = atoi(argv[3]);
	int Max_Iteration = atoi(argv[4]);
	N_THREADS = atoi(argv[5]);
	noise_budget = atof(argv[6]);

	if (strcmp(argv[7], "privacy") == 0)
		privacy = true;
	else privacy = false;

	method = argv[8];	
	AGG_NUM_withNoise = atoi(argv[9]);
	BASELINE = int(atof(argv[10]));

	bits = atoi(argv[11]);
	ncluster = pow(2, bits);

	msg_upperbound = stof(argv[12]);

	MAP_GRAPH = "Notmap";
	cent_bgt_percent = 0.5;
	numofdcs = 5;  

	std::vector<DataCenter*> DCs;
	vector<int> pri_rank_arrs(numofdcs);
	pri_rank_arrs = { 2,3,1,3,3 };

	srand(time(NULL));

	for (int i = 0; i < numofdcs; i++) {
		DataCenter* DC = new DataCenter();
		DC->id = i;
		DC->location = Amazon_EC2_regions(i);
		DC->pri_rank = pri_rank_arrs[i];
		DCs.push_back(DC);
	}
	for (int i = 0; i < numofdcs; i++)
	{
		int weight = 0;
		for (int index = 0; index < numofdcs; index++)
		{
			if (index == i)
				continue;
			if (DCs[i]->pri_rank > DCs[index]->pri_rank)
				weight++;
		}
		DCs[i]->budget_weight = weight;
		weight = 0;
	}

	float sum_weight = 0;
	for (int i = 0; i < numofdcs; i++)
		sum_weight += DCs[i]->budget_weight;
	for (int i = 0; i < numofdcs; i++)
	{
		DCs[i]->budget_sum_weight = sum_weight;
		DCs[i]->budget_weight_percentage = DCs[i]->budget_weight / sum_weight;
	}
		
	
	
	
	distributed_graph* dag = new distributed_graph();
	
	if (graphfile == "bitcoinotc") {
		string filename = "bitcoinotc/soc-sign-bitcoinotc.txt";
		int v = 5881;
		long int e = 35592;
		dag->load_from_file((char*)filename.data(), v, e);
	}
	else if (graphfile == "bitcoinalpha") {
		string filename = "bitcoinalpha/soc-sign-bitcoinalpha.txt";
		int v = 3783;
		long int e = 24186;
		dag->load_from_file((char*)filename.data(), v, e);
	}

	
	

	 
	BaseApp* myapp;
	if (algorithm == "pagerank"){
		myapp = new PageRank(); 
		myapp->mytype = pagerank;
		for(int v=0; v<dag->num_vertices; v++){
			(dag->g)->myvertex[v]->data = new PageRankVertexData();
			(dag->g)->myvertex[v]->accum = new PageRankVertexData();
		}
	}
	else if (algorithm == "personalpagerank") {
		myapp = new PersonalPageRank();
		myapp->mytype = personalpagerank; 
		for (int v = 0; v < dag->num_vertices; v++) {
			(dag->g)->myvertex[v]->data = new PersonalPageRankVertexData();
			(dag->g)->myvertex[v]->accum = new PersonalPageRankVertexData();
		}
	}  
	else if (algorithm == "hits") {
		myapp = new HITS();
		myapp->mytype = hits;
		for (int v = 0; v < dag->num_vertices; v++) {
			(dag->g)->myvertex[v]->data = new HITSVertexData();
			(dag->g)->myvertex[v]->accum = new HITSVertexData();
		}
		printf("Use the HITS application.\n");
	}
	else if (algorithm == "salsa") {
		myapp = new Salsa();
		myapp->mytype = salsa;
		for (int v = 0; v < dag->num_vertices; v++) {
			(dag->g)->myvertex[v]->data = new salsaVertexData();
			(dag->g)->myvertex[v]->accum = new salsaVertexData();
		}
		printf("Use the salsa application.\n");
	}	

	myapp->ITERATIONS = Max_Iteration;
	myapp->global_graph = dag;
	

	EngineType type = synchronous;
	GraphEngine* engine = new GraphEngine(type,DCs.size());
	engine->myapp = myapp;
	engine->DCs = DCs;

	

	if (method == "pregel")
		engine->Pregel(argv[3]);
	else if (method == "cfdgraph")
		engine->CFDGraph(argv[3]);

	time_t end = time(NULL);
	strftime(str, 50, "%x %X", localtime(&end));
	std::cout << str << std::endl;
	cout << "Total used " << end - now << "s" << endl;
	return 0;
}

