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
float sampling_rate;
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
float percent;

int BASELINE;
string graphfile;

// clip msg
float msg_upperbound;

// budget assignment
float cent_bgt_percent;

// vertex assignment
string assgin_mode;

int main(int argc, char* argv[])
{
	char str[50];
	time_t now = time(NULL);
	strftime(str, 50, "%x %X", localtime(&now));
	std::cout << str << std::endl;

	int numpoint = atoi(argv[3]);
	bits = atoi(argv[18]);
	ncluster = pow(2, bits);

	percent = stof(argv[19]);

	msg_upperbound = stof(argv[20]); 
	cent_bgt_percent = stof(argv[21]); 
	numofdcs = stoi(argv[22]);  
	assgin_mode = argv[23];  

	std::vector<DataCenter*> DCs;
	if(strcmp(argv[9],"real") == 0){ //for Amazon EC2 experiments
		numofdcs = 8; // real cloud
		for (int i = 0; i < numofdcs; i++){
			DataCenter* DC = new DataCenter(Amazon_EC2_regions(i));
			DC->id = i;
			DC->location = Amazon_EC2_regions(i);
			if(i == 0 || i == 5)
				DC->download_band = 100;	//usually it's 500
			DCs.push_back(DC);		
		}
	} 
	else {
		vector<int> pri_rank_arrs(numofdcs);
		if (assgin_mode == "uniform") {
			for (int i = 0; i < numofdcs; i++)
				pri_rank_arrs[i] = rand() % numofdcs;
		}
		else {
			pri_rank_arrs = { 2,3,1,3,3 };
		}
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
	}
	N_THREADS = atoi(argv[7]);
	MAP_GRAPH = argv[16];
	BASELINE = int(atof(argv[17]));
	graphfile = argv[1];
	string degree = (argv[3]); 
	distributed_graph* dag = new distributed_graph();
	
	if (strcmp(argv[1], "bitcoinotc") == 0) {
		string filename = "bitcoinotc/soc-sign-bitcoinotc.txt";
		int v = 5881;
		long int e = 35592;
		dag->load_from_file((char*)filename.data(), v, e);
	}
	else if (strcmp(argv[1], "bitcoinalpha") == 0) {
		string filename = "bitcoinalpha/soc-sign-bitcoinalpha.txt";
		int v = 3783;
		long int e = 24186;
		dag->load_from_file((char*)filename.data(), v, e);
	}

	int Max_Iteration = atoi(argv[4]); 
	float budget = atof(argv[6]);	
	noise_budget = atof(argv[8]);
	sampling_rate = atof(argv[15]);
	if(strcmp(argv[10], "privacy") == 0)
		privacy = true;
	else privacy = false;
	 
	BaseApp* myapp;
	if (strcmp(argv[2], "pagerank") == 0){
		myapp = new PageRank(); 
		myapp->mytype = pagerank;
		for(int v=0; v<dag->num_vertices; v++){
			(dag->g)->myvertex[v]->data = new PageRankVertexData();
			(dag->g)->myvertex[v]->accum = new PageRankVertexData();
		}
	}
	else if (strcmp(argv[2], "personalpagerank") == 0) {
		myapp = new PersonalPageRank();
		myapp->mytype = personalpagerank; 
		for (int v = 0; v < dag->num_vertices; v++) {
			(dag->g)->myvertex[v]->data = new PersonalPageRankVertexData();
			(dag->g)->myvertex[v]->accum = new PersonalPageRankVertexData();
		}
	}  	 
	else if (strcmp(argv[2], "salsa") == 0) {
		myapp = new Salsa();
		myapp->mytype = salsa;
		for (int v = 0; v < dag->num_vertices; v++) {
			(dag->g)->myvertex[v]->data = new salsaVertexData();
			(dag->g)->myvertex[v]->accum = new salsaVertexData();
		}
		printf("Use the salsa application.\n");
	}
	else if (strcmp(argv[2], "hits") == 0) {
		myapp = new HITS();
		myapp->mytype = hits;
		for (int v = 0; v < dag->num_vertices; v++) {
			(dag->g)->myvertex[v]->data = new HITSVertexData();
			(dag->g)->myvertex[v]->accum = new HITSVertexData();
		}
		printf("Use the HITS application.\n");
	}

	myapp->budget = budget;
	myapp->ITERATIONS = Max_Iteration;
	myapp->global_graph = dag;
	

	EngineType type = synchronous;
	GraphEngine* engine = new GraphEngine(type,DCs.size());
	engine->myapp = myapp;
	engine->DCs = DCs;

	AGG_NUM_withNoise = atoi(argv[13]);

	if (strcmp(argv[12], "pregel") == 0)
		engine->Pregel(argv[3]);
	else if (strcmp(argv[12], "cfdgraph") == 0)
		engine->QuantizePR(argv[3]);

	time_t end = time(NULL);
	strftime(str, 50, "%x %X", localtime(&end));
	std::cout << str << std::endl;
	cout << "Total used " << end - now << "s" << endl;
	return 0;
}

