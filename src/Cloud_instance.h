#ifndef ENTITIES_H
#define ENTITIES_H
#include "Distributed_graph.h"
#endif

/*
	Created by Qiulin Wu on 2024-12-02.
*/

extern int NUM_DC;
extern float OnDemandLag;

//DataCenter definition
class DataCenter {
public:
	int id;
	int size;
	int pri_rank;   
	float budget_weight;  
	float budget_weight_percentage;
	float budget_sum_weight;  //Record the total number of times messages need to be sent
	Amazon_EC2_regions location;
	
	float data_size; //the total input data size of all vertices in this dc

	/** hpc setting uses upload/download bandwidth*/
	float upload_band;
	float download_band;
	float upload_price;//outbound price to other regions, per GB
	float download_price;

	float g_upload_data = 0.0f;
	float g_dnload_data = 0.0f;
	float a_dnload_data = 0.0f;
	float a_upload_data = 0.0f;

	//privacy
	map<int, vector<Aggregator*>> out_k_aggregators; 
	map<int, vector<Aggregator*>> in_k_aggregators;  
	std::map<string,multimap<double, int>> msgCollecter;  
	DataCenter(){ 
		upload_band = 100;
		download_band = 500;
		data_size = 0;	
		upload_price = 0.02f;
		download_price = 0;
	}
	DataCenter(Amazon_EC2_regions l){
		location = l;
		upload_band = 100;
		download_band = 500;
		if (l == useast){
			upload_price = 0.02f;
		}
		else if (l == uswest_northcalifornia){
			upload_price = 0.02f;
		}
		else if (l == uswest_oregon){
			upload_price = 0.02f;
		}
		else if (l == ap_singapore){
			upload_price = 0.09f;
		}
		else if (l == ap_sydney){
			upload_price = 0.14f;
		}
		else if (l == ap_tokyo){
			upload_price = 0.09f;
		}
		else if (l == eu_ireland){
			upload_price = 0.02f;
		}
		else if (l == saeast){
			upload_price = 0.16f;
		}
	
	}
};