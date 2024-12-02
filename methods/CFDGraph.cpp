#include "../src/Simulator.h"
#include "../utils/Utils.h"
#include <omp.h>
#include <string>
#include <algorithm> 
#include <iostream>
#include <fstream> 
#include <sstream>
#include <unordered_map>
#include "../src/kmeans.h"
#include <chrono> 

/*
	Created by Qiulin Wu on 2024-12-02.
*/

extern int BASELINE;
extern int N_THREADS; 
extern float noise_budget;
extern bool privacy;
extern double PR_MAX;
extern double PR_MIN;
extern string graphfile; 

// k-means by 70
extern kmeans* mykmeans;
extern int ncluster;

// combiner
extern int AGG_NUM_withNoise;

// clip msg
extern float msg_upperbound;

extern float cent_bgt_percent;


// Our proposed method CFDGraph
void GraphEngine::CFDGraph(const char* sizeofgraph){

	int total_vertices = (myapp->global_graph->g)->num_vertices; 
	const int num_vertex_locks = total_vertices;
	omp_lock_t* lock_vertex = new omp_lock_t[num_vertex_locks];

	for (int i = 0; i < num_vertex_locks; i++) 
		omp_init_lock(&lock_vertex[i]);
    

	int num_of_DC_combiner = AGG_NUM_withNoise;
	const int num_combiner_locks = num_threads * num_threads * num_of_DC_combiner;  // 扩大到K个
	omp_lock_t* lock_out_combiner = new omp_lock_t[num_combiner_locks];

	for (int i = 0; i < num_combiner_locks; i++)
		omp_init_lock(&lock_out_combiner[i]);

	omp_lock_t* lock_in_combiner = new omp_lock_t[num_combiner_locks];

	for (int i = 0; i < num_combiner_locks; i++)
		omp_init_lock(&lock_in_combiner[i]);

    /**
    * Clear memory.
    */
    Threads.clear();
	for (int i = 0; i < num_threads; i++) {
        Threads.push_back(new Thread());
    }
	
    omp_set_num_threads(N_THREADS);

    #pragma omp parallel for
	for (int i = 0; i < num_threads; i++) {
		if (num_threads != DCs.size()) {
			std::cout << "the number of threads has to be the same as the number of DCs" << std::endl;
			exit(1);
		}
		printf("constructing local graph for thread %d\n", i);
		Threads[i]->DC_loc_id = i;
		Threads[i]->l_dag = new distributed_graph();
		Threads[i]->l_dag->g = new Graph();

		//add vertices
		int local_id = 0;
		for (int vi = 0; vi < total_vertices; vi++) {
			if ((myapp->global_graph->g)->myvertex[vi]->location_id == i) {
				MyVertex* v = new MyVertex(*(myapp->global_graph->g)->myvertex[vi]);
				v->local_vid = local_id;
				v->is_master = true;
				Threads[i]->l_dag->g->myvertex.push_back(v);
				//v->vertex_id = i;
				local_id++;
				Threads[i]->l_dag->g->num_vertices++;
				Threads[i]->l_dag->num_vertices++;

			}
		}
		printf("Thread %d\n #vertices: %d\n", i, Threads[i]->l_dag->g->num_vertices);
	}

	vector<vector<int>> out_msg_counts(num_threads); 
	vector<int> Inter_out_msg_counts(num_threads);   
	for (int tr = 0; tr < num_threads; tr++) 
		out_msg_counts[tr].resize(num_threads);
	vector<vector<int>> in_msg_counts(num_threads); 
	vector<int> Inter_in_msg_counts(num_threads);    
	for (int tr = 0; tr < num_threads; tr++)
		in_msg_counts[tr].resize(num_threads);

	vector<int> inter_in_degree(total_vertices);
	vector<int> intra_in_degree(total_vertices);
	vector<int> inter_out_degree(total_vertices);
	vector<int> intra_out_degree(total_vertices);
	vector<int> inter_all_degree(total_vertices);

	vector<vector<int>> inter_in_degree_pair(num_threads);
	vector<vector<int>> inter_out_degree_pair(num_threads);
	vector<vector<int>> inter_all_degree_pair(num_threads); 
	for (int i = 0; i < num_threads; i++)
	{
		inter_in_degree_pair[i].resize(total_vertices);
		inter_out_degree_pair[i].resize(total_vertices);
		inter_all_degree_pair[i].resize(total_vertices);
	}

    for(int tr = 0; tr< num_threads; tr++){
        /**
        * Signal vertices
        */
        for(int v=0; v<Threads[tr]->l_dag->num_vertices; v++){
            int vid = (Threads[tr]->l_dag->g)->myvertex[v]->vertex_id;
			if (myapp->mytype == pagerank)
			{
				dynamic_cast<PageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank = 0;
			}
			else if (myapp->mytype == personalpagerank)
			{
				if ((myapp->global_graph->g)->myvertex[vid]->isChoose == false)
					dynamic_cast<PersonalPageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank = 0;
				else
					dynamic_cast<PersonalPageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank = 1;
			}
			else if (myapp->mytype == hits)
			{
				dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->a = 1;
				dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->h = 1;
			}
			else if (myapp->mytype == salsa)
			{
				dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->authority = 1;
				dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->hub = 1;
			}

			int current_dc = (myapp->global_graph->g)->myvertex[vid]->location_id; 
			vector<int> out_nbrs = (myapp->global_graph->g)->myvertex[vid]->out_neighbour;			
			for (int nid = 0; nid < out_nbrs.size(); nid++)
			{
				int other_dc = (myapp->global_graph->g)->myvertex[out_nbrs[nid]]->location_id; 
				out_msg_counts[tr][other_dc]++;
				if (current_dc != other_dc)
					inter_out_degree[vid]++;
				else
					intra_out_degree[vid]++;
				inter_out_degree_pair[other_dc][vid]++;
			}
				
			vector<int> in_nbrs = (myapp->global_graph->g)->myvertex[vid]->in_neighbour;		
			for (int nid = 0; nid < in_nbrs.size(); nid++) 
			{
				int other_dc = (myapp->global_graph->g)->myvertex[in_nbrs[nid]]->location_id; 
				in_msg_counts[tr][other_dc]++;
				if (current_dc != other_dc)
					inter_in_degree[vid]++;
				else
					intra_in_degree[vid]++;
				inter_in_degree_pair[other_dc][vid]++;
			}
			inter_all_degree[vid] = inter_out_degree[vid] + inter_in_degree[vid];
			for (int i = 0; i < num_threads; i++)
				inter_all_degree_pair[i][vid] = inter_out_degree_pair[i][vid] + inter_in_degree_pair[i][vid];

			// clear
			vector<int>().swap(out_nbrs);
			vector<int>().swap(in_nbrs);
        }
    }

	for (int tr = 0; tr < num_threads; tr++)
	{
		int intra_out_msg_count = 0;
		int intra_in_msg_count = 0;
		for (int other_dc = 0; other_dc < num_threads; other_dc++){
			if (tr != other_dc)
			{
				Inter_out_msg_counts[tr] += out_msg_counts[tr][other_dc];  
				Inter_in_msg_counts[tr] += in_msg_counts[tr][other_dc];
			}
			else
			{
				intra_out_msg_count += out_msg_counts[tr][other_dc];
				intra_in_msg_count += in_msg_counts[tr][other_dc];
			}
		}
	}	

    /* Start the engine, each thread executes at the same time */
    int iter_counter = 0;
    double gan_usage = 0.0;
	double hits_a_norm = 1, hits_h_norm = 1; 
	vector<double> rank_, rank_h, rank_a;

	int topk_size = 10;
	vector<vector<double>> accuracy(topk_size);
	vector<vector<double>> accuracy_h(topk_size); 
	for (int i = 0; i < topk_size; i++)
	{
		accuracy[i].resize(myapp->ITERATIONS);
		accuracy_h[i].resize(myapp->ITERATIONS);
	}

	char str[50];

	// reindexing module
	vector<vector<pair<unsigned int, int>>> global_to_local(num_threads);  
	vector<vector<unordered_map<string, int>>> local_to_global(num_threads);  
	
	vector<vector<map<unsigned int, int>>> id_to_vertex(num_threads);  
	vector<vector<unsigned int>> common_id(num_threads);    
	vector<vector<int>> common_bit_width(num_threads);  
	for (int i = 0; i < num_threads; i++)
	{
		global_to_local[i].resize(total_vertices);
		local_to_global[i].resize(num_threads);
		id_to_vertex[i].resize(num_threads);
		common_id[i].resize(num_threads, 0);
		common_bit_width[i].resize(num_threads);
	}

	if (myapp->mytype == pagerank || myapp->mytype == personalpagerank) {
		for (int vi = 0; vi < total_vertices; vi++) {
			int current_dc = (myapp->global_graph->g)->myvertex[vi]->location_id; 
			for (int i = 0; i < num_threads; i++)
			{
				if (i == current_dc) 
					continue;
				if (inter_in_degree_pair[i][vi] > 0) {
					id_to_vertex[current_dc][i][common_id[current_dc][i]] = vi;
					common_id[current_dc][i]++;
				}
							
			}
		}
	}
	else if (myapp->mytype == hits || myapp->mytype == salsa) { 
		for (int vi = 0; vi < total_vertices; vi++) {
			int current_dc = (myapp->global_graph->g)->myvertex[vi]->location_id;
			for (int i = 0; i < num_threads; i++) {
				if (i == current_dc)
					continue;

				if (inter_all_degree_pair[i][vi] > 0)
				{
					id_to_vertex[current_dc][i][common_id[current_dc][i]] = vi;
					common_id[current_dc][i]++;
				}
			}
			
		}
	}
	for (int i = 0; i < num_threads; i++)
	{
		for (int j = 0; j < num_threads; j++)
		{
			if (i == j)
				continue;
			common_bit_width[i][j] = log2(utils::nextPowerOfTwo(common_id[i][j]));
		}
	}

	double code_size = 0;  
	int total_code_num = 0; 
	double average_code_size = 0;
	double total_Sequence_size = 0;
	int total_Sequence = 0;

	for (int i = 0; i < num_threads; i++)
	{
		for (int j = 0; j < num_threads; j++)
		{
			if (i == j) 
				continue;
			for (int k = 0; k < common_id[i][j]; k++) {
				int vertex_id = id_to_vertex[i][j][k];
				string code = utils::toBinaryString(k, common_bit_width[i][j]);
				int code_len = int(code.length());

				unsigned int codeint = utils::toUnsignedInt(code);
				global_to_local[j][vertex_id] = { codeint, code_len };
				local_to_global[i][j][code] = vertex_id;
				code_size += code.length() * 0.000000125f + 0.000004f;
			}
			total_Sequence += common_id[i][j];
			total_Sequence_size += common_id[i][j] * common_bit_width[i][j];
		}
	}
	total_code_num = total_Sequence;
	average_code_size += total_Sequence_size;
	average_code_size /= (double)total_code_num;
	gan_usage += code_size;


	double index_size = 0;
	if (myapp->mytype == pagerank || myapp->mytype == personalpagerank)
		index_size = 0.0000005f;  // 4bit
	else if (myapp->mytype == hits)
	{
		index_size = 0.00000025f;  // 2bit
	}
	else if(myapp->mytype == salsa)
		index_size = 0.000000625f;  // 5bit

    if(type == synchronous){  
        if(myapp->ITERATIONS != 0){
            /* converge within the fixed number of iterations */
            iter_counter = myapp->ITERATIONS;
			double noise_budget_all = 0;
			double noise_budget_iter;  
			noise_budget_iter = noise_budget / (double)myapp->ITERATIONS;  
			float noise_budget_cent = noise_budget_iter * cent_bgt_percent; 
			float noise_budget_pair = noise_budget_iter - noise_budget_cent;
			noise_budget_cent /= (double)num_threads;  		

			int num_noisy_dc_pair = 0; 
			for (int dc1 = 0; dc1 < num_threads; dc1++)
			{
				for (int dc2 = 0; dc2 < num_threads; dc2++)
				{
					if (dc1 == dc2)
						continue;
					num_noisy_dc_pair++;
				}
			}
			noise_budget_pair /= num_noisy_dc_pair;
			

			if (privacy)
				cout << "Differential Privacy is on running." << endl;

			if (myapp->mytype == hits || myapp->mytype == salsa)
			{
				noise_budget_pair /= 2;
				noise_budget_cent /= 2;
			}

			vector<float> step_withNoise(AGG_NUM_withNoise + 1);
			vector<float> privacy_budgets(AGG_NUM_withNoise);
			vector<float> intervals(AGG_NUM_withNoise + 1);
			float sum = utils::feibonaqi_sum(AGG_NUM_withNoise); 
			for (int index = 1; index <= AGG_NUM_withNoise; index++)
			{
				double percent = utils::feibonaqi_n(0, 0, (double)(index)) / sum;  
				step_withNoise[index] = step_withNoise[index - 1] + 1 * percent;
				privacy_budgets[AGG_NUM_withNoise - index] = percent * noise_budget_pair;  
			}

			PR_MAX = std::numeric_limits<int>::max();
			PR_MIN = std::numeric_limits<int>::min();
			
			// local DP k-means module
			int kmeans_num = num_threads;
			if (myapp->mytype == hits || myapp->mytype == salsa)
				kmeans_num *= 2;
			vector<kmeans*> localkmeans(kmeans_num);
			for (int tr = 0; tr < num_threads; tr++)
			{
				localkmeans[tr] = new kmeans(ncluster, tr);
				if (myapp->mytype == hits || myapp->mytype == salsa)
					localkmeans[num_threads + tr] = new kmeans(ncluster, num_threads + tr);
			}

			while (iter_counter > 0) {

				/**
				* Execute Compute on all active vertices
				* Sync before sending messages
				*/
				std::cout << "-------------- Compute stage of iteration " << myapp->ITERATIONS - iter_counter << " -------------" << std::endl;
				

				hits_a_norm = 0.0;
				hits_h_norm = 0.0;
				vector<double> hits_a_each(total_vertices);
				vector<double> hits_h_each(total_vertices);
				for (int tr = 0; tr < num_threads; tr++) {
					if ((myapp->ITERATIONS - iter_counter) == 0 && (myapp->mytype == hits || myapp->mytype == salsa))
					{
						continue;
					}
                    #pragma omp parallel for
					for (int v = 0; v < Threads[tr]->l_dag->num_vertices; v++) {
						int vid = (Threads[tr]->l_dag->g)->myvertex[v]->vertex_id;
						vector<double> hits_norm;  
						hits_norm = myapp->Compute(vid, myapp->global_graph);
						if (myapp->mytype == hits) {
							hits_a_each[vid] = pow(hits_norm[0], 2);
							hits_h_each[vid] = pow(hits_norm[1], 2);
						}
						vector<double>().swap(hits_norm);
					}
				}
				if (myapp->mytype == hits) {
					for (int vi = 0; vi < total_vertices; vi++)
					{
						hits_a_norm += hits_a_each[vi];
						hits_h_norm += hits_h_each[vi];
					}
					vector<double>().swap(hits_a_each);
					vector<double>().swap(hits_h_each);
					hits_a_norm = std::sqrt(hits_a_norm);
					hits_h_norm = std::sqrt(hits_h_norm);
					if (hits_a_norm == 0)
					{
						hits_a_norm = sqrt((myapp->global_graph->g)->num_vertices);
					}
					if (hits_h_norm == 0)
					{
						hits_h_norm = sqrt((myapp->global_graph->g)->num_vertices);
					}
					cout << "After compute, hits_a_norm = " << hits_a_norm << ", hits_h_norm = " << hits_h_norm << endl;

				}
				
				vector<int> out_msg_indexes(num_threads, 0);  
				vector<int> in_msg_indexes(num_threads, 0);
				vector<vector<pair<int, float>>> out_msgdatas(num_threads);
				vector<vector<pair<int, float>>> in_msgdatas(num_threads);
				vector<float> out_ranges(num_threads);
				vector<float> in_ranges(num_threads);
				for (int i = 0; i < num_threads; i++)
					out_msgdatas[i].resize(Inter_out_msg_counts[i]);

				if (myapp->mytype == hits || myapp->mytype == salsa) {
					for (int i = 0; i < num_threads; i++)
						in_msgdatas[i].resize(Inter_in_msg_counts[i]);					
				}
				time_t start_msg = time(NULL);

                #pragma omp parallel for  
				for (int tr = 0; tr < num_threads; tr++) {                    
					for (int v = 0; v < Threads[tr]->l_dag->num_vertices; v++) { 
						int vid = (Threads[tr]->l_dag->g)->myvertex[v]->vertex_id;
						vector<int> out_nbrs = (myapp->global_graph->g)->myvertex[vid]->out_neighbour;
						vector<int> in_nbrs = (myapp->global_graph->g)->myvertex[vid]->in_neighbour;
						double msg_rank, msg_authoriy, pr_rank, ppr_rank, authority, hub;
						if (myapp->mytype == pagerank) {
							pr_rank = dynamic_cast<PageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank;
							msg_rank = pr_rank / out_nbrs.size();
						}
						else if (myapp->mytype == personalpagerank) {
							ppr_rank = dynamic_cast<PersonalPageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank;
							msg_rank = ppr_rank / out_nbrs.size();
						}
						else if (myapp->mytype == hits) {
							msg_rank = dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->h / hits_h_norm + 1;
							msg_authoriy = dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->a / hits_a_norm + 1;
						}
						else if (myapp->mytype == salsa) {
							hub = dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->hub;
							msg_rank = hub / out_nbrs.size();
							authority = dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->authority;
							msg_authoriy = authority / in_nbrs.size();
						}
						for (int nid = 0; nid < out_nbrs.size(); nid++) 
						{
							int other_dc = (myapp->global_graph->g)->myvertex[out_nbrs[nid]]->location_id; 
							if (other_dc != tr)
							{
								out_msgdatas[tr][out_msg_indexes[tr]] = { vid, msg_rank };
								out_msg_indexes[tr]++;
							}
						}
						if (myapp->mytype == hits || myapp->mytype == salsa) {
							for (int nid = 0; nid < in_nbrs.size(); nid++) 
							{
								int other_dc = (myapp->global_graph->g)->myvertex[in_nbrs[nid]]->location_id; 
								if (other_dc != tr)
								{
									in_msgdatas[tr][in_msg_indexes[tr]] = { vid, msg_authoriy };
									in_msg_indexes[tr]++;
								}
							}
						}
						vector<int>().swap(out_nbrs);
						vector<int>().swap(in_nbrs);
					}
				}	
								
				float msg_lowerbound = 0;
				if (myapp->mytype == hits)
				{
					msg_lowerbound = 1;
					msg_upperbound = 2;
				}
				else if (myapp->mytype == salsa)
					msg_upperbound = 1;

				float global_sensitivity = msg_upperbound - msg_lowerbound;
				cout << "All messages are clipped at [" << msg_lowerbound << ", " << msg_upperbound << "]" << endl;
				cout << "Global sensitivity: " << global_sensitivity << endl;

				for (int index = 0; index < step_withNoise.size(); index++) {
					intervals[index] = msg_lowerbound + step_withNoise[index] * global_sensitivity;
				}
								
				gan_usage += kmeans_num * (kmeans_num - 1) * (ncluster * (0.000004f) + 0.000020f);
				
				omp_set_nested(1);
                #pragma omp parallel for  
				for (int tr = 0; tr < kmeans_num; tr++) {  
					if (tr < num_threads)
						localkmeans[tr]->set_pcount(out_msgdatas[tr].size(), myapp->ITERATIONS - iter_counter);
					else
						localkmeans[tr]->set_pcount(in_msgdatas[tr - num_threads].size(), myapp->ITERATIONS - iter_counter);

					if (privacy)
						if (tr < num_threads)
							localkmeans[tr]->Kmeans("", out_msgdatas[tr], noise_budget_cent, msg_upperbound, msg_lowerbound);
						else
							localkmeans[tr]->Kmeans("", in_msgdatas[tr - num_threads], noise_budget_cent, msg_upperbound, msg_lowerbound);
					else
						if (tr < num_threads)
							localkmeans[tr]->Kmeans("", out_msgdatas[tr], 0, msg_upperbound, msg_lowerbound);
						else
							localkmeans[tr]->Kmeans("", in_msgdatas[tr - num_threads], 0, msg_upperbound, msg_lowerbound);						
				}

				vector<vector<vector<int>>> out_agg_indexes(num_threads);
				vector<vector<vector<int>>> in_agg_indexes(num_threads);

				// combiner module
				for (int tr = 0; tr < num_threads; tr++) {	
					int agg = 0;
					out_agg_indexes[tr].resize(num_threads);
					for (int other_dc = 0; other_dc < num_threads; other_dc++) {						
						if (tr != other_dc) {
							out_agg_indexes[tr][other_dc].resize(AGG_NUM_withNoise);
							DCs[tr]->out_k_aggregators[other_dc].resize(AGG_NUM_withNoise);
							for (int si = 0; si < AGG_NUM_withNoise; si++)  
							{
								VertexData* l_accum;
								if (myapp->mytype == pagerank)
									l_accum = new PageRankVertexData();
								else if (myapp->mytype == personalpagerank)
									l_accum = new PersonalPageRankVertexData();
								else if (myapp->mytype == hits)
									l_accum = new HITSVertexData();
								else if(myapp->mytype == salsa)
									l_accum = new salsaVertexData();
								Aggregator* new_agg = new Aggregator();
								new_agg->DC_id = other_dc;
								new_agg->aggregated_data = l_accum;
								DCs[tr]->out_k_aggregators[other_dc][si] = new_agg;
								agg++;
								DCs[tr]->out_k_aggregators[other_dc][si]->noise_budget = privacy_budgets[si];
							}
						}
					}

					if (myapp->mytype == hits || myapp->mytype == salsa) {
						in_agg_indexes[tr].resize(num_threads);
						for (int other_dc = 0; other_dc < num_threads; other_dc++) {
							if (tr != other_dc) {
								in_agg_indexes[tr][other_dc].resize(AGG_NUM_withNoise);
								DCs[tr]->in_k_aggregators[other_dc].resize(AGG_NUM_withNoise);
								for (int si = 0; si < AGG_NUM_withNoise; si++)  
								{
									VertexData* l_accum;
									if (myapp->mytype == hits)
										l_accum = new HITSVertexData();
									else if (myapp->mytype == salsa)
										l_accum = new salsaVertexData();
									Aggregator* new_agg = new Aggregator();
									new_agg->DC_id = other_dc;
									new_agg->aggregated_data = l_accum;
									DCs[tr]->in_k_aggregators[other_dc][si] = new_agg;
									agg++;
									DCs[tr]->in_k_aggregators[other_dc][si]->noise_budget = privacy_budgets[si];
								}
							}
						}
					}
					cout << "DC " << tr << " has total " << agg << std::endl;
				}
				
				omp_set_nested(1);
                #pragma omp parallel for 
				for (int tr = 0; tr < num_threads; tr++) {  
                    #pragma omp parallel for 
					for (int v = 0; v < Threads[tr]->l_dag->num_vertices; v++) { 
						int vid = (Threads[tr]->l_dag->g)->myvertex[v]->vertex_id;
						std::vector<int> out_nbrs = (myapp->global_graph->g)->myvertex[vid]->out_neighbour;
						std::vector<int> in_nbrs = (myapp->global_graph->g)->myvertex[vid]->in_neighbour;
						double msg_rank, msg_rank_1, msg_hub, msg_authority, receive_msg_a, receive_msg_h;
						if (myapp->mytype == pagerank) 
							msg_rank = dynamic_cast<PageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank / (double)out_nbrs.size();													
						else if (myapp->mytype == personalpagerank)
							msg_rank = dynamic_cast<PersonalPageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank / (double)out_nbrs.size();
						else if (myapp->mytype == hits)
						{
							msg_hub = dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->h / hits_h_norm + 1;
							msg_authority = dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->a / hits_a_norm + 1;
						}
						else if (myapp->mytype == salsa)
						{
							msg_hub = dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->hub / (double)(myapp->global_graph->g)->myvertex[vid]->outSize;
							msg_authority = dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->authority / (double)(myapp->global_graph->g)->myvertex[vid]->inSize;
						}						
						for (int nid = 0; nid < out_nbrs.size(); nid++) { 
							VertexData* l_accum;
							if (myapp->mytype == pagerank)
							{
								l_accum = new PageRankVertexData();
								dynamic_cast<PageRankVertexData*>(l_accum)->rank = msg_rank;
							}
							else if (myapp->mytype == personalpagerank)
							{
								l_accum = new PersonalPageRankVertexData();
								dynamic_cast<PersonalPageRankVertexData*>(l_accum)->rank = msg_rank;
							}
							else if (myapp->mytype == hits)
							{
								l_accum = new HITSVertexData();
								dynamic_cast<HITSVertexData*>(l_accum)->h = msg_hub;
							}
							else if (myapp->mytype == salsa)
							{
								l_accum = new salsaVertexData();
								dynamic_cast<salsaVertexData*>(l_accum)->hub = msg_hub;
							}
							int other_dc = (myapp->global_graph->g)->myvertex[out_nbrs[nid]]->location_id;
							if (other_dc != tr) 
							{
								float new_msg = msg_rank;
								if(myapp->mytype == hits || myapp->mytype == salsa)
									new_msg = msg_hub;
								// clip the msg
								if (new_msg > msg_upperbound)
									new_msg = msg_upperbound;
								if (new_msg < msg_lowerbound)
									new_msg = msg_lowerbound;
								if (privacy) {
									int agg_index = 0;
									for (int si = 0; si < AGG_NUM_withNoise - 1; si++)
									{
										if (new_msg >= intervals[si] && new_msg < intervals[si + 1]) {
											agg_index = si;
											break;
										}
									}
									if (new_msg >= intervals[AGG_NUM_withNoise - 1]) {
										agg_index = AGG_NUM_withNoise - 1;
									}
									int lock_index = tr * num_threads * num_of_DC_combiner + other_dc * num_of_DC_combiner + agg_index;
									omp_set_lock(&lock_out_combiner[lock_index]);
									DCs[tr]->out_k_aggregators[other_dc][agg_index]->num_list.push_back(global_to_local[tr][out_nbrs[nid]]);

									out_agg_indexes[tr][other_dc][agg_index]++;
									if(myapp->mytype == pagerank)
										dynamic_cast<PageRankVertexData*>(DCs[tr]->out_k_aggregators[other_dc][agg_index]->aggregated_data)->rank += new_msg;
									else if(myapp->mytype == personalpagerank)
										dynamic_cast<PersonalPageRankVertexData*>(DCs[tr]->out_k_aggregators[other_dc][agg_index]->aggregated_data)->rank += new_msg;
									else if (myapp->mytype == hits)
										dynamic_cast<HITSVertexData*>(DCs[tr]->out_k_aggregators[other_dc][agg_index]->aggregated_data)->h += new_msg;
									else if (myapp->mytype == salsa)
										dynamic_cast<salsaVertexData*>(DCs[tr]->out_k_aggregators[other_dc][agg_index]->aggregated_data)->hub += new_msg;
									omp_unset_lock(&lock_out_combiner[lock_index]);
									delete l_accum;
									l_accum = NULL;
									continue; 
								}

								if (myapp->mytype == pagerank) 
									dynamic_cast<PageRankVertexData*>(l_accum)->rank = new_msg;
								else if(myapp->mytype == personalpagerank)
									dynamic_cast<PersonalPageRankVertexData*>(l_accum)->rank = new_msg;
								else if (myapp->mytype == hits)
									dynamic_cast<HITSVertexData*>(l_accum)->h = new_msg;
								else if (myapp->mytype == salsa)
									dynamic_cast<salsaVertexData*>(l_accum)->hub = new_msg;
								omp_set_lock(&lock_vertex[out_nbrs[nid]]); 
								(myapp->global_graph->g)->myvertex[out_nbrs[nid]]->messages_sendout_h.push_back(l_accum); 
								omp_unset_lock(&lock_vertex[out_nbrs[nid]]); 

							}
							else
							{
								omp_set_lock(&lock_vertex[out_nbrs[nid]]);
								(myapp->global_graph->g)->myvertex[out_nbrs[nid]]->messages_sendout_h.push_back(l_accum); 
								omp_unset_lock(&lock_vertex[out_nbrs[nid]]);
							}						
						}

						if (myapp->mytype == hits || myapp->mytype == salsa)
						{
							for (int nid = 0; nid < in_nbrs.size(); nid++) {  
								VertexData* l_accum;
								if (myapp->mytype == hits)
								{
									l_accum = new HITSVertexData();
									dynamic_cast<HITSVertexData*>(l_accum)->a = msg_authority;
								}
								else if (myapp->mytype == salsa)
								{
									l_accum = new salsaVertexData();
									dynamic_cast<salsaVertexData*>(l_accum)->authority = msg_authority;
								}
								int other_dc = (myapp->global_graph->g)->myvertex[in_nbrs[nid]]->location_id;
								if (other_dc != tr)  
								{
									float new_msg = msg_rank_1;
									if (myapp->mytype == hits || myapp->mytype == salsa)
										new_msg = msg_authority;
									// clip the msg
									if (new_msg > msg_upperbound)
										new_msg = msg_upperbound;
									if (new_msg < msg_lowerbound)
										new_msg = msg_lowerbound;

									if (privacy)
									{
										int agg_index = 0;
										for (int si = 0; si < AGG_NUM_withNoise - 1; si++)
										{
											if (new_msg >= intervals[si] && new_msg < intervals[si + 1]) {
												agg_index = si;
												break;
											}
										}
										if (new_msg >= intervals[AGG_NUM_withNoise - 1]) {
											agg_index = AGG_NUM_withNoise - 1;
										}
										int lock_index = tr * num_threads * num_of_DC_combiner + other_dc * num_of_DC_combiner + agg_index;
										omp_set_lock(&lock_in_combiner[lock_index]);
										DCs[tr]->in_k_aggregators[other_dc][agg_index]->num_list.push_back(global_to_local[tr][in_nbrs[nid]]);
										in_agg_indexes[tr][other_dc][agg_index]++;
										if (myapp->mytype == hits)
											dynamic_cast<HITSVertexData*>(DCs[tr]->in_k_aggregators[other_dc][agg_index]->aggregated_data)->a += new_msg;
										else if (myapp->mytype == salsa)
											dynamic_cast<salsaVertexData*>(DCs[tr]->in_k_aggregators[other_dc][agg_index]->aggregated_data)->authority += new_msg;
										omp_unset_lock(&lock_in_combiner[lock_index]);
										delete l_accum;
										l_accum = NULL;
										continue; 
									}
									omp_set_lock(&lock_vertex[in_nbrs[nid]]);
									(myapp->global_graph->g)->myvertex[in_nbrs[nid]]->messages_sendin_a.push_back(l_accum);  
									omp_unset_lock(&lock_vertex[in_nbrs[nid]]);
								}
								else
								{
									omp_set_lock(&lock_vertex[in_nbrs[nid]]);
									(myapp->global_graph->g)->myvertex[in_nbrs[nid]]->messages_sendin_a.push_back(l_accum); 
									omp_unset_lock(&lock_vertex[in_nbrs[nid]]);
								}
							}
						}

						//clear
						vector<int>().swap(out_nbrs);
						vector<int>().swap(in_nbrs);
					}										
				}
								
				vector<vector<vector<int>>> index_list(kmeans_num);
				for (int i = 0; i < kmeans_num; i++)
				{
					index_list[i].resize(total_vertices);
					for (int j = 0; j < total_vertices; j++)
					{
						index_list[i][j].resize(ncluster);
					}
				}

				if (privacy) {
					for (int tr = 0; tr < num_threads; tr++) {
						map<int, vector<Aggregator*>>::iterator iter_gg;
						for (iter_gg = DCs[tr]->out_k_aggregators.begin(); iter_gg != DCs[tr]->out_k_aggregators.end(); iter_gg++) {
							for (int si = 0; si < iter_gg->second.size(); si++)
							{
								// add noise
								float msg_counts_combiner = out_agg_indexes[tr][iter_gg->first][si];
								float noise = myapp->global_graph->laplace_generator(0, global_sensitivity / (DCs[tr]->out_k_aggregators[iter_gg->first][si]->noise_budget) / 2);
								float noise_count = myapp->global_graph->laplace_generator(0, global_sensitivity / (DCs[tr]->out_k_aggregators[iter_gg->first][si]->noise_budget) / 2);
								float sum_msg;
								if (myapp->mytype == pagerank)
									sum_msg = dynamic_cast<PageRankVertexData*>(DCs[tr]->out_k_aggregators[iter_gg->first][si]->aggregated_data)->rank;
								else if (myapp->mytype == personalpagerank)
									sum_msg = dynamic_cast<PersonalPageRankVertexData*>(DCs[tr]->out_k_aggregators[iter_gg->first][si]->aggregated_data)->rank;
								else if (myapp->mytype == hits)
									sum_msg = dynamic_cast<HITSVertexData*>(DCs[tr]->out_k_aggregators[iter_gg->first][si]->aggregated_data)->h;
								else if (myapp->mytype == salsa)
									sum_msg = dynamic_cast<salsaVertexData*>(DCs[tr]->out_k_aggregators[iter_gg->first][si]->aggregated_data)->hub;
								sum_msg += noise;
								msg_counts_combiner += noise_count;
								float avg_msg = sum_msg / msg_counts_combiner;

								// dequantize
								kmeans::point p = (kmeans::point_t*)malloc(sizeof(kmeans::point_t));
								p->group = 0;
								p->x = avg_msg;
								int group = localkmeans[tr]->nearest(p, ncluster, 0); 
								delete p;

								float new_msg = localkmeans[tr]->cent[group].x;
								// distribute the message
								int vertex_counts = out_agg_indexes[tr][iter_gg->first][si];								
                                #pragma omp parallel for  
								for (long int nid = 0; nid < vertex_counts; nid++) {

									pair<unsigned int, int> codepair = DCs[tr]->out_k_aggregators[iter_gg->first][si]->num_list[nid];
									string other_vertex_string = utils::toBinaryString(codepair.first, codepair.second);
									int other_vertex = local_to_global[iter_gg->first][tr][other_vertex_string]; 
									VertexData* l_accum;
									if (myapp->mytype == pagerank)
									{
										l_accum = new PageRankVertexData();
										dynamic_cast<PageRankVertexData*>(l_accum)->rank = new_msg;
										omp_set_lock(&lock_vertex[other_vertex]);
										(myapp->global_graph->g)->myvertex[other_vertex]->messages_sendout_h.push_back(l_accum);

										index_list[tr][other_vertex][group]++;

										omp_unset_lock(&lock_vertex[other_vertex]);
									}
									else if (myapp->mytype == personalpagerank) {
										l_accum = new PersonalPageRankVertexData();
										dynamic_cast<PersonalPageRankVertexData*>(l_accum)->rank = new_msg;
										omp_set_lock(&lock_vertex[other_vertex]);
										(myapp->global_graph->g)->myvertex[other_vertex]->messages_sendout_h.push_back(l_accum);
										index_list[tr][other_vertex][group]++;
										omp_unset_lock(&lock_vertex[other_vertex]);
									}
									else if (myapp->mytype == hits) {
										l_accum = new HITSVertexData();
										dynamic_cast<HITSVertexData*>(l_accum)->h = new_msg;
										omp_set_lock(&lock_vertex[other_vertex]);
										(myapp->global_graph->g)->myvertex[other_vertex]->messages_sendout_h.push_back(l_accum);
										index_list[tr][other_vertex][group]++;
										omp_unset_lock(&lock_vertex[other_vertex]);
									}
									else if (myapp->mytype == salsa) {
										l_accum = new salsaVertexData();
										dynamic_cast<salsaVertexData*>(l_accum)->hub = new_msg;
										omp_set_lock(&lock_vertex[other_vertex]);
										(myapp->global_graph->g)->myvertex[other_vertex]->messages_sendout_h.push_back(l_accum);
										index_list[tr][other_vertex][group]++;
										omp_unset_lock(&lock_vertex[other_vertex]);
									}
								}
							}
						}
					}
					if (myapp->mytype == hits || myapp->mytype == salsa) {
						for (int tr = 0; tr < num_threads; tr++) {
							map<int, vector<Aggregator*>>::iterator iter_gg;
							for (iter_gg = DCs[tr]->in_k_aggregators.begin(); iter_gg != DCs[tr]->in_k_aggregators.end(); iter_gg++) {
								for (int si = 0; si < iter_gg->second.size(); si++)
								{
									// add noise
									float msg_counts_combiner = in_agg_indexes[tr][iter_gg->first][si];
									float noise = myapp->global_graph->laplace_generator(0, global_sensitivity / (DCs[tr]->in_k_aggregators[iter_gg->first][si]->noise_budget) / 2);
									float noise_count = myapp->global_graph->laplace_generator(0, global_sensitivity / (DCs[tr]->in_k_aggregators[iter_gg->first][si]->noise_budget) / 2);									// 分发给对应DC的各个顶点
									float sum_msg;
									if (myapp->mytype == hits)
										sum_msg = dynamic_cast<HITSVertexData*>(DCs[tr]->in_k_aggregators[iter_gg->first][si]->aggregated_data)->a;
									else if (myapp->mytype == salsa)
										sum_msg = dynamic_cast<salsaVertexData*>(DCs[tr]->in_k_aggregators[iter_gg->first][si]->aggregated_data)->authority;
									sum_msg += noise;
									msg_counts_combiner += noise_count;
									float avg_msg = sum_msg / msg_counts_combiner;

									// dequantize
									kmeans::point p = (kmeans::point_t*)malloc(sizeof(kmeans::point_t));
									p->group = 0;
									p->x = avg_msg;
									int group = localkmeans[tr + num_threads]->nearest(p, ncluster, 0); 
									delete p;

									float new_msg = localkmeans[tr + num_threads]->cent[group].x;
		
									// distribute the message
									int vertex_counts = in_agg_indexes[tr][iter_gg->first][si];
                                    #pragma omp parallel for 
									for (long int nid = 0; nid < vertex_counts; nid++) {
										pair<unsigned int, int> codepair = DCs[tr]->in_k_aggregators[iter_gg->first][si]->num_list[nid];
										string other_vertex_string = utils::toBinaryString(codepair.first, codepair.second);
										int other_vertex = local_to_global[iter_gg->first][tr][other_vertex_string]; 
										VertexData* l_accum;
										if (myapp->mytype == hits) {
											l_accum = new HITSVertexData();
											dynamic_cast<HITSVertexData*>(l_accum)->a = new_msg;
											omp_set_lock(&lock_vertex[other_vertex]);
											(myapp->global_graph->g)->myvertex[other_vertex]->messages_sendin_a.push_back(l_accum);
											index_list[tr + num_threads][other_vertex][group]++; 
											omp_unset_lock(&lock_vertex[other_vertex]);
										}
										else if (myapp->mytype == salsa) {
											l_accum = new salsaVertexData();
											dynamic_cast<salsaVertexData*>(l_accum)->authority = new_msg;
											omp_set_lock(&lock_vertex[other_vertex]);
											(myapp->global_graph->g)->myvertex[other_vertex]->messages_sendin_a.push_back(l_accum);
											index_list[tr + num_threads][other_vertex][group]++;  
											omp_unset_lock(&lock_vertex[other_vertex]);
										}
									}
								}								
							}
						}
					}
				}

				// calculate gan usage
				for (int i = 0; i < numofdcs; i++)
				{
					for (int vi = 0; vi < total_vertices; vi++)
					{
						bool send_status = false;
						for (int j = 0; j < ncluster; j++)
						{
							if (index_list[i][vi][j] != 0)
							{
								gan_usage += index_size + 0.000003f;  // index + count	
								send_status = true;
							}
						}
						if (myapp->mytype == hits || myapp->mytype == salsa)
						{
							for (int j = 0; j < ncluster; j++)
							{
								if (index_list[i + numofdcs][vi][j] != 0)
								{
									gan_usage += index_size + 0.000003f; // index + count
									send_status = true;
								}
							}
						}
						if(send_status) 
							gan_usage += 0.000020f;
					}
				}		
				iter_counter--;

				if (myapp->mytype == pagerank)
				{
						double avg_rank = 0;
						for (int vi = 0; vi < total_vertices; vi++) {
							avg_rank += dynamic_cast<PageRankVertexData*>((myapp->global_graph->g)->myvertex[vi]->data)->rank;
						}
						avg_rank /= (double)total_vertices;
						printf("iter %d: average rank %.4f\n", myapp->ITERATIONS - iter_counter, avg_rank);
						rank_.push_back(avg_rank);
				}
				else if (myapp->mytype == personalpagerank)
				{
						double avg_rank = 0;
						for (int vi = 0; vi < total_vertices; vi++) {
							avg_rank += dynamic_cast<PersonalPageRankVertexData*>((myapp->global_graph->g)->myvertex[vi]->data)->rank;
						}
						avg_rank /= (double)total_vertices;
						printf("iter %d: average rank %.4f\n", myapp->ITERATIONS - iter_counter, avg_rank);
						rank_.push_back(avg_rank);
				}
				if (myapp->mytype == hits)
				{
					double avg_a = 0, avg_h = 0;
					for (int vi = 0; vi < total_vertices; vi++) {
						avg_a += ((dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vi]->data)->a) / hits_a_norm + 1);
						avg_h += ((dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vi]->data)->h) / hits_h_norm + 1);
					}
					avg_a /= (double)total_vertices;
					avg_h /= (double)total_vertices;
					printf("iter %d: average authority %.4f\n", myapp->ITERATIONS - iter_counter, avg_a);
					printf("iter %d: average hub %.4f\n", myapp->ITERATIONS - iter_counter, avg_h);
					rank_a.push_back(avg_a);
					rank_h.push_back(avg_h);						
				}
				else if (myapp->mytype == salsa)
				{
					double avg_a = 0, avg_h = 0;
					for (int vi = 0; vi < total_vertices; vi++) {
						avg_a += dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vi]->data)->authority;
						avg_h += dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vi]->data)->hub;
					}
					avg_a /= (double)total_vertices;
					avg_h /= (double)total_vertices;
					printf("iter %d: average authority %.4f\n", myapp->ITERATIONS - iter_counter, avg_a);
					printf("iter %d: average hub %.4f\n", myapp->ITERATIONS - iter_counter, avg_h);
					rank_a.push_back(avg_a);
					rank_h.push_back(avg_h);
				}
				// clear
				for (int tr = 0; tr < num_threads; tr++) {                     
					map<int, vector<Aggregator*>>::iterator iter_gg;
					for (iter_gg = DCs[tr]->out_k_aggregators.begin(); iter_gg != DCs[tr]->out_k_aggregators.end(); iter_gg++) {
						for (int si = 0; si < iter_gg->second.size(); si++) {
							delete DCs[tr]->out_k_aggregators[iter_gg->first][si]->aggregated_data;
							DCs[tr]->out_k_aggregators[iter_gg->first][si]->aggregated_data = NULL;
							vector<pair<unsigned int, int>>().swap(DCs[tr]->out_k_aggregators[iter_gg->first][si]->num_list);
							delete DCs[tr]->out_k_aggregators[iter_gg->first][si];
							DCs[tr]->out_k_aggregators[iter_gg->first][si] = NULL;
						}
						vector<Aggregator*>().swap(DCs[tr]->out_k_aggregators[iter_gg->first]);
					}
					map<int, vector<Aggregator*>>().swap(DCs[tr]->out_k_aggregators);
				}
				for (int i = 0; i < num_threads; i++)
				{
					vector<pair<int, float>>().swap(out_msgdatas[i]);
					for (int j = 0; j < num_threads; j++)
						vector<int>().swap(out_agg_indexes[i][j]);
					vector<vector<int>>().swap(out_agg_indexes[i]);
				}
				vector<vector<pair<int, float>>>().swap(out_msgdatas);
				vector<vector<vector<int>>>().swap(out_agg_indexes);
				if (myapp->mytype == hits || myapp->mytype == salsa) {
					for (int tr = 0; tr < num_threads; tr++) {
						map<int, vector<Aggregator*>>::iterator iter_gg;
						for (iter_gg = DCs[tr]->in_k_aggregators.begin(); iter_gg != DCs[tr]->in_k_aggregators.end(); iter_gg++) {
							for (int si = 0; si < iter_gg->second.size(); si++) {
								delete DCs[tr]->in_k_aggregators[iter_gg->first][si]->aggregated_data;
								DCs[tr]->in_k_aggregators[iter_gg->first][si]->aggregated_data = NULL;
								vector<pair<unsigned int, int>>().swap(DCs[tr]->in_k_aggregators[iter_gg->first][si]->num_list);
								delete DCs[tr]->in_k_aggregators[iter_gg->first][si];
								DCs[tr]->in_k_aggregators[iter_gg->first][si] = NULL;
							}
							vector<Aggregator*>().swap(DCs[tr]->in_k_aggregators[iter_gg->first]);
						}
						map<int, vector<Aggregator*>>().swap(DCs[tr]->in_k_aggregators);
					}
					for (int i = 0; i < num_threads; i++)
					{
						vector<pair<int, float>>().swap(in_msgdatas[i]);
						for (int j = 0; j < num_threads; j++)
							vector<int>().swap(in_agg_indexes[i][j]);
						vector<vector<int>>().swap(in_agg_indexes[i]);
					}
					vector<vector<pair<int, float>>>().swap(in_msgdatas);
					vector<vector<vector<int>>>().swap(in_agg_indexes);
				}			
			}
        }   
		else 
		{
            /* converge according to the tolerance */
            iter_counter ++;
        }
    }//end of pagerank

	if (myapp->mytype == pagerank)
	{
		printf("rank value distribution：\n");
		for (int i = 0; i < rank_.size(); i++)
			std::cout << rank_[i] << std::endl;
	}
	else if (myapp->mytype == personalpagerank)
	{
		printf("rank value distribution:\n");
		for (int i = 0; i < rank_.size(); i++)		
			std::cout << rank_[i] << std::endl;		
	}
	else if (myapp->mytype == hits)
	{
		printf("authority value distribution：\n");
		for (int i = 0; i < rank_a.size(); i++)
			std::cout << rank_a[i] << std::endl;
		printf("hub value distribution：\n");
		for (int i = 0; i < rank_h.size(); i++)
			std::cout << rank_h[i] << std::endl;
	}
	else if (myapp->mytype == salsa)
	{
		printf("authority value distribution：\n");
		for (int i = 0; i < rank_a.size(); i++)
			std::cout << rank_a[i] << std::endl;
		printf("hub value distribution：\n");
		for (int i = 0; i < rank_h.size(); i++)
			std::cout << rank_h[i] << std::endl;
	}
	printf("-------------------------------------------------------\n");

	cout << "gan_usage = " << gan_usage << endl;

	if (true)
	{
		//calculate the AP of all the rank application
		vector<double> tp = { 0.0001, 0.001, 0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8 };
		vector<string> tp_file = { "0.01%", "0.1%", "1%", "2%", "5%", "10%", "20%", "40%", "60%", "80%" };
        #pragma omp parallel for  
		for (int ic = 0; ic < tp.size(); ic++)
		{
			double top_percent = tp[ic];
			double start = 1.0, end = 2.0, midle = 0;
			double min = std::numeric_limits<double>::max(), max = -std::numeric_limits<double>::max();  //find the max and min value of all vertices
			double min_h = std::numeric_limits<double>::max(), max_h = -std::numeric_limits<double>::max();  //find the max and min value of all vertices
			int a_count = 0, h_count = 0;
			map<int, double> data_a, data_h;
			map<int, double> save_a, save_h;
			int top_k = (int)(top_percent * atoi(sizeofgraph));

			for (int v = 0; v < total_vertices; v++) {
				if (myapp->mytype == hits)
				{
					data_a[v] = abs(((dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[v]->data)->a) / hits_a_norm + 1));
					data_h[v] = abs(((dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[v]->data)->h) / hits_h_norm + 1));
					min = min < data_a[v] ? min : data_a[v];
					max = max > data_a[v] ? max : data_a[v];
					min_h = min_h < data_h[v] ? min_h : data_h[v];
					max_h = max_h > data_h[v] ? max_h : data_h[v];
				}
				else if (myapp->mytype == salsa)
				{
					data_a[v] = abs(dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[v]->data)->authority);
					data_h[v] = abs(dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[v]->data)->hub);
					min = min < data_a[v] ? min : data_a[v];
					max = max > data_a[v] ? max : data_a[v];
					min_h = min_h < data_h[v] ? min_h : data_h[v];
					max_h = max_h > data_h[v] ? max_h : data_h[v];
				}
				else
				{
					if (myapp->mytype == pagerank)
						data_a[v] = abs(dynamic_cast<PageRankVertexData*>((myapp->global_graph->g)->myvertex[v]->data)->rank);
					else if (myapp->mytype == personalpagerank)
						data_a[v] = abs(dynamic_cast<PersonalPageRankVertexData*>((myapp->global_graph->g)->myvertex[v]->data)->rank);

					min = min < data_a[v] ? min : data_a[v];
					max = max > data_a[v] ? max : data_a[v];
				}
			}

			start = min;
			end = max;
			midle = (start + end) / 2;
							
			printf("expected %d vertices.\n", top_k);
			printf("vertices value interval: [%f,%f]\n", start, end);

			while (a_count != top_k)
			{
				if (a_count < top_k) 
				{
					std::map<int, double>::iterator iter;
					std::map<int, double> need_change;
					for (iter = data_a.begin(); iter != data_a.end(); iter++) {
						if ((double)iter->second >= midle)
						{
							need_change[iter->first] = iter->second;
						}
					}

					std::map<double, int> is_same;
					is_same.clear();
					for (iter = need_change.begin(); iter != need_change.end(); iter++)
					{
						is_same[iter->second] = iter->first;
					}
					if (is_same.size() == 1)  
					{
						iter = need_change.begin();
						int num = abs(top_k - a_count);
						for (int i = 0; i < num; i++)
						{
							if (i >= need_change.size())
								break;
							save_a[iter->first] = iter->second;
							data_a.erase(iter->first); 
							a_count++;
							iter++;
						}
					}
					else
					{
						for (iter = need_change.begin(); iter != need_change.end(); iter++)
						{
							save_a[iter->first] = iter->second;
							data_a.erase(iter->first); 
							a_count++;
						}
					}
					need_change.clear();
				}

				else if (a_count > top_k) 
				{
					std::map<int, double>::iterator iter;
					std::map<int, double> need_change;
					for (iter = save_a.begin(); iter != save_a.end(); iter++) {
						if ((double)iter->second <= midle)
						{
							need_change[iter->first] = iter->second;
						}
					}
					std::map<double, int> is_same;
					is_same.clear();
					for (iter = need_change.begin(); iter != need_change.end(); iter++)
					{
						is_same[iter->second] = iter->first;
					}
					if (is_same.size() == 1) 
					{
						iter = need_change.begin();
						int num = abs(top_k - a_count);
						for (int i = 0; i < num; i++)
						{
							if (i >= need_change.size())
								break;
							data_a[iter->first] = iter->second;
							save_a.erase(iter->first);  
							a_count--;
							iter++;
						}
					}
					else
					{
						for (iter = need_change.begin(); iter != need_change.end(); iter++)
						{
							data_a[iter->first] = iter->second;
							save_a.erase(iter->first);  
							a_count--;
						}
					}
					need_change.clear();

				}
				if (a_count > top_k)
				{
					start = midle;
					midle = (start + end) / 2.0;
				}
				else if (a_count < top_k)
				{
					end = midle;
					midle = (end + start) / 2.0;
				}

				std::cout << "a_count = " << a_count << ", The midle = " << midle << std::endl;
				if (a_count != (int)save_a.size())
				{
					printf("a_count != save_a.size()\n");
					int a;
					scanf("%d", &a);
				}
			}
			if (myapp->mytype == hits || myapp->mytype == salsa) {
				start = min_h; end = max_h; midle = (start + end) / 2;
				while (h_count != top_k)
				{
					if (h_count < top_k)  
					{
						std::map<int, double> need_change;
						std::map<int, double>::iterator iter;
						for (iter = data_h.begin(); iter != data_h.end(); iter++)
						{
							if ((double)iter->second >= midle)
							{
								need_change[iter->first] = iter->second;
							}
						}
						std::map<double, int> is_same;
						is_same.clear();
						for (iter = need_change.begin(); iter != need_change.end(); iter++)
						{
							is_same[iter->second] = iter->first;
						}
						if (is_same.size() == 1)  
						{
							iter = need_change.begin();
							int num = abs(top_k - h_count);
							for (int i = 0; i < num; i++)
							{
								if (i >= need_change.size())
									break;
								save_h[iter->first] = iter->second;
								data_h.erase(iter->first); 
								h_count++;
								iter++;
							}
						}
						else
						{
							for (iter = need_change.begin(); iter != need_change.end(); iter++)
							{
								save_h[iter->first] = iter->second;
								data_h.erase(iter->first); 
								h_count++;
							}
						}
						need_change.clear();

					}
					else if (h_count > top_k)  
					{
						std::map<int, double> need_change;
						std::map<int, double>::iterator iter;
						for (iter = save_h.begin(); iter != save_h.end(); iter++) {
							if ((double)iter->second <= midle)
							{
								need_change[iter->first] = iter->second;

							}
						}

						std::map<double, int> is_same;
						is_same.clear();
						for (iter = need_change.begin(); iter != need_change.end(); iter++)
						{
							is_same[iter->second] = iter->first;
						}
						if (is_same.size() == 1) 
						{
							iter = need_change.begin();
							int num = abs(top_k - h_count);
							for (int i = 0; i < num; i++)
							{
								if (i >= need_change.size())
									break;
								data_h[iter->first] = iter->second;
								save_h.erase(iter->first);  
								h_count--;
								iter++;
							}
						}
						else
						{
							for (iter = need_change.begin(); iter != need_change.end(); iter++)
							{
								data_h[iter->first] = iter->second;
								save_h.erase(iter->first);  
								h_count--;
							}
						}
						need_change.clear();
					}


					if (h_count > top_k)
					{
						start = midle;
						midle = (start + end) / 2.0;
					}
					else if (h_count < top_k)
					{
						end = midle;
						midle = (end + start) / 2.0;
					}
					std::cout << "h_count = " << h_count << ", The midle = " << midle << std::endl;
					if (h_count != (int)save_h.size())
					{
						printf("h_count != save_h.size()\n");
						int a;
						scanf("%d", &a);
					}
				}
			}
			cout << "Find top k done. Begin to sort the top k." << endl;
			map<int, double>::iterator iter;
			vector<pair<int, double>> a_vec(save_a.size()), h_vec(save_h.size());
			int i = 0;
			for (iter = save_a.begin(); iter != save_a.end(); iter++) {
				a_vec[i] = (make_pair(int(iter->first), double(iter->second)));
				i++;
			}

			if (myapp->mytype == hits || myapp->mytype == salsa)
			{
				i = 0;
				for (iter = save_h.begin(); iter != save_h.end(); iter++) {
					h_vec[i] = (make_pair(int(iter->first), double(iter->second)));
					i++;
				}
			}

			string datafile;
			map<int, int> baseline, baseline_h;
			string datafile_1;
			if (myapp->mytype == hits)
				datafile = graphfile + "/hits_";
			else if (myapp->mytype == salsa)
				datafile = graphfile + "/salsa_";
			else if (myapp->mytype == pagerank)
				datafile = graphfile + "/pagerank_";
			else if (myapp->mytype == personalpagerank)
				datafile = graphfile + "/personalpagerank_";
			datafile += sizeofgraph;
			datafile_1 += datafile;
			datafile += "_a_top";
			datafile_1 += "_h_top";
			datafile += tp_file[ic];
			datafile_1 += tp_file[ic];
			std::cout << "file name:" << datafile << std::endl;
			std::cout << "file name:" << datafile_1 << std::endl;
			std::ifstream non_privacy(datafile);
			std::ifstream non_privacy_1(datafile_1);

			if (non_privacy.is_open() == false || (non_privacy_1.is_open() == false && myapp->mytype == hits) || (non_privacy_1.is_open() == false && myapp->mytype == salsa))
				std::cout << "file is not exist." << std::endl;
			else {
				std::string rndline = "0";
				std::string rndline_1 = "0";
				int loc, loc_1;
				while (1) {
					if (rndline.empty() == true)
						break;
					std::getline(non_privacy, rndline);
					std::stringstream strm(rndline);  
					strm >> loc;
					baseline[loc] = loc;
				}
				cout << "read a value's baseline from file done." << endl;
				if (myapp->mytype == hits || myapp->mytype == salsa)
				{
					while (1) {
						if (rndline_1.empty() == true)
							break;
						std::getline(non_privacy_1, rndline_1);
						std::stringstream strm_1(rndline_1); 
						strm_1 >> loc_1;
						baseline_h[loc_1] = loc_1;
					}
					cout << "read h value's baseline from file done." << endl;
				}
			}


			int ap_right = 0;
			for (int i = 0; i < a_vec.size(); i++)
			{
				if (baseline.find(a_vec[i].first) != baseline.end())  
				{
					ap_right++;
				}
			}
			printf("P = %f\n", (double)ap_right / (double)top_k);
			accuracy[ic][myapp->ITERATIONS - 1] = (double)ap_right / (double)top_k;

			if (myapp->mytype == hits || myapp->mytype == salsa)
			{
				int ap_right_h = 0;
				for (int i = 0; i < h_vec.size(); i++)
				{
					if (baseline_h.find(h_vec[i].first) != baseline_h.end())  
					{
						ap_right_h++;
					}
				}
				printf("P_h = %f\n", (double)ap_right_h / (double)top_k);
				accuracy_h[ic][myapp->ITERATIONS - 1] = (double)ap_right_h / (double)top_k;
			}
		}

		cout << "#################################################" << endl;
		cout << "rank/authority:" << endl;
		for (int i = 0; i < accuracy.size(); i++)
		{
			cout << accuracy[i][accuracy[i].size() - 1] << ",";
		}
		cout << endl;
		if (myapp->mytype == hits || myapp->mytype == salsa) {
			cout << "hub:" << endl;
			for (int i = 0; i < accuracy_h.size(); i++)
			{
				cout << accuracy_h[i][accuracy_h[i].size() - 1] << ",";
			}
		}
		cout << endl << "#################################################" << endl;
		cout << "rank/authority:" << endl;
		for (int i = 0; i < accuracy.size(); i++)
		{
			cout << tp_file[i] << ": ";
			if (i < 3) {
				for (int j = 0; j < myapp->ITERATIONS && j < accuracy[i].size(); j++)
				{
					cout << accuracy[i][j] << ",";
				}
			}
			else {
				cout << accuracy[i][accuracy[i].size() - 1];
			}
			cout << endl;
		}
		if (myapp->mytype == hits || myapp->mytype == salsa) {
			cout << "hub:" << endl;
			for (int i = 0; i < accuracy_h.size(); i++)
			{
				cout << tp_file[i] << ": ";
				if (i < 3) {
					for (int j = 0; j < myapp->ITERATIONS && j < accuracy_h[i].size(); j++)
					{
						cout << accuracy_h[i][j] << ",";
					}
				}
				else {
					cout << accuracy_h[i][accuracy_h[i].size() - 1];
				}
				cout << endl;
			}
		}
		cout << endl << "#################################################" << endl;

	}

	time_t now = time(NULL);
	strftime(str, 50, "%x %X", localtime(&now));
	cout << str << endl;
}