#include "../src/Simulator.h"
#include "../utils/Utils.h"
#include <omp.h>
#include <string>
#include <algorithm> 
#include <iostream>
#include <fstream>  
#include <sstream>

/*
	Created by Qiulin Wu on 2024-12-02.
*/

extern int BASELINE;
extern int N_THREADS; 
extern float noise_budget;
extern bool privacy;

extern double PR_MAX;
extern double PR_MIN;

extern string assgin_mode;
extern string graphfile; 

void GraphEngine::Pregel(char* sizeofgraph){
	int total_vertices = (myapp->global_graph->g)->num_vertices;
	const int num_vertex_locks = total_vertices;
	omp_lock_t* lock_vertex = new omp_lock_t[num_vertex_locks];

	for (int i = 0; i < num_vertex_locks; i++)
		omp_init_lock(&lock_vertex[i]);

    /**
    * Clear memory.
    */
    Threads.clear();
    for(int i=0; i< num_threads; i++){
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
	 
	for (int tr = 0; tr < num_threads; tr++) {
        /**
        * Signal vertices
        */
		for (int v = 0; v < Threads[tr]->l_dag->num_vertices; v++) {
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
        }
    }

    /** Start the engine, each thread executes at the same time */
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

    if(type == synchronous){   
        if(myapp->ITERATIONS != 0){
            /* converge within the fixed number of iterations */
            iter_counter = myapp->ITERATIONS;

			double noise_budget_all; 
			int need_allocated = 0;
			noise_budget_all = noise_budget / (double)myapp->ITERATIONS; 
			for (int tr = 0; tr < num_threads; tr++) {  
				for (int v = 0; v < Threads[tr]->l_dag->num_vertices; v++) {
					int vid = (Threads[tr]->l_dag->g)->myvertex[v]->vertex_id;
					std::vector<int> out_nbrs = (myapp->global_graph->g)->myvertex[vid]->out_neighbour;
					for (int nid = 0; nid < (myapp->global_graph->g)->myvertex[vid]->outSize; nid++) {  
						int other_dc = (myapp->global_graph->g)->myvertex[out_nbrs[nid]]->location_id;
						if (other_dc != tr ) 
							need_allocated++;
					}
					if (myapp->mytype == salsa || myapp->mytype == hits)
					{
						std::vector<int> in_nbrs = (myapp->global_graph->g)->myvertex[vid]->in_neighbour;
						for (int nid = 0; nid < (myapp->global_graph->g)->myvertex[vid]->inSize; nid++) { 
							int other_dc = (myapp->global_graph->g)->myvertex[in_nbrs[nid]]->location_id;
							if (other_dc != tr)  
								need_allocated++;
						}
					}
				}
			}

			noise_budget_all = noise_budget_all / (double)need_allocated;  
			cout << "noise_budget_all = " << noise_budget_all << endl;

			if (BASELINE)  
			{
				PR_MAX = std::numeric_limits<int>::max();
				PR_MIN = std::numeric_limits<int>::min();
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

				P_MAX = 0.0;
				Q_MAX = 0.0;
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
					} 
				}
				if (myapp->mytype == hits) {
					for (int vi = 0; vi < total_vertices; vi++)
					{
						hits_a_norm += hits_a_each[vi];
						hits_h_norm += hits_h_each[vi];
					}
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

				vector<double> vertex_gan_usage(total_vertices);
				for (int tr = 0; tr < num_threads; tr++) { 

					double deltaf = 0; 
					if (myapp->mytype == pagerank)
						deltaf = PR_MAX - PR_MIN;
					else if (myapp->mytype == personalpagerank)					
						deltaf = PR_MAX - 0;					
					else if (myapp->mytype == hits)
						deltaf = 1;					
					else if (myapp->mytype == salsa)					
						deltaf = 1;
					
					printf("Gf is: %f\n", deltaf);

					
					#pragma omp parallel for  
					for (int v = 0; v < Threads[tr]->l_dag->num_vertices; v++) {  
						int vid = (Threads[tr]->l_dag->g)->myvertex[v]->vertex_id;
						std::vector<int> out_nbrs = (myapp->global_graph->g)->myvertex[vid]->out_neighbour;
						std::vector<int> in_nbrs = (myapp->global_graph->g)->myvertex[vid]->in_neighbour;
						double msg_rank, msg_rank_1;
						if (myapp->mytype == pagerank)
							msg_rank = dynamic_cast<PageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank / (double)out_nbrs.size();
						else if (myapp->mytype == personalpagerank)
							msg_rank = dynamic_cast<PersonalPageRankVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->rank / (double)out_nbrs.size();
						else if (myapp->mytype == hits)
						{
							msg_rank = dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->h;
							msg_rank_1 = dynamic_cast<HITSVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->a;
						}
						else if (myapp->mytype == salsa)
						{
							msg_rank = dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->hub / (double)(myapp->global_graph->g)->myvertex[vid]->outSize;
							msg_rank_1 = dynamic_cast<salsaVertexData*>((myapp->global_graph->g)->myvertex[vid]->data)->authority / (double)(myapp->global_graph->g)->myvertex[vid]->inSize;
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
								dynamic_cast<HITSVertexData*>(l_accum)->h = msg_rank / hits_h_norm + 1;
							}
							else if (myapp->mytype == salsa)
							{
								l_accum = new salsaVertexData();
								dynamic_cast<salsaVertexData*>(l_accum)->hub = msg_rank;
							}
							int other_dc = (myapp->global_graph->g)->myvertex[out_nbrs[nid]]->location_id;
							if (other_dc != tr)  
							{
								vertex_gan_usage[vid] += (0.000004f + 0.000004f + 0.000020f); 
								if (privacy)
								{
									double noise = myapp->global_graph->laplace_generator(0, deltaf / noise_budget_all);
									if (myapp->mytype == pagerank)																		
										dynamic_cast<PageRankVertexData*>(l_accum)->rank += noise;									
									else if (myapp->mytype == personalpagerank)									
										dynamic_cast<PersonalPageRankVertexData*>(l_accum)->rank += noise;
									else if (myapp->mytype == hits)									
										dynamic_cast<HITSVertexData*>(l_accum)->h += noise;
									else if (myapp->mytype == salsa)									
										dynamic_cast<salsaVertexData*>(l_accum)->hub += noise;
								}								
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
									dynamic_cast<HITSVertexData*>(l_accum)->a = msg_rank_1 / hits_a_norm + 1;
								}
								else if (myapp->mytype == salsa)
								{
									l_accum = new salsaVertexData();
									dynamic_cast<salsaVertexData*>(l_accum)->authority = msg_rank_1;
								}
								int other_dc = (myapp->global_graph->g)->myvertex[in_nbrs[nid]]->location_id;
								if (other_dc != tr)  
								{
									vertex_gan_usage[vid] += (0.000004f + 0.000004f + 0.000020f);  
									if (privacy)
									{
										double noise = myapp->global_graph->laplace_generator(0, deltaf / noise_budget_all);
										if (myapp->mytype == hits)									
											dynamic_cast<HITSVertexData*>(l_accum)->a += noise;										
										else if (myapp->mytype == salsa)										
											dynamic_cast<salsaVertexData*>(l_accum)->authority += noise;
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
					}

				}
				for (int vi = 0; vi < total_vertices; vi++) 
					gan_usage += vertex_gan_usage[vi];

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
				else if (myapp->mytype == hits)
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
		printf("rank value distribution£º\n");
		for (int i = 0; i < rank_.size(); i++)
			std::cout << rank_[i] << std::endl;
	}
	else if (myapp->mytype == personalpagerank)
	{
		printf("rank value distribution:\n");
		for (int i = 0; i < rank_.size(); i++)
		{
			std::cout << rank_[i] << std::endl;
		}
	}
	else if (myapp->mytype == hits)
	{
		printf("authority value distribution£º\n");
		for (int i = 0; i < rank_a.size(); i++)
			std::cout << rank_a[i] << std::endl;
		printf("hub value distribution£º\n");
		for (int i = 0; i < rank_h.size(); i++)
			std::cout << rank_h[i] << std::endl;
	}
	else if (myapp->mytype == salsa)
	{
		printf("authority value distribution£º\n");
		for (int i = 0; i < rank_a.size(); i++)
			std::cout << rank_a[i] << std::endl;
		printf("hub value distribution£º\n");
		for (int i = 0; i < rank_h.size(); i++)
			std::cout << rank_h[i] << std::endl;
	}
	printf("-------------------------------------------------------\n");



	map<int, double> dc0;
	map<int, double> dc0_h;

	std::cout << "begin to output vertex's value." << std::endl;
	if (myapp->mytype == pagerank) {

		string datafile;
		if (BASELINE)   
		{
			datafile = graphfile + "/" + assgin_mode + "/pagerank_";
			datafile += sizeofgraph;
			datafile += ".txt";
		}
		
		ofstream OutFile_1(datafile); 

		for (int i = 0; i < num_threads; i++) {
			for (int v = 0; v < Threads[i]->l_dag->num_vertices; v++) {
				if ((Threads[i]->l_dag->g)->myvertex[v]->is_master)
				{
					dc0[(Threads[i]->l_dag->g)->myvertex[v]->vertex_id] = (dynamic_cast<PageRankVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->rank);					
					if (BASELINE)  
						OutFile_1 << (Threads[i]->l_dag->g)->myvertex[v]->vertex_id << " " << dynamic_cast<PageRankVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->rank << std::endl;
				}
			}
		} 
		OutFile_1.close();
		printf("output pagerank value done.\n");
	}
	else if (myapp->mytype == personalpagerank) {
		string datafile;
		if (BASELINE)   
		{
			datafile = graphfile + "/" + assgin_mode + "/personalpagerank_";
			datafile += sizeofgraph;
			datafile += ".txt";
		}
		ofstream OutFile_1(datafile);

		for (int i = 0; i < num_threads; i++) {
			for (int v = 0; v < Threads[i]->l_dag->num_vertices; v++) {
				if ((Threads[i]->l_dag->g)->myvertex[v]->is_master)
				{
					dc0[(Threads[i]->l_dag->g)->myvertex[v]->vertex_id] = (dynamic_cast<PersonalPageRankVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->rank);
					if (BASELINE)
						OutFile_1 << (Threads[i]->l_dag->g)->myvertex[v]->vertex_id << " " << dynamic_cast<PersonalPageRankVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->rank << endl;
				} 
			} 
		} 
		printf("output personalpagerank value done.\n");
	}
	else if (myapp->mytype == hits || myapp->mytype == salsa) {

		string datafile;
		if (BASELINE)   
		{

			if (myapp->mytype == hits)
				datafile = graphfile + "/" + assgin_mode + "/hits_";
			else if (myapp->mytype == salsa)
				datafile = graphfile + "/" + assgin_mode + "/salsa_";
			datafile += sizeofgraph;
			datafile += "_a.txt";
		}
		ofstream OutFile_2(datafile); 

		for (int i = 0; i < num_threads; i++) {
			for (int v = 0; v < Threads[i]->l_dag->num_vertices; v++) {
				if ((Threads[i]->l_dag->g)->myvertex[v]->is_master)
				{
					if (myapp->mytype == hits)
					{
						dc0[(Threads[i]->l_dag->g)->myvertex[v]->vertex_id] = ((dynamic_cast<HITSVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->a) / hits_a_norm + 1);
						if (BASELINE)   
							OutFile_2 << (Threads[i]->l_dag->g)->myvertex[v]->vertex_id << " " << (dynamic_cast<HITSVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->a) / hits_a_norm + 1 << std::endl;
					}
					else if (myapp->mytype == salsa)
					{
						dc0[(Threads[i]->l_dag->g)->myvertex[v]->vertex_id] = ((dynamic_cast<salsaVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->authority));
						if (BASELINE)   
							OutFile_2 << (Threads[i]->l_dag->g)->myvertex[v]->vertex_id << " " << dynamic_cast<salsaVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->authority << std::endl;
					}
				}
			}
		}
		OutFile_2.close();
		if (myapp->mytype == hits)
			printf("output HITS's authority value done.\n");
		else if (myapp->mytype == salsa)
			printf("output salsa value done.\n");


		if (myapp->mytype == hits || myapp->mytype == salsa)  
		{
			string datafile;
			if (BASELINE)   
			{
				if (myapp->mytype == hits)
					datafile = graphfile + "/" + assgin_mode + "/hits_";
				else if (myapp->mytype == salsa)
					datafile = graphfile + "/" + assgin_mode + "/salsa_";
				datafile += sizeofgraph;
				datafile += "_h.txt";
			}
			ofstream OutFile_3(datafile); 


			for (int i = 0; i < num_threads; i++) {
				for (int v = 0; v < Threads[i]->l_dag->num_vertices; v++) {
					if ((Threads[i]->l_dag->g)->myvertex[v]->is_master)
					{
						if (myapp->mytype == hits)
						{
							dc0_h[(Threads[i]->l_dag->g)->myvertex[v]->vertex_id]=((dynamic_cast<HITSVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->h) / hits_h_norm + 1);
							if (BASELINE)  
								OutFile_3 << (Threads[i]->l_dag->g)->myvertex[v]->vertex_id << " " << (dynamic_cast<HITSVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->h) / hits_h_norm + 1 << std::endl;
						}
						else if (myapp->mytype == salsa)
						{
							dc0_h[(Threads[i]->l_dag->g)->myvertex[v]->vertex_id]=((dynamic_cast<salsaVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->hub));
							if (BASELINE)  
								OutFile_3 << (Threads[i]->l_dag->g)->myvertex[v]->vertex_id << " " << dynamic_cast<salsaVertexData*>((Threads[i]->l_dag->g)->myvertex[v]->data)->hub << std::endl;
						}
					}
				}
			}
			OutFile_3.close();
			if (myapp->mytype == hits)
				printf("output HITS's hub value done.\n");
			else if (myapp->mytype == salsa)
				printf("output salsa's hub value done.\n");
		}
	}

	cout << "gan_usage=" << gan_usage << endl;
	cout << "msg_size=" << 0.000004f << std::endl;

	if (true)
	{
		//calculate the AP of all the rank application
		vector<double> tp = { 0.0001, 0.001, 0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8};
		vector<string> tp_file = { "0.01%", "0.1%", "1%", "2%", "5%", "10%", "20%", "40%", "60%", "80%"};
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
			if (BASELINE)  
			{
				if (myapp->mytype == pagerank)
					datafile = graphfile + "/" + assgin_mode + "/pagerank_";
				else if (myapp->mytype == personalpagerank)
					datafile = graphfile + "/" + assgin_mode + "/personalpagerank_";
				else if (myapp->mytype == hits)
					datafile = graphfile + "/" + assgin_mode + "/hits_";
				else if (myapp->mytype == salsa)
					datafile = graphfile + "/" + assgin_mode + "/salsa_";
				datafile += sizeofgraph;
				datafile += "_a_top" + tp_file[ic];
			}
			cout << "open file:" << datafile << endl;
			ofstream OutFile_4(datafile);
			std::cout << "begin to output top " << tp_file[ic] << "% vertices list." << std::endl;
			for (int i = 0; i < a_vec.size(); i++)
			{
				if (BASELINE)  
					OutFile_4 << (int)(a_vec[i].first) << std::endl;
			}
			OutFile_4.close();
			cout << "output top " << tp_file[ic] << "% vertices' authority/rank value done." << endl;
			if (BASELINE)  
			{
				if (myapp->mytype == pagerank)
					datafile = graphfile + "/" + assgin_mode + "/pagerank_";
				else if (myapp->mytype == personalpagerank)
					datafile = graphfile + "/" + assgin_mode + "/personalpagerank_";
				else if (myapp->mytype == hits)
					datafile = graphfile + "/" + assgin_mode + "/hits_";
				else if (myapp->mytype == salsa)
					datafile = graphfile + "/" + assgin_mode + "/salsa_";
				datafile += sizeofgraph;
				datafile += "_h_top" + tp_file[ic];
			}
			ofstream OutFile_5(datafile); 
			if (myapp->mytype == hits || myapp->mytype == salsa)
			{
				for (int i = 0; i < h_vec.size(); i++)
				{
					if (BASELINE) 
						OutFile_5 << (int)(h_vec[i].first) << std::endl;
				}
				OutFile_5.close();
				cout << "output top " << tp_file[ic] << "% vertices' hub value done." << endl;
			}

			map<int, int> baseline, baseline_h;
			//string datafile;
			string datafile_1;
			if (myapp->mytype == hits)
				datafile = graphfile + "/" + assgin_mode + "/hits_";
			else if (myapp->mytype == salsa)
				datafile = graphfile + "/" + assgin_mode + "/salsa_";
			else if (myapp->mytype == pagerank)
				datafile = graphfile + "/" + assgin_mode + "/pagerank_";
			else if (myapp->mytype == personalpagerank)
				datafile = graphfile + "/" + assgin_mode + "/personalpagerank_";
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
	std::cout << str << std::endl;
}