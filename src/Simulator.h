#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "Apps.h"
#include <iostream>
#include <unordered_map>
#include <algorithm>

/*
	Created by Qiulin Wu on 2024-12-02.
*/

class Optimizer{
public:	
	BaseApp* myapp;	
	distributed_graph* dag;
	std::vector<DataCenter*> DCs;	
};

enum EngineType{
	synchronous,
	asynchronous,
	numofengine
};
	
 class GraphEngine{
	
public:
	EngineType type;
	int num_threads; //number of threads
	BaseApp* myapp;	
	Optimizer* myopt;
	std::vector<DataCenter*> DCs;
	
	class Thread{
	public:
		distributed_graph* l_dag;
		int DC_loc_id;
	};
	
	std::vector<Thread* > Threads;
	
	GraphEngine(EngineType t, int n){
		type = t; num_threads = n; 		
	}
	void Pregel(const char*);
	void CFDGraph(const char*);

};
#endif