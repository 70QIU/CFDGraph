#ifndef APPS_H
#define APPS_H
#include <limits>
#include "Cloud_instance.h"
#include "Simulator.h"
#include "../utils/Utils.h"
#include <omp.h>
#endif

/*
	Created by Qiulin Wu on 2024-12-02.
*/

extern int numofdcs;
extern bool privacy;
extern double PR_MAX;
extern double PR_MIN;

// Application Type
enum AppType{
	pagerank,
	sssp,
	subgraph,
	typesofapps,
	hits,
	leaderrank,
	personalpagerank,
	labelpropagation,
	salsa,
	als
};
// Base application
class BaseApp {
public:

	AppType mytype;
	distributed_graph* global_graph;
	float budget;
	int ITERATIONS;
	float msg_size;
	virtual vector<double> Compute(int, distributed_graph*) = 0;

};
// PageRank application
class PageRank:public BaseApp{
public:
	/* The message passing model in CFDGraph */
	vector<double> Compute(int vindex, distributed_graph* dag){
	    int numofmsg = (dag->g)->myvertex[vindex]->messages_sendout_h.size();
	    PageRankVertexData* sum = new PageRankVertexData();
        sum->rank = 0;
		for (int m = 0; m < numofmsg; m++) {
            sum->rank += dynamic_cast<PageRankVertexData*>((dag->g)->myvertex[vindex]->messages_sendout_h[m])->rank;
        }
		
		float newvalue = RESET_PROB + (1 - RESET_PROB) * sum->rank;
		delete sum;
		sum = NULL;
		//bound the PR value
		if (newvalue > PR_MAX)
			newvalue = PR_MAX;
		else if(newvalue < PR_MIN)
			newvalue = PR_MIN;

		float temp = 1;
		temp = dynamic_cast<PageRankVertexData*>((dag->g)->myvertex[vindex]->data)->rank;

		float last_change = abs((newvalue - temp) / temp);
		dynamic_cast<PageRankVertexData*>((dag->g)->myvertex[vindex]->data)->last_change = last_change;
		dynamic_cast<PageRankVertexData*>((dag->g)->myvertex[vindex]->data)->rank = newvalue;
		//clear message queue
		for(int m=0; m<numofmsg; m++){
			delete (dag->g)->myvertex[vindex]->messages_sendout_h[m];
			(dag->g)->myvertex[vindex]->messages_sendout_h[m] = NULL;
		}
		(dag->g)->myvertex[vindex]->messages_sendout_h.clear();

		vector<double> result;
		result.push_back(dynamic_cast<PageRankVertexData*>((dag->g)->myvertex[vindex]->data)->rank);
		result.push_back(-1);
		return result;
	}
	PageRank(){
		msg_size = 0.000008f;
	}
};

// Personalized PageRank application
class PersonalPageRank :public BaseApp {
public:
   /* The message passing model in CFDGraph */
	vector<double> Compute(int vindex, distributed_graph* dag) {
		int numofmsg = (dag->g)->myvertex[vindex]->messages_sendout_h.size();
		PersonalPageRankVertexData* sum = new PersonalPageRankVertexData();
		sum->rank = 0;
		for (int m = 0; m < numofmsg; m++) 
			sum->rank += dynamic_cast<PersonalPageRankVertexData*>((dag->g)->myvertex[vindex]->messages_sendout_h[m])->rank;

		float newvalue;
		if ((dag->g)->myvertex[vindex]->isChoose)  // if is the choosed vertices
			newvalue = RESET_PROB + (1 - RESET_PROB) * sum->rank;
		else
			newvalue = (1 - RESET_PROB) * sum->rank;

		//bound the PPR value
		if (newvalue > PR_MAX)
			newvalue = PR_MAX;
		else if (newvalue < 0)
			newvalue = 0;
		delete sum;
		sum = NULL;
		float temp = 1;
		if (abs(dynamic_cast<PersonalPageRankVertexData*>((dag->g)->myvertex[vindex]->data)->rank) > 0.00001)
			temp = dynamic_cast<PersonalPageRankVertexData*>((dag->g)->myvertex[vindex]->data)->rank;
		float last_change = abs((newvalue - dynamic_cast<PersonalPageRankVertexData*>((dag->g)->myvertex[vindex]->data)->rank) / (float)temp);
		dynamic_cast<PersonalPageRankVertexData*>((dag->g)->myvertex[vindex]->data)->last_change = last_change;
		dynamic_cast<PersonalPageRankVertexData*>((dag->g)->myvertex[vindex]->data)->rank = newvalue;
		//clear message queue
		for (int m = 0; m < numofmsg; m++) {
			delete (dag->g)->myvertex[vindex]->messages_sendout_h[m];
			(dag->g)->myvertex[vindex]->messages_sendout_h[m] = NULL;
		}
		(dag->g)->myvertex[vindex]->messages_sendout_h.clear();

		vector<double> result;
		result.push_back(-1);
		result.push_back(-2);
		return result;
	}

	PersonalPageRank() {
		msg_size = 0.000008f;
	}
};

// HITS application
class HITS :public BaseApp {
public:
   /* The message passing model in CFDGraph */
	vector<double> Compute(int vindex, distributed_graph* dag) {
		(dag->g)->myvertex[vindex]->status = (vertex_status)deactivated;
		int numofmsg_h = (dag->g)->myvertex[vindex]->messages_sendout_h.size();
		int numofmsg_a = (dag->g)->myvertex[vindex]->messages_sendin_a.size();
		HITSVertexData* sum = new HITSVertexData();
		sum->h = 0.0;
		sum->a = 0.0;
		for (int m = 0; m < numofmsg_h; m++) {
			sum->h += dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->messages_sendout_h[m])->h;
		}
		for (int m = 0; m < numofmsg_a; m++) {
			sum->a += dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->messages_sendin_a[m])->a;
		}

		float newvalue_a = sum->a;
		float newvalue_h = sum->h;
		delete sum;
		sum = NULL;

		float temp = 1;
		if (dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->a != 0.0)
			temp = dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->a;
		else
			temp = 0.000001;
		float last_change_a = abs((newvalue_a - temp) / temp);

		if (dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->h != 0.0)
			temp = dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->h;
		else
			temp = 0.000001;
		float last_change_h = abs((newvalue_h - temp) / temp);

		dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->last_change_a = last_change_a;
		dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->last_change_h = last_change_h;
		//update
		dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->h = newvalue_a;
		dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->a = newvalue_h;
		
		
		//clear message queue
		for (int m = 0; m < numofmsg_h; m++) {
			delete (dag->g)->myvertex[vindex]->messages_sendout_h[m];
			(dag->g)->myvertex[vindex]->messages_sendout_h[m] = NULL;
		}
		for (int m = 0; m < numofmsg_a; m++) {
			delete (dag->g)->myvertex[vindex]->messages_sendin_a[m];
			(dag->g)->myvertex[vindex]->messages_sendin_a[m] = NULL;
		}
		(dag->g)->myvertex[vindex]->messages_sendout_h.clear();
		(dag->g)->myvertex[vindex]->messages_sendin_a.clear();

		vector<double> result;
		result.push_back(dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->a);
		result.push_back(dynamic_cast<HITSVertexData*>((dag->g)->myvertex[vindex]->data)->h);
		return result;
	}

	HITS() {
		msg_size = 0.000008f;
	}
};

// Salsa application
class Salsa :public BaseApp {
public:
	/* The message passing model in CFDGraph */
	vector<double> Compute(int vindex, distributed_graph* dag) {
		int numofmsg_out_h = (dag->g)->myvertex[vindex]->messages_sendout_h.size();
		int numofmsg_in_a = (dag->g)->myvertex[vindex]->messages_sendin_a.size();
		salsaVertexData* sum = new salsaVertexData();
		sum->hub = 0.0;
		sum->authority = 0.0;

		for (int m = 0; m < numofmsg_out_h; m++)
			sum->hub += dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->messages_sendout_h[m])->hub;

		for (int m = 0; m < numofmsg_in_a; m++)
			sum->authority += dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->messages_sendin_a[m])->authority;

		float newvalue_a = sum->hub;
		float newvalue_h = sum->authority;

		float temp = 1;

		if (dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->authority != 0.0)
			temp = dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->authority;
		else
			temp = 0.000001;
		float last_change_a = abs((newvalue_a - temp) / temp);

		if (dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->hub != 0.0)
			temp = dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->hub;
		else
			temp = 0.000001;
		float last_change_h = abs((newvalue_h - temp) / temp);

		dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->last_change_a = last_change_a;
		dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->last_change_h = last_change_h;
		//update
		dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->authority = sum->hub;
		dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->hub = sum->authority;


		delete sum;
		sum = NULL;

		//clear message queue
		for (int m = 0; m < numofmsg_out_h; m++) {
			delete (dag->g)->myvertex[vindex]->messages_sendout_h[m];
			(dag->g)->myvertex[vindex]->messages_sendout_h[m] = NULL;
		}
		for (int m = 0; m < numofmsg_in_a; m++) {
			delete (dag->g)->myvertex[vindex]->messages_sendin_a[m];
			(dag->g)->myvertex[vindex]->messages_sendin_a[m] = NULL;
		}

		(dag->g)->myvertex[vindex]->messages_sendout_h.clear();
		(dag->g)->myvertex[vindex]->messages_sendin_a.clear();

		vector<double> result;
		result.push_back(dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->authority);
		result.push_back(dynamic_cast<salsaVertexData*>((dag->g)->myvertex[vindex]->data)->hub);
		return result;
	}
	Salsa() {
		msg_size = 0.000008f;
	}
};