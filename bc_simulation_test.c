#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// from SYMPHONY's headers
#include "compute_cost.h"
#include "vrp_common_types.h"
#include "vrp_const.h"
#include "vrp_io.h"
#include "vrp_macros.h"
#include "vrp_types.h"
#include "sym_macros.h"
#include "sym_proto.h"
#include "sym_types.h"
#include "vrp_lp_params.h"
#include "vrp_cg_params.h"


#include "uct_with_cws.h"
#include "cws.h"
#include "uct_const.h"
#include "my_lib.h"

int CUSTOMER_SIZE; //!< 倉庫(depot)と顧客(customer)を合わせた数 
int VEHICLE_SIZE; //!< 車体(vehicle)の数
int CAPACITY; //!< 車体(vehicle)の容量
int ROUTE_MAX; // = CUSTOMER_SIZE * VEHICLE_SIZE;

int CHILD_MAX;// = CUSTOMER_SIZE; //!< NODE構造体のメンバchild配列の最大要素数。

//! 各車体(vehicle)が走行したルート。
/*! 倉庫(depot)を意味する値0をこの配列に代入する事は無い。\n
 *  0番目の車体(vehicle)が2番目に訪れた顧客(customer)を表すにはroute[CUSTOMER_SIZE * 0 + 2]とする
 */
int *route;
int *route_size; //!< 各routeに属している顧客の数
int *current_cap; //!< 各車体(vehicle)が訪れた顧客の要求(demand)の合計。1番目の車体(current_vehicle=1)が訪れた顧客の要求の合計を表すにはcurrent_cap[1]とする


vrp_problem *vrp;

CWS_NODE *node;
int num_nodes;

S_LIST *sl;

//! グローバル変数を初期化する関数
/*!
 * CUSTOMER_SIZE, CAPACITY, VEHICLE_SIZE, ROUTE_MAX, route, current_capを初期化する。\n
 * route, current_capはcallocしているので、必ずfree_global_var()でメモリを解放すること。
 */
void initialize_global_var(vrp_problem *vrp) {
	char *vsp; // vehicle sizeを表す文字列へのポインタ

	/********************************/
	/* Insert some global valuables */
	/********************************/
	CUSTOMER_SIZE = vrp->vertnum;
	CAPACITY      = vrp->capacity;

	vsp = strchr(vrp->name, 'k');
	vsp = vsp + 1;
	VEHICLE_SIZE = atoi(vsp);

	ROUTE_MAX = CUSTOMER_SIZE * VEHICLE_SIZE;
	CHILD_MAX = vrp->edgenum;

	route       = (int *)ec_calloc(ROUTE_MAX, sizeof(int));
	route_size  = (int *)ec_calloc(VEHICLE_SIZE, sizeof(int));
	current_cap = (int *)ec_calloc(VEHICLE_SIZE, sizeof(int));

	node  = (CWS_NODE *)ec_calloc(NODE_MAX, sizeof(CWS_NODE));
}

//! initialize_global_varで割り当てたメモリを解放する関数
/*!
 * route, current_capに割り当てられたメモリを解放する
 */
void free_global_var() {
	free(route);
	free(route_size);
	free(current_cap);
	free(node);
}


int main(int argc, char **argv) {
	int best_savings_idx;
	int best_distance, current_distance;
	int *best_route;
	int *best_cap;
	time_t start_time, current_time;
	S_LIST *sl_cpy;
	int rest;
	int error;
	int distance;
	int count;

	char infile_name[LENGTH];
	char paper_problem[33][LENGTH] = {
		"E-n13-k4",
		"E-n22-k4",
		"E-n23-k3",
		"E-n30-k3",
		"E-n31-k7",
		"E-n33-k4",
		"E-n51-k5",
		"E-n76-k7",
		"E-n76-k8",
		"E-n76-k10",
		"E-n76-k14",
		"E-n101-k8",
		"E-n101-k14",
		"G-n262-k25"
	};

	char coff_test[LENGTH] = "Vrp-All/E/E-n13-k4.vrp";

	//unsigned long init[4] = { 0x123, 0x234, 0x345, 0x456 }, length = 4; // mt用

	srand(2013);
	//init_by_array(init, length); // mt用

	/*********************************/
	/* Read vrp_problem file section */
	/*********************************/
	strcpy(infile_name, "");
	strcpy(infile_name, "Vrp-All/E/");
	strcat(infile_name, paper_problem[3]);
	strcat(infile_name, ".vrp");

	vrp = (vrp_problem *)ec_malloc(sizeof(vrp_problem));
	vrp_io(vrp, infile_name);
	initialize_global_var(vrp); // 中身がわかりづらい？

	sl = (S_LIST *)ec_malloc(sizeof(S_LIST));
	sl->edge = (EDGE *)ec_calloc(vrp->edgenum, sizeof(EDGE));



	best_distance = 1e+9;
	count = 0;
	while (count < 1000) {
		memset(route, 0, sizeof(int)*ROUTE_MAX);
		memset(current_cap, 0, sizeof(int)*VEHICLE_SIZE);
		memset(route_size, 0, sizeof(int)*VEHICLE_SIZE);
		sl->num_sav = 0;
		memset(sl->edge, 0, sizeof(EDGE)*vrp->edgenum);
		create_savings_list(sl, vrp);
		distance = binary_cws_simulation();
		distance = allVisit() ? distance : distance + 10000;
		//printf("allVisit():");
		/*
		if (allVisit()) {
			printf("YES\n");
		} else {
			printf("NO\n");
		}
		printf("distance:%4d\n", distance);
		*/
		if (best_distance > distance) {
			best_distance = distance;
		}
		count++;
	}
	printf("best_distance:%4d\n", best_distance);


	free_global_var();
	free(vrp);
	return 0;
}
