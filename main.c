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

#include "mt19937.h" // メルセンヌツイスターのため
#include "uct_types.h"
#include "uct_const.h"
#include "uct_with_cws.h"
#include "my_lib.h"


//! 各車体(vehicle)が走行したルート。
/*! 倉庫(depot)を意味する値0をこの配列に代入する事は無い。\n
 *  0番目の車体(vehicle)が2番目に訪れた顧客(customer)を表すにはroute[CUSTOMER_SIZE * 0 + 2]とする
 */
//int *route;
//int *route_size; //!< 各routeに属している顧客の数
//int *current_cap; //!< 各車体(vehicle)が訪れた顧客の要求(demand)の合計。1番目の車体(current_vehicle=1)が訪れた顧客の要求の合計を表すにはcurrent_cap[1]とする

RDATA *rdata;

CWS_NODE *node; //!< 探索木を表す配列 
int num_nodes = 0; //!< 探索木のノードの数

vrp_problem *vrp;
S_LIST *sl;

// ucb値のテスト用係数
int testC;


int main(int argc, char **argv) {
	int best_savings_idx;
	int best_distance, current_distance;
	int *best_route;
	int *best_cap;
	time_t start_time, current_time;
	S_LIST *sl_cpy;
	int rest;
	int error;

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
	int pidx = 0;

	unsigned long init[4] = { 0x123, 0x234, 0x345, 0x456 }, length = 4; // mt用


	init_by_array(init, length); // mt用

	/*********************************/
	/* Read vrp_problem file section */
	/*********************************/
	strcpy(infile_name, "");
	strcpy(infile_name, "Vrp-All/E/");
	strcat(infile_name, paper_problem[pidx]);
	strcat(infile_name, ".vrp");

	vrp = (vrp_problem *)ec_malloc(sizeof(vrp_problem));
	vrp_io(vrp, infile_name);
	if (vrp->numroutes == 0) {
		vrp->numroutes = atoi(strchr(vrp->name, 'k')+1);
	}

	rdata        = (RDATA *)ec_malloc(sizeof(RDATA));
	node         = (CWS_NODE *)ec_calloc(NODE_MAX, sizeof(CWS_NODE));
	best_route   = (int *)ec_calloc(vrp->vertnum*vrp->numroutes, sizeof(int));
	best_cap     = (int *)ec_calloc(vrp->numroutes, sizeof(int));
	sl           = (S_LIST *)ec_malloc(sizeof(S_LIST));
	sl->edge     = (EDGE *)ec_calloc(vrp->edgenum, sizeof(EDGE));
	sl_cpy       = (S_LIST *)ec_malloc(sizeof(S_LIST));
	sl_cpy->edge = (EDGE *)ec_calloc(vrp->edgenum, sizeof(EDGE));
	rdata_init(rdata, vrp);

	memset(rdata->route, 0, sizeof(int)*vrp->vertnum*vrp->numroutes);
	memset(rdata->route_cap, 0, sizeof(int)*vrp->numroutes);
	memset(rdata->route_size, 0, sizeof(int)*vrp->numroutes);
	sl->num_sav = 0;
	memset(sl->edge, 0, sizeof(EDGE)*vrp->edgenum);
	create_savings_list(sl);
	best_distance = 1e+7;
	num_nodes = 0;
	while (0 < sl->num_sav) {
		// slの内容を退避
		sl_cpy->num_sav = sl->num_sav;
		memcpy(sl_cpy->edge, sl->edge, sizeof(EDGE)*vrp->edgenum);

		best_savings_idx = search_best_savings(sl, rdata); // 最も良いsavings listのインデックスを取得

		// 退避させていたslの内容を復元
		sl->num_sav = sl_cpy->num_sav;
		memcpy(sl->edge, sl_cpy->edge, sizeof(EDGE)*vrp->edgenum);

		if (best_savings_idx != -1) {
			branch_cws_method(sl, best_savings_idx, rdata);
			print_route_tsplib_format(rdata);
		} else {
			break;
		}
	}
	while ((rest = unvisited_customer(rdata)) != 0) { // this while-loop do once, I think.
		if (create_new_route(rest, 0, rdata) == -1) {
				break;
			}
	}
	current_distance = calc_distance(rdata);
	if (best_distance > current_distance) {
		memcpy((void *)best_route, (void *)rdata->route, sizeof(int)*vrp->vertnum*vrp->numroutes);
		memcpy((void *)best_cap, (void *)rdata->route_cap, sizeof(int)*vrp->numroutes);
		best_distance = current_distance;
	}

	memcpy((void *)rdata->route, (void *)best_route, sizeof(int)*vrp->vertnum*vrp->numroutes);
	memcpy((void *)rdata->route_cap, (void *)best_cap, sizeof(int)*vrp->numroutes);
	printf("\n****************\n");
	printf("*     best     *\n");
	printf("****************\n");
	print_route_tsplib_format(rdata);
	printf("best distance = %d\n", best_distance);

	rdata_free(rdata);
	free(best_route);
	free(best_cap);
	free(sl->edge);
	free(sl);
	free(sl_cpy->edge);
	free(sl_cpy);
	free(vrp);
	return 0;
	}
