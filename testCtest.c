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


CWS_NODE *node; //!< 探索木を表す配列 
int num_nodes = 0; //!< 探索木のノードの数

vrp_problem *vrp;
S_LIST *sl;

// ucb値のテスト用係数
int testC;

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
	int p1, p2;
	int nrp1, nrp2;
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

	//unsigned long init[4] = { 0x123, 0x234, 0x345, 0x456 }, length = 4; // mt用

	for (testC =900; testC < 1005; testC += 5) {
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
		best_route = (int *)ec_calloc(ROUTE_MAX, sizeof(int));
		best_cap = (int *)ec_calloc(VEHICLE_SIZE, sizeof(int));

		sl = (S_LIST *)ec_malloc(sizeof(S_LIST));
		sl->edge = (EDGE *)ec_calloc(vrp->edgenum, sizeof(EDGE));
		sl_cpy = (S_LIST *)ec_malloc(sizeof(S_LIST));
		sl_cpy->edge = (EDGE *)ec_calloc(vrp->edgenum, sizeof(EDGE));


		best_distance = 1e+7;
		start_time = current_time = time(0);
		while (current_time - start_time <= 300) { // do-whileにするのかwhileにするのか
			printf("[MAIN WHILE]\n");
			memset(route, 0, sizeof(int)*ROUTE_MAX);
			memset(current_cap, 0, sizeof(int)*VEHICLE_SIZE);
			memset(route_size, 0, sizeof(int)*VEHICLE_SIZE);
			sl->num_sav = 0;
			memset(sl->edge, 0, sizeof(EDGE)*vrp->edgenum);
			create_savings_list(sl, vrp);
			while (0 < sl->num_sav) {
				best_savings_idx = 0; // 念のため初期化

				// slの内容を退避
				sl_cpy->num_sav = sl->num_sav;
				memcpy(sl_cpy->edge, sl->edge, sizeof(EDGE)*vrp->edgenum);

				best_savings_idx = search_best_savings(); // 最も良いsavings listのインデックスを取得

				// 退避させていたslの内容を復元
				sl->num_sav = sl_cpy->num_sav;
				memcpy(sl->edge, sl_cpy->edge, sizeof(EDGE)*vrp->edgenum);

				if (best_savings_idx != -1) {
					branch_cws_method(sl, best_savings_idx);
					//printf("sl->num_sav:%3d\n", sl->num_sav);
	  //print_route_tsplib_format();
				} else {
					break;
				}
			}
			while (allVisit() != 1) { // this while-loop do once, I think.
				rest = unvisited_customer(route);
				if (rest != 0) {
					printf("customer %d is rest\n", rest);
					error = create_new_route(rest, 0, route, route_size, current_cap);
					if (error == -1) {
						break;
					}
				} else {
					break;
				}
			}
			current_distance = calc_distance();
			//printf("[MAIN WHILE END](distance:%4d)\n", current_distance);
      //print_route_tsplib_format();
			if (best_distance > current_distance) {
				memcpy((void *)best_route, (void *)route, sizeof(int)*ROUTE_MAX);
				memcpy((void *)best_cap, (void *)current_cap, sizeof(int)*VEHICLE_SIZE);
				best_distance = current_distance;
			}
			current_time = time(0);
		}

		memcpy((void *)route, (void *)best_route, sizeof(int)*ROUTE_MAX);
		memcpy((void *)current_cap, (void *)best_cap, sizeof(int)*VEHICLE_SIZE);
		printf("\n****************\n");
		printf("*     best     *\n");
		printf("****************\n");
		printf("testC:%4d\n", testC);
		print_route_tsplib_format();
		printf("best distance = %d\n", best_distance);
	}

	free_global_var();
	free(best_route);
	free(best_cap);
	free(sl->edge);
	free(sl);
	free(sl_cpy->edge);
	free(sl_cpy);
	free(vrp);
	return 0;
}
