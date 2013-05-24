/* DESCRIPTION
 *    このファイルはCapacitated Vehicle Routing Problem(CVRP)を解くために
 *    モンテカルロ木探索のUCTアルゴリズムを用いたプログラムを記述している。
 *    UCTアルゴリズムのシミュレーションプログラムにはClark And Wright's Method
 *    を使用している。
 */

/* CHANGE LOG
 * 2013.04.28 このファイルを作成。
 */

/* CAUTION
 *
 */

/* TO DO LIST
 *
 */
/*******************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

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


#include "mt19937.h"
#include "uct_types.h"
#include "uct_const.h"
#include "cws.h"
#include "my_lib.h"


extern vrp_problem *vrp;
extern S_LIST *sl;
extern int testC;

extern CWS_NODE *node;
extern int num_nodes;


//! RDATA構造体のコンストラクタ
/*!
 * 顧客の数、ルートの数、容量を初期し、route, route_size, route_capにはメモリを割り当てる。
 * 割当が失敗するとexit(1)をする。引数vrpの値がnullのときもexitする。
 */
void rdata_init(RDATA *rdata, vrp_problem *vrp) {
	if (rdata == NULL || vrp == NULL) {
		printf("argumetn vrp or rdata is null.\n");
		exit(1);
	}

	rdata->num_customers = vrp->vertnum;
	rdata->num_routes    = vrp->numroutes;
	rdata->capacity      = vrp->capacity;
	rdata->route         = (int *)ec_calloc(vrp->vertnum*vrp->numroutes, sizeof(int));
	rdata->route_size    = (int *)ec_calloc(vrp->numroutes, sizeof(int));
	rdata->route_cap     = (int *)ec_calloc(vrp->numroutes, sizeof(int));
}

//! RDATA構造体のメンバおよび構造体のメモリを解放する
void rdata_free(RDATA *rdata) {
	free(rdata->route);
	free(rdata->route_size);
	free(rdata->route_cap);
	free(rdata);
}

//! 各ルートの内容を出力する。
/*!
 * 各ルートの現在の容量、ルートの最大容量、ルートの内容を出力する。
 */
void print_route_tsplib_format(RDATA *rdata) {
	int i, j;
	int customer;

	printf("%s\n", vrp->name);
	for (i=0; i < rdata->num_routes; i++) {
		printf("Route #%d[Cap:%4d/%4d]", i, rdata->route_cap[i], rdata->capacity);
		for (j=0; j < rdata->num_customers; j++) {
			customer = rdata->route[i*rdata->num_customers+j];
			if (customer != 0)
				printf("%4d", customer);
		}
		printf("\n");
	}
	putchar('\n');
}

//! ノードの子の訪問回数を表示する
/*!
 * 引数pNodeのメンバchild配列の各訪問回数を表示する。pNodeの子が無い場合は何も表示せず終了。
 */
void print_customer_count_cws(CWS_NODE *pNode, S_LIST *sl) {
	int i;
	CWS_CHILD *c;

	printf(" idx  sav_idx      edge count distance_ave\n");
	for (i=0; i < pNode->nc; i++) {
		c = &pNode->child[i];
		printf("%4d %8d (%3d,%3d) ", i, c->savings_idx, sl->edge[c->savings_idx].from, sl->edge[c->savings_idx].to);
		printf("%5d %12lg\n", c->count, c->distance_ave);
	}
	putchar('\n');
}

//! 顧客customer_numを既に訪れているか確認する関数
/*!
 * route配列を最初から見て行き、customer_numを既に訪れている場合は1、そうでない場合は0を返す。
 */
int isVisited(RDATA *rdata, int customer_num) {
	int i;
	int route_max;

	route_max = rdata->num_customers*rdata->num_routes;
	for (i=0; i < route_max; i++) {
		if (rdata->route[i] == customer_num)
			return 1;
	}
	return 0;
}

//! 全ての顧客を訪れているか確認する関数
/*!
 * route配列を確認し、全ての顧客を訪れている場合は1、そうでない場合は0を返す, time complacity O(n*k)
 */
int allVisit(RDATA *rdata) {
	int customer_num, route_max;
	int i;

	route_max = rdata->num_customers*rdata->num_routes;
	for (customer_num=1; customer_num < rdata->num_customers; customer_num++) {
		for (i=0; i < route_max; i++) {
			if (rdata->route[i] == customer_num) // find
				break;
			else if (i == route_max-1) // didn't find
				return 0;
		}
	}
	return 1;
}


//! ルートの合計距離を返す関数
/*!
 * 倉庫と最初の車体(vehicle_num)が最初に訪問した顧客との距離、その最初に訪問した顧客と2番目に訪問した顧客との距離、2番目に訪問した顧客と3番目に訪問した顧客との距離...
 * 最後に訪問した顧客と倉庫との距離、これらを全て合計する。これを全ての車体に対して行う。そうして全てを合計した距離を返す。\n
 */
int calc_distance(RDATA *rdata) {
	int sum_distance;
	int route_num;
	int i;
	int c1, c2;
	distances *dist = &vrp->dist;

	sum_distance = 0;
	for (route_num=0; route_num < rdata->num_routes; route_num++) {
		c2 = DEPOT_NUM;
		for (i=0; i < rdata->num_customers; i++) {
			c1 = c2;
			c2 = rdata->route[route_num*rdata->num_customers+i];
			if (c2 == DEPOT_NUM) {
				sum_distance += dist->cost[INDEX(c1, DEPOT_NUM)];
				break; // 次のルートへ
			} else {
				sum_distance += dist->cost[INDEX(c1, c2)];
			}
		}
		if (i == rdata->num_customers) {
			sum_distance += dist->cost[INDEX(c2, DEPOT_NUM)];
		}
	}
	return sum_distance;
}


//! binary cws simulationを行う。
/*!
 * 確率probabilityに基づいてシミュレーションを行う。
 */
int binary_cws_simulation(RDATA *rdata, S_LIST *sl, int probability) { // TODO: 確率を範囲で指定する場合をどうするか
	int i, x, rest;
	int num_sav_cpy; //! edge[i].fromが0でないものの個数。つまり利用可能なsavingsの数

	num_sav_cpy = 0;
	for (i=0; i < sl->num_sav; i++) { // 削除されていないsavingsの数を数える
		if (sl->edge[i].from != 0) {
			num_sav_cpy++;
		}
	}
	if (num_sav_cpy == 0) { // エラー処理
		printf("num_sav_cpy is zero.\n");
		exit(1);
	}
	i=0;
	while (num_sav_cpy > 0) {
		while (sl->edge[i].from == 0) { // 削除されていないsavingsを上から順に選択
			i++;
			if (i >= sl->num_sav) { // savings listを最後まで見終わったら最初に戻る
				i = 0;
			}
		}
		x = genrand_int32() % 100;
		if (x >= probability) { // probabilityはsavingsをスキップする確率
			branch_cws_method(sl, i, rdata);
			num_sav_cpy--;
		} else { // 処理しない場合
			i++;
		}
	}

	while ((rest = unvisited_customer(rdata)) != 0) {
		if (create_new_route(rest, 0, rdata) == -1) {
			break; // 新しいルートを作れなかった
		}
	}
	return calc_distance(rdata);
}

//! ノードの子を作る関数
/*!
 * 子の初期設定を行う。
 * この関数はcreate_node関数の中でのみ呼び出される。
 */
static void create_cws_child(CWS_NODE *pNode, int savings_idx) {
	int n = pNode->nc;

	pNode->child[n].savings_idx  = savings_idx;
	pNode->child[n].count        = 0;
	pNode->child[n].distance_ave = 0.0;
	pNode->child[n].next         = NODE_EMPTY;
	pNode->nc++;
}

//! 探索木のノードを作る関数
/*!
 * 探索木のノードを格納しているnode配列にノードを作成し、値の初期化を行う。
 * 作成するノードはまだ訪れておらず、要求(demands)の合計が容量を超えないような顧客を子とする。
 * current_vehicleがVEHICLE_SIZEであれば、DONT_HAVE_CHILDを返す。
 * 探索木のノードの数がNODE_MAXであればexitする。
 */
static int create_cws_node(RDATA *rdata, S_LIST *sl) {
	CWS_NODE *pNode;
	int i;

	if (num_nodes == NODE_MAX) {
		printf("ERROR: node max\n");
		exit(1);
	}

	pNode = &node[num_nodes];
	pNode->sum_cnt = 0;
	pNode->nc = 0;
	pNode->child = (CWS_CHILD *)ec_calloc(sl->num_sav, sizeof(CWS_CHILD));
	for (i=0; i < sl->num_sav; i++) {
		if (isAddCws(rdata, sl, i)) {
			create_cws_child(pNode, i);
		}
	}
	if (pNode->nc == 0) {
		return DONT_HAVE_CHILD;
	}
	num_nodes++;
	return num_nodes-1;
}

//! savings listのsavingsを子として加えるかを判定する
int isAddCws(RDATA *rdata, S_LIST *sl, int sl_idx) {
	int p1, p2;
	int nrp1, nrp2;
	int i;
	EDGE *e;

	e = &sl->edge[sl_idx];
	p1 = e->from;
	p2 = e->to;
	nrp1 = search_route_belonging_to(p1, rdata);
	nrp2 = search_route_belonging_to(p2, rdata);

	if ((nrp1 == -1) && (nrp2 == -1)) { // どちらの顧客(customer)もルートに属していない
		if (vrp->demand[p1] + vrp->demand[p2] <= rdata->capacity) {
			for (i=0; i < rdata->num_routes; i++) {
				if (rdata->route_size[i] == 0) {
					return 1;
				}
			}
			return 0;
		}
	} else if ((nrp1 == -1) && (nrp2 != -1) && (is_interior(rdata, p2) == 0)) { // 顧客p1のみがルートに属しておらず、p2がinteriorでない
		if (rdata->route_cap[nrp2] + vrp->demand[p1] <= rdata->capacity) {
			return 1;
		}
	} else if ((nrp2 == -1) && (nrp1 != -1) && (is_interior(rdata, p1) == 0)) { // 顧客p2のみがルートに属しておらず、p1がinteriorでない
		if (rdata->route_cap[nrp1] + vrp->demand[p2] <= rdata->capacity) {
			return 1;
		}
	} else if ((nrp1 != -1) && (nrp2 != -1) && (nrp1 != nrp2) && (is_interior(rdata, p1) || is_interior(rdata, p2)) == 0) { // どちらの顧客も互いに異なるルートに属しており
		if (rdata->route_cap[nrp1] + rdata->route_cap[nrp2] <= rdata->capacity) {
			return 1;
		}
	}
	return 0;
}

//! モンテカルロ木探索UCTアルゴリズムを行う関数
/*!
 * ノードの子の中で、最もucb値が大きいものを選ぶ。
 * 選ばれた子が初めて訪れた点であれば、その子からシミュレーションを行い、距離を返す。そうでなければ、その子を引数として再びこの関数を呼ぶ。
 * ノードに子が無ければ、その状態での距離を返す。
 * ucb値の第1項をall_distance - c->distance_aveとしているのは、distance_aveが小さいほどucb値を大きくしたいためである。
 */
int uct_with_cws(int num_node, S_LIST *sl, RDATA *rdata) {
	CWS_NODE *pNode;
	CWS_CHILD *c;
	double ucb, max_ucb;
	int i, select, distance;
	double C = 9350; //! ucb値の第２項の係数。実験によりこの値を決める。

	pNode = &node[num_node];

	ucb = 0.0;
	max_ucb = -1.0e+10;
	select = -1;
	for (i=0; i < pNode->nc; i++) {
		c = &pNode->child[i];
		if (c->count == 0) {
			ucb = genrand_int32(); // means infinite
		} else {
			ucb = - c->distance_ave + C * sqrt(2.0*log((double)pNode->sum_cnt) / c->count); // inner of function 'sqrt' is 'int'?
		}
		if (max_ucb < ucb) {
			max_ucb = ucb;
			select = i;
		}
	}
	if (select == -1) {
		printf("ERROR: in uct; may be pNode->nc = 0\n");
		exit(1);
	}

	c = &pNode->child[select];
	i = c->savings_idx;
	branch_cws_method(sl, i, rdata);
	if (c->count == 0) { // threshold is 1
		distance = binary_cws_simulation(rdata, sl, 45);
	} else {
		if (c->next == NODE_EMPTY) {
			c->next = create_cws_node(rdata, sl);
		}
		if (c->next != DONT_HAVE_CHILD) {
			distance = uct_with_cws(c->next, sl, rdata);
		} else { // pNode is leaf
			distance = calc_distance(rdata);
		}
	}
	distance = allVisit(rdata) ? distance : 10000; // 全ての顧客を訪問していなければ距離を10000で統一
	c->distance_ave = (c->distance_ave * c->count + distance) / (c->count + 1);
	c->count++;
	pNode->sum_cnt++;
	return distance;
}


//! uct関数に基づく最良の顧客を選ぶ関数
/*!
 * uct関数をloop_max回繰り返した後で、一番多く訪れられた子のsavings番号を返す。
 * ただし、この関数を呼び出した時の状態では、どこにも進めないときは-1を返す
 */
int search_best_savings(S_LIST *sl, RDATA *rdata) {
	CWS_NODE *pNode;
	CWS_CHILD *c;
	S_LIST *sl_copy;
	int max_count;
	int best_savings_idx;
	int *route_copy, *cap_copy, *rsize_copy;
	int root;
	int i;
	int best_distance_ave;
	int route_max, num_route;
	int loop_max = 100000;

	route_max     = rdata->num_customers*rdata->num_routes;
	num_route     = rdata->num_routes;
	route_copy    = (int *)calloc(route_max, sizeof(int));
	cap_copy      = (int *)calloc(num_route, sizeof(int));
	rsize_copy    = (int *)calloc(num_route, sizeof(int));
	sl_copy       = (S_LIST *)ec_malloc(sizeof(S_LIST));
	sl_copy->edge = (EDGE *)ec_calloc(vrp->edgenum, sizeof(EDGE)); //TODO: ここはvrp->edgenumでなくてsl->num_savでも動くのでは？

	for ( ; num_nodes > 0; num_nodes--) {
		free(node[num_nodes-1].child);
	}
	root = create_cws_node(rdata, sl); // create root node
	/*!
	 * (*1)作成したノードに子が無ければ-1を返す。子が無ければ探索木を深くは潜れないからである。(すなわち作成したノードは探索木の葉)
	 * 未訪問の顧客は残っているものの現在の車体(current_vehicle)では容量の都合で次の顧客(customer)に進めず
	 * さらに、次の車体が無いという場合が起こることを想定している。このような時は、main関数でroute配列を解として表示する。
   */
	if (root == DONT_HAVE_CHILD) { // (*1)
		printf("return -1 in search_best_customer\n");
		return -1;
	}

	printf("loop count");
	for (i=0; i < loop_max; i++) {
		memcpy(route_copy, rdata->route, sizeof(int)*route_max);
		memcpy(cap_copy, rdata->route_cap, sizeof(int)*num_route);
		memcpy(rsize_copy, rdata->route_size, sizeof(int)*num_route);
		sl_copy->num_sav = sl->num_sav;
		memcpy(sl_copy->edge, sl->edge, sizeof(EDGE)*vrp->edgenum); //TODO: ここも上記todoと同じ

		if (i == 0) {
			printf("[%6d/%6d]", i+1, loop_max);
			fflush(stdout);
			printf("\033[8D");
		} else {
			printf("\033[6D");
			fflush(stdout);
			printf("%6d", i+1);
		}

		(void)uct_with_cws(root, sl, rdata);

		memcpy(rdata->route, route_copy, sizeof(int)*route_max);
		memcpy(rdata->route_cap, cap_copy, sizeof(int)*num_route);
		memcpy(rdata->route_size, rsize_copy, sizeof(int)*num_route);
		sl->num_sav = sl_copy->num_sav;
		memcpy(sl->edge, sl_copy->edge, sizeof(EDGE)*vrp->edgenum); //TODO: 上記todoと同様
	}
	putchar('\n');

	pNode = &node[root];
	print_customer_count_cws(pNode, sl);

	max_count = -1;
	best_savings_idx = -1;
	best_distance_ave = 1.0e+6;
	for (i=0; i < pNode->nc; i++) {
		c = &pNode->child[i];
		if ((c->count >= max_count) && (c->distance_ave < best_distance_ave)) { // c->distance_aveはどんなに悪くとも100000は超えない
				max_count = c->count;
				best_savings_idx = c->savings_idx;
				best_distance_ave = c->distance_ave;
		}
	}
	if (best_savings_idx == -1) {
		printf("ERROR: in search_best_savings\n");
		printf("may be pNode->nc = 0\n");
		exit(1);
	}
	free(route_copy);
	free(cap_copy);
	free(rsize_copy);
	free(sl_copy->edge);
	free(sl_copy);
	return best_savings_idx;
}
