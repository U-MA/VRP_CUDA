/* DESCRIPTION
 * 2013.04.04 車体の動きについて:
 *            最初の車体が倉庫を出発し、倉庫に戻ってくるまで他の車体は動かない。
 *            最初の車体が倉庫に戻って来たときに次の車体が倉庫を出発する
 */

/* CHANGE LOG
 * 2013.04.12 - uct_jp.cppからc++の記法を排除し、拡張子をcに変更。
 *            - vrpファイルから読み込みを行う部分を追加。
 *              dimandsやdistanceに対する変更は現時点(12:49PM)では行っていない。
 *            - SYMPHONYのコードではdepotは1、自分のコードではdepotは0。自分のコードを
 *              変更することにした(但し、17:00PM現在この変更は行っていない)。
 * 2013.04.14 - 上記の問題を直した。
 * 2013.04.16 - gitを用いてヴァージョン管理の勉強。これはコードとは関係ない！テスト。
 */

/* CAUTION
 * 2013.04.12 - 突貫工事として、NODE構造体のメンバchild配列を動的に確保した。
 *              しかし、それらを解放はしていないのでガーベジコレクタ的なものを
 *              実装する必要がある。
 */

/* TO DO LIST
 * 2013.04.12 - NODE構造体でcallocしたchildを解放する必要がある。uct関数が終われば
 *              解放出来るはず
 *            - SYMPHONYのファイルから必要のないものを削除
 *            - シミュレーションとしてCWSアルゴリズムを用いる
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

#include "uct_jp.h"
#include "uct_const.h"

#define DEPOT_NUM 0

int CUSTOMER_SIZE; //!< 倉庫(depot)と顧客(customer)を合わせた数 
int VEHICLE_SIZE; //!< 車体(vehicle)の数

int CAPACITY; //!< 車体(vehicle)の容量
const double all_distance = 1.0e+4; //!< ucb値に用いる。
//-------------------------------

int ROUTE_MAX; // = CUSTOMER_SIZE * VEHICLE_SIZE;

//! 各車体(vehicle)が走行したルート。
/*! 倉庫(depot)を意味する値0をこの配列に代入する事は無い。\n
 *  0番目の車体(vehicle)が2番目に訪れた顧客(customer)を表すにはroute[CUSTOMER_SIZE * 0 + 2]とする
 */
int *route;
int current_vehicle;  //!< 現在走行している車体(vehicle)の番号。0から始まる。関数moveでのみ増やされる。
int idx; //!< 現在の車体(current_vehicle)が訪れた顧客の数
extern int *current_cap; //!< 各車体(vehicle)が訪れた顧客の要求(demand)の合計。1番目の車体(current_vehicle=1)が訪れた顧客の要求の合計を表すにはcurrent_cap[1]とする

int *best_route;
int *best_cap;

NODE *node;//[NODE_MAX] = { 0 }; //!< 探索木を表す配列 
int num_nodes = 0; //!< 探索木のノードの数

extern vrp_problem *vrp;

double testC;
double maximum;
double minimum;

//! 車体番号、要求の合計、ルートを表示する関数
/*!
 * 左から車体番号(veh)、要求の合計(cap)、ルート(route)を順に表示する。
 * ルートに倉庫(depot)は表示しない。
 */
void print_route() {
	int i, j, k;
	int customer;

	printf("current_cap: %p, route: %p\n", current_cap, route);
	printf("+-----+-----+");
	for (i=0; i < CUSTOMER_SIZE; i++)
		printf("----");
	printf("+\n");

	printf("| veh | cap | route  ");
	for (i=0; i <= CUSTOMER_SIZE-3; i++)
		printf("    ");
	printf("|\n");

	printf("+-----+-----+");
	for (i=0; i < CUSTOMER_SIZE; i++)
		printf("----");
	printf("+\n");

	for (i=0; i < VEHICLE_SIZE; i++) {
		printf("|%4d |%4d |", i, current_cap[i]);
		for (j=0; j < CUSTOMER_SIZE; j++) {
			customer = route[i * CUSTOMER_SIZE + j];
			if (customer == 0)
				break;
			else
				printf("%3d ", customer);
		}
		for(k=0; k < CUSTOMER_SIZE - j; k++) { // CUSTOMER_SIZE - jが0より小さい事はないのか？
			printf("    ");
		}
		printf("|\n");
		printf("+-----+-----+");
		for (k=0; k < CUSTOMER_SIZE; k++)
			printf("----");
		printf("+\n");
	}
}

//! ノードの子の訪問回数を表示する
/*!
 * 引数pNodeのメンバchild配列の各訪問回数を表示する。pNodeの子が無い場合は何も表示せず終了。
 */
void print_customer_count(NODE *pNode) {
	int i;

	if (pNode->nc == 0)
		return;

	for (i=0; i < pNode->nc; i++) {
		printf("+-----");
	}
	printf("+\n");
	for (i=0; i < pNode->nc; i++) {
		CHILD *c = &pNode->child[i];
		printf("|%4d ", c->customer);
	}
	printf("|\n");
	for (i=0; i < pNode->nc; i++) {
		printf("+-----");
	}
	printf("+\n");
	for (i=0; i < pNode->nc; i++) {
		CHILD *c = &pNode->child[i];
		printf("|%4d ", c->count);
	}
	printf("|\n");
	for (i=0; i < pNode->nc; i++) {
		printf("+-----");
	}
	printf("+\n");
}

//! 顧客customer_numを既に訪れているか確認する関数
/*!
 * route配列を最初から見て行き、customer_numを既に訪れている場合は1、そうでない場合は0を返す。
 */
int isVisited(int customer_num) {
	int i;

	for (i=0; i < ROUTE_MAX; i++) {
		if (route[i] == customer_num)
			return 1;
	}
	return 0;
}

//! 全ての顧客を訪れているか確認する関数
/*!
 * route配列を確認し、全ての顧客を訪れている場合は1、そうでない場合は0を返す
 */
int allVisit() {
	int i, j;

	for (i=1; i < CUSTOMER_SIZE; i++) {
		for (j=0; j < ROUTE_MAX; j++) {
			if (route[j] == i) // find
				break;
			else if (j == ROUTE_MAX-1) // didn't find
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
int calc_distance(vrp_problem *vrp) {
	int sum_distance;
	int vehicle_num;
	int i;
	int c1, c2;
	distances *dist = &vrp->dist;

	vehicle_num = sum_distance = 0;
	while (vehicle_num < VEHICLE_SIZE) {
		c1 = DEPOT_NUM;
		c2 = route[vehicle_num * CUSTOMER_SIZE]; // 車体vehicle_numが最初に訪れた顧客
		for (i=1; i <= CUSTOMER_SIZE; i++) {
			if ((c1 == 0) && (c2 == 0)) {
				continue;
			} else if (c2 == DEPOT_NUM) {
				sum_distance += dist->cost[INDEX(c1, c2)];
				break;
			}
			sum_distance += dist->cost[INDEX(c1, c2)];
			c1 = c2;
			c2 = route[vehicle_num * CUSTOMER_SIZE + i];
			if (i == CUSTOMER_SIZE)
				sum_distance += dist->cost[INDEX(c1, c2)];
		}
		vehicle_num++;
	}
	return sum_distance;
}

//! 次の顧客に進む関数
/*!
 * この関数は現在の車体(current_vehicle)で顧客customer_numに進むことを試みる。\n
 * 顧客customer_numに進む事で、現在の車体の要求(demands)の合計が最大容量を超えてしまう場合は-1を返す。これは先に進めなかったことを表す。\n
 * そうでない場合は顧客customer_numに進み、現在の車体の要求の合計を更新する。そして、正常終了を意味する0を返す。\n
 * 顧客customer_numが0の場合は、車体を変更する。そして、正常終了を意味する0を返す。
 */
int move(int customer_num, int *route) {
	if (customer_num == 0) {
		//printf("change vehicle in move()\n");
		current_vehicle++;
		//printf("vehicle %d run from now!!\n", current_vehicle);
		idx = 0;
		return 0;
	} else if (current_cap[current_vehicle] + vrp->demand[customer_num] > CAPACITY) { // capacity is exceeded.
		return -1;
	} else {
		route[current_vehicle * CUSTOMER_SIZE + idx] = customer_num;
		current_cap[current_vehicle] += vrp->demand[customer_num];
		idx++;
		return 0;
	}
}


//! ランダムシミュレーションを行う関数
/*!
 * route配列を確認し、次に進める顧客の候補を選び、その中からランダムに1人の顧客を選ぶ。\n
 * 候補者の数が0のとき次の車体に変更する。\n
 * シミュレーションが終わったときの距離の合計を返す。\n
 * しかし、全ての顧客を通っていない場合は距離に10000を加算し返す。10000という数字に根拠は無い。
 * この関数は、全ての顧客(customer)を訪問していなくてもシミュレーションが終了し、距離を測ることがある。そうした場合、全ての顧客を訪問したときの距離よりもその距離が小さいことがあり得る。
 * そのため、10000を加算しているのは全ての顧客を訪問していない場合は補正値を設けてみては、というアイデアである。
 * これでうまくいかない場合もあるのではと思っている。
 */
int simulation() {
	int candidates[CUSTOMER_SIZE];
	int num_candidates;
	int error;
	int i;
	int distance;

	//printf("-----< Now simulation >------\n");
	while (current_vehicle < VEHICLE_SIZE && !allVisit()) {
		num_candidates = 0;
		error = -1;

		for (i=1; i < CUSTOMER_SIZE; i++) { // i=0 means the depot.
			if (!isVisited(i)) { // customer i is not visited.
				if (current_cap[current_vehicle] + vrp->demand[i] <= CAPACITY) {
					candidates[num_candidates] = i;
					num_candidates++;
				}
			}
		}

		while (1) {
			if (num_candidates == 0) {
				(void)move(0, route); // choose next vehicle
				break;
			} else {
				i = (int)(genrand_int32() % (unsigned long)num_candidates);
				error = move(candidates[i], route);
				if (error != -1) {
					break;
				}
				/*else {
	  printf("...falure.\n");
	  candidates[i] = candidates[num_candidates-1];
	  num_candidates--;
	  }*/
			}
		}
	}
	distance = allVisit() ? calc_distance(vrp) : calc_distance(vrp) + 10000;
	return distance;
}

//! ノードの子を作る関数
/*!
 * 子の初期設定を行う。
 * この関数はcreate_node関数の中でのみ呼び出される。
 */
void create_child(NODE *pNode, int customer_num) {
	int n = pNode->nc;

	pNode->child[n].customer = customer_num;
	pNode->child[n].count = 0;
	pNode->child[n].distance_ave = 0.0;
	pNode->child[n].next = NODE_EMPTY;
	pNode->nc++;
}

//! 探索木のノードを作る関数
/*!
 * 探索木のノードを格納しているnode配列にノードを作成し、値の初期化を行う。
 * 作成するノードはまだ訪れておらず、要求(demands)の合計が容量を超えないような顧客を子とする。
 * current_vehicleがVEHICLE_SIZEであれば、DONT_HAVE_CHILDを返す。
 * 探索木のノードの数がNODE_MAXであればexitする。
 */
int create_node() {
	NODE *pNode;
	int i;


	if (num_nodes == NODE_MAX) {
		printf("ERROR: node max\n");
		exit(1);
	}

	pNode = &node[num_nodes];
	pNode->vehicle = -1;
	pNode->idx = -1;
	pNode->sum_cnt = 0;
	pNode->nc = 0;
	pNode->child = (CHILD *)ec_calloc(CHILD_MAX, sizeof(CHILD));
	while (pNode->nc == 0) {
		for (i=1; i < CUSTOMER_SIZE; i++) { // i=0 means the depot.
			if (!isVisited(i)) {
				if (current_cap[current_vehicle] + vrp->demand[i] <= CAPACITY)
					create_child(pNode, i);
			}
		}
		if (pNode->nc == 0) {
			(void)move(0, route); // 車体(current_vehicle)の変更
			if (current_vehicle == VEHICLE_SIZE) {
				return DONT_HAVE_CHILD;
			}
		}
	}
	pNode->vehicle = current_vehicle;
	pNode->idx = idx;
	num_nodes++;
	return num_nodes-1;
}


//! モンテカルロ木探索UCTアルゴリズムを行う関数
/*!
 * ノードの子の中で、最もucb値が大きいものを選ぶ。
 * 選ばれた子が初めて訪れた点であれば、その子からシミュレーションを行い、距離を返す。そうでなければ、その子を引数として再びこの関数を呼ぶ。
 * ノードに子が無ければ、その状態での距離を返す。
 * ucb値の第1項をall_distance - c->distance_aveとしているのは、distance_aveが小さいほどucb値を大きくしたいためである。
 */
int uct(int num_node) {
	NODE *pNode;
	CHILD *c;
	double ucb = 0.0;
	double max_ucb = -1.0e+10;
	double min_ucb = 1.0e+10;
	int select = -1;
	int i;
	int current_vehicle_copy;
	int idx_copy;
	int distance = 0;
	int error;
	double C = 97.5; //! ucb値の第２項の係数。実験によりこの値を決める。現在は仮の値を設定。

	pNode = &node[num_node];

	current_vehicle_copy = current_vehicle;
	current_vehicle = pNode->vehicle;

	idx_copy = idx;
	idx = pNode->idx;

	//printf("In uct:\n");
  //printf("\n ucb value[nodes number:%d, vehicle number:%d, idx = %d]\n", num_node, pNode->vehicle, pNode->idx);

  //printf("+---------+--------------+---------+-------+\n");
  //printf("| cus(i)  | ucb          | sum_cnt | count |\n");
  //printf("+---------+--------------+---------+-------+\n");
	for (i=0; i < pNode->nc; i++) {
		c = &pNode->child[i];
		if (c->count == 0)
			ucb = 1000 + rand(); // means infinite
		else {
			/*!
			 * ucb値の候補
			 *   - [normal_ucb.txt] (all_distance - c->distance_ave) + C * ...
			 *   - [ucb_min.txt]    c->distance_ave - C * ...
			 *   - [divide_ucb.txt] (c->distance_ave / all_distance) + C * ...
			 *   - [minus_ucb.txt]  - c->distance_ave + C * ...
       */
			ucb = - c->distance_ave + C * sqrt(2.0*log(pNode->sum_cnt) / c->count); // inner of function 'sqrt' is 'int'?
			if (maximum < -c->distance_ave) {
				maximum = -c->distance_ave;
			} else if (minimum > -c->distance_ave) {
				minimum = -c->distance_ave;
			}
		}

		//printf("|%4d(%2d) | %12.7g | %4d    |  %4d |\n", c->customer, i, ucb, pNode->sum_cnt, c->count);

		if (max_ucb < ucb) {
			max_ucb = ucb;
			select = i;
		}
		//printf("+---------+--------------+---------+-------+\n");
	}
	if (select == -1) {
		printf("ERROR: in uct; may be pNode->nc = 0\n");
		printf("ucb: %lg, min_ucb: %lg, select: %d\n", ucb, min_ucb, select);
		exit(1);
	}

	c = &pNode->child[select];

	//printf("customer %d is selected.\n", c->customer);
	error = move(c->customer, route);
	if (error != 0) {
		printf("ERROR: move error in uct\n");
		exit(1);
	}

	if (c->count == 0) {// threshold is 1
		distance = simulation();
	} else {
		if (c->next == NODE_EMPTY) {
			c->next = create_node();
		}

		if (c->next != DONT_HAVE_CHILD) {
			distance = uct(c->next);
		} else { // pNode is leaf
			//print_route();
			distance = allVisit() ? calc_distance(vrp) : calc_distance(vrp) + 10000; //! 探索木の葉におけるルートが全顧客を訪問してなければ距離に10000を加算する。この値に根拠は無い。
		}
	}
	current_vehicle = current_vehicle_copy;
	idx = idx_copy;
	c->distance_ave = (c->distance_ave * c->count + distance) / (c->count + 1);
	c->count++;
	pNode->sum_cnt++;
	//printf(":End uct\n\n");
	return distance;
}


//! uct関数に基づく最良の顧客を選ぶ関数
/*!
 * uct関数をloop_max回繰り返した後で、一番多く訪れられた子の顧客番号を返す。
 * ただし、この関数を呼び出した時の状態では、どこにも進めないときは-1を返す
 */
int search_best_customer() {
	int loop_max = 1000;
	NODE *pNode;
	int max_count;
	int best_customer;
	int route_copy[ROUTE_MAX];
	int cap_copy[VEHICLE_SIZE];
	int current_vehicle_copy;
	int idx_copy;
	int root;
	int i;
	CHILD *c;

	num_nodes = 0;
	root = create_node(); // create root node
	/*!
	 * (*1)作成したノードに子が無ければ-1を返す。子が無ければ探索木を深くは潜れないからである。(すなわち作成したノードは探索木の葉)
	 * 未訪問の顧客は残っているものの現在の車体(current_vehicle)では容量の都合で次の顧客(customer)に進めず
	 * さらに、次の車体が無いという場合が起こることを想定している。このような時は、main関数でroute配列を解として表示する。
   */
	if (root == DONT_HAVE_CHILD) { // (*1)
		printf("return -1 in search_best_customer\n");
		return -1;
	}

	for (i=0; i < loop_max; i++) {
		memcpy(route_copy, route, sizeof(int)*ROUTE_MAX);
		memcpy(cap_copy, current_cap, sizeof(int)*VEHICLE_SIZE);
		current_vehicle_copy = current_vehicle;
		idx_copy = idx;

		(void)uct(root);

		memcpy(route, route_copy, sizeof(int)*ROUTE_MAX);
		memcpy(current_cap, cap_copy, sizeof(int)*VEHICLE_SIZE);
		current_vehicle = current_vehicle_copy;
		idx = idx_copy;
	}

	max_count = -1;
	best_customer = -1;
	pNode = &node[root];

	//printf("*** customer's count ***\n");
  //print_customer_count(pNode);

	for (i=0; i < pNode->nc; i++) {
		c = &pNode->child[i];
		if (c->count > max_count) {
			max_count = c->count;
			best_customer = c->customer;
		}
	}
	if (best_customer == -1) {
		printf("ERROR: in search_best_customer\n");
		printf("may be pNode->nc = 0\n");
		exit(1);
	}

	return best_customer;
}

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

	route       = (int *)ec_calloc(ROUTE_MAX, sizeof(int));
	current_cap = (int *)ec_calloc(VEHICLE_SIZE, sizeof(int));
	best_route  = (int *)ec_calloc(ROUTE_MAX, sizeof(int));
	best_cap    = (int *)ec_calloc(VEHICLE_SIZE, sizeof(int));

	CHILD_MAX = CUSTOMER_SIZE;
	node  = (NODE *)ec_calloc(NODE_MAX, sizeof(NODE));
}

//! initialize_global_varで割り当てたメモリを解放する関数
/*!
 * route, current_capに割り当てられたメモリを解放する
 */
void free_global_var() {
	free(route);
	free(current_cap);
	free(node);
	free(best_route);
	free(best_cap);
}

void tolower_s(char *str) {
	int len_str;
	int i;

	len_str = strlen(str);
	for (i=0; i < len_str; i++) {
		str[i] = tolower(str[i]);
	}
}

int get_opt_from_file(char *optfile) {
	FILE *fp;
	int cost;
	char *str_pointer;
	char buf[10], line[80];
	char filename[80];

	strcpy(filename, optfile);
	str_pointer = strchr(filename, '.');
	*str_pointer = '\0';
	strcat(filename, ".opt");

	fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("can't open this file\n");
		exit(1);
	}

	while(fgets(line, 80, fp) != NULL) {
		tolower_s(line);
		strcpy(buf, "");
		sscanf(line, "%s", buf);
		if (strcmp(buf, "cost") == 0) {
			sscanf(line, "%s%d", buf, &cost);
			break;
		}
	}
	fclose(fp);
	return cost;
}

/*
int main(int argc, char **argv) {
	int best_customer;
	int best_distance;
	int i, opt;
	time_t start_time, current_time;

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

	char rest_paper[33][LENGTH] = {
		"Vrp-All/E/E-n76-k14.vrp",
		"Vrp-All/E/E-n101-k8.vrp",
		"Vrp-All/E/E-n101-k14.vrp",
		"Vrp-All/G/G-n262-k25.vrp"
	};

	char coff_test[LENGTH] = "Vrp-All/E/E-n13-k4.vrp";

	unsigned long init[4] = { 0x123, 0x234, 0x345, 0x456 }, length = 4;

	maximum = -1.0e+9;
	minimum = 1.0e+9;
	//for (testC=80.0; testC < 90.0; testC += 0.5) {
		srand(2013);
		//init_by_array(init, length);
		best_customer = -1;
		best_distance = 1e+7;

		*********************************/
		/* Read vrp_problem file section */
		/*********************************
		//strcpy(infile_name, "");
		*
		strcpy(infile_name, "Vrp-All/E/");
		strcat(infile_name, paper_problem[i]);
		strcat(infile_name, ".vrp");
		*
		//strcpy(infile_name, rest_paper[i]);

		vrp = (vrp_problem *) malloc(sizeof(vrp_problem));
		vrp_io(vrp, coff_test);
		//opt = get_opt_from_file(infile_name);

		initialize_global_var(vrp);

		start_time = current_time = time(0);
		while (current_time - start_time <= 300) { // do-whileにするのかwhileにするのか
			memset(route, '\0', sizeof(int)*ROUTE_MAX);
			memset(current_cap, '\0', sizeof(int)*VEHICLE_SIZE);
			current_vehicle = 0;
			idx = 0;
			while ((current_vehicle < VEHICLE_SIZE) && !allVisit()) {
				if ((best_customer = search_best_customer()) == -1) { // 現在のルート(route)ではこれ以上進めない場合
					break;
				} else {
					route[CUSTOMER_SIZE * current_vehicle + idx] = best_customer;
					current_cap[current_vehicle] += vrp->demand[best_customer];
					idx++;
				}
			}

			
	 printf("\n****************\n");
	 printf("*   solution   *\n");
	 printf("****************\n");
	 print_route();
			//printf("distance: %d, optimal: %d\n\n", calc_distance(), opt);
			if (best_distance > calc_distance()) {
				memcpy((void *)best_route, (void *)route, sizeof(int)*ROUTE_MAX);
				memcpy((void *)best_cap, (void *)current_cap, sizeof(int)*VEHICLE_SIZE);
				best_distance = calc_distance();
			}
			current_time = time(0);
		}

		memcpy((void *)route, (void *)best_route, sizeof(int)*ROUTE_MAX);
		memcpy((void *)current_cap, (void *)best_cap, sizeof(int)*VEHICLE_SIZE);
		//printf("\n****************\n");
		//printf("*     best     *\n");
		//printf("****************\n");
		//print_route();
		printf("testC = %lg, best distance = %d\n", testC, best_distance);
		//printf("maximum = %lg, minimum = %lg\n", maximum, minimum);

		free_global_var();
	//}
	exit(0);
	return 0;
}
*/
