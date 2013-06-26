#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cws.h"
#include "uct_types.h"
#include "uct_with_cws.h"
#include "my_lib.h"

extern vrp_problem *vrp;
extern S_LIST *sl;



void print_savings_list(const S_LIST *sl) {
	int i;

	printf("SAVINGS LIST(size: %3d)\n", sl->num_sav);
	for (i=0; i < sl->num_sav; i++) {
		printf("[%3d] %3d - %3d\n", i+1, sl->edge[i].from, sl->edge[i].to);
	}
}

//! 二つのsavings構造体の中身を入れ替える関数 (tested)
void swap_savings(SAVS *x, SAVS *y) {
	SAVS tmp;

	tmp = *x;
	*x = *y;
	*y = tmp;
}


//! 二分ヒープ構造にsavingsを追加する関数
/*!
 * 二分ヒープ構造を表す配列sに、ヒープのノードとしてメンバに値from, to, savを持つ値をを格納する。
 * 配列sの0番目の要素のメンバsavは配列の要素数を表す。関数内で配列sの要素数は１増加する。
 */
void insert_heap_savings(SAVS *s, int from, int to, int sav) {
	int n;
	EDGE *e;

	n = ++s[0].sav; // s[0].savは配列sの要素数を表す!!
	e = &s[n].edge;

	e->from  = from;
	e->to    = to;
	s[n].sav = sav;

	while ((n != 1) && (s[n].sav > s[n/2].sav)) {
		swap_savings(&s[n], &s[n/2]);
		n = n/2;
	}
}


//! 最大二分ヒープを表す配列sの根をsから削除し、その構造体が表す枝を返す関数
/*!
 * s[0].savは配列の要素数を表している。
 */
EDGE delete_max_savings(SAVS *s) {
	int n, i;
	EDGE e;

	n = s[0].sav;
	if (n == 0) {
		printf("heap is empty\n");
		exit(1);
	}

	e = s[1].edge;
	s[1] = s[n];
	i = 2;
	while (i < n) {
		if (s[i].sav < s[i+1].sav) {
			i++;
		}
		if (s[i/2].sav < s[i].sav) {
			swap_savings(&s[i/2], &s[i]);
		} else {
			break;
		}
		i = 2*i;
	}
	s[0].sav--;
	return e;
}

//! S_LISTにメモリを割り当てvrp_problemの内容に従ってsavings listを初期化する
void slist_init(S_LIST **sl, vrp_problem *vrp) {
	int i, j, sav;

	*sl = (S_LIST *)ec_malloc(sizeof(S_LIST));
	(*sl)->edge = (EDGE *)ec_calloc(vrp->edgenum, sizeof(EDGE));

	distances *dist = &vrp->dist;
	SAVS *s = (SAVS *)calloc(vrp->edgenum, sizeof(SAVS));

	(*sl)->num_sav = 0;
	for (i=2; i < vrp->vertnum; i++) {
		for (j=1; j < i; j++) {
			sav = dist->cost[INDEX(0, i)] + dist->cost[INDEX(0, j)] - dist->cost[INDEX(i, j)];
			insert_heap_savings(s, i, j, sav);
		}
	}
	(*sl)->num_sav = (*sl)->num_enable_sav = s[0].sav;
	for (i=0; i < (*sl)->num_sav; i++) {
		(*sl)->edge[i] = delete_max_savings(s);
	}
	free(s);
}

//! SAVS listを作成する関数
/*!
 * slはメモリアロケーションをしている必要がある。
 */
void create_savings_list(S_LIST *sl) {
	int i, j;
	int sav;
	SAVS *s;
	distances *dist;

	dist = &vrp->dist;
	s  = (SAVS *)calloc(vrp->edgenum, sizeof(SAVS));

	sl->num_sav = 0;
	for (i=2; i < vrp->vertnum; i++) {
		for (j=1; j < i; j++) {
			sav = dist->cost[INDEX(0, i)] + dist->cost[INDEX(0, j)] - dist->cost[INDEX(i, j)];
			insert_heap_savings(s, i, j, sav);
		}
	}
	sl->num_sav = s[0].sav;
	for (i=0; i < sl->num_sav; i++) {
		sl->edge[i] = delete_max_savings(s);
	}
	free(s);
}

//! 顧客customerが属しているルートを返す関数 (tested)
/*!
 * 顧客customerがどのルートにも属していなければ-1を返す
 */
int search_route_belonging_to(int customer, RDATA *rdata) {
	int i, j;

	for (i=0; i < rdata->num_routes; i++) {
		for (j=0; j < rdata->num_customers; j++) {
			if (rdata->route[rdata->num_customers*i+j] == 0) {
				break; // 次のルートを探す
			} else if (rdata->route[rdata->num_customers*i+j] == customer) {
				return i;
			}
		}
	}
	return -1;
}

void search_route_belonging_to_a(int *belong, RDATA *rdata) {
	int i, j;
	int customer_num;

	for (i=0; i < rdata->num_routes; i++) {
		for (j=0; j < rdata->num_customers; j++) {
			customer_num = rdata->route[rdata->num_customers*i+j];
			if (customer_num == 0) {
				break;
			} else {
				belong[customer_num] = i;
			}
		}
	}
}




//! 顧客customerが属しているルート番号とインデックス値を返す関数 (tested)
/*!
 * 顧客customerが属しているルート番号をroute_num, インデックス値をidxに格納し、0を返す。
 * customerがどのルートにも属していなければroute_num, idxの値は変更せず、1を返す
 */

int search_customer(RDATA *rdata, int customer, int *route_num, int *idx) {
	int i, j;

	for (i=0; i < rdata->num_routes; i++) {
		for (j=0; j < rdata->num_customers; j++) {
			if (rdata->route[rdata->num_customers*i+j] == 0) {
				break; // 次のルートを探す
			} else if (rdata->route[rdata->num_customers*i+j] == customer) {
				*route_num = i;
				*idx = j;
				return 0;
			}
		}
	}
	return 1;
}


//! 顧客customerがinteriorかどうかを調べる関数 (tested)
/*!
 * 顧客customerがinteriorであれば1, そうでなければ0を返す
 */
int is_interior(RDATA *rdata, int customer) {
	int rbc, tail; // rbc means the route for customer belonging to.
	               // tail means the tail of the rcb's route.

	rbc  = search_route_belonging_to(customer, rdata);
	if (rbc == -1) {
		printf("ERROR: In is_interior(), customer %d do not exist in route\n", customer);
		exit(1);
	}
	tail = rdata->route_size[rbc]-1;
	return ((rdata->route[rdata->num_customers*rbc] == customer) || (rdata->route[rdata->num_customers*rbc+tail] == customer)) ? 0 : 1;
}

//! 訪れていない顧客を返す関数
/*!
 * 始めに見つけた未訪問の顧客を返す。未訪問の顧客が見つからなかった場合は0を返す
 */
int unvisited_customer(RDATA *rdata) {
	int customer;

	for (customer=1; customer < rdata->num_customers; customer++) {
		if (isVisited(rdata, customer) == 0) {
			return customer;
		}
	}
	return 0;
}

//! 新しいルートを作成する関数
/*!
 * p1, p2を含むルートを新しく作成する。p2が0のとき、新しく作るルートにはp1のみを含むルートを作成する
 */
int  create_new_route(int p1, int p2, RDATA *rdata) {
	int i, j;

	for (i=0; i < rdata->num_routes; i++) { // 空のルートを探す
		if (rdata->route_size[i] == 0) {
			break;
		}
	}
	if (i == rdata->num_routes) { // 空のルートが無ければ-1を返す
		return -1;
	}

	if (p2 != 0) {
		rdata->route[rdata->num_customers*i]   = p1;
		rdata->route[rdata->num_customers*i+1] = p2;
		rdata->route_size[i] = 2;
		rdata->route_cap[i] = (vrp->demand[p1] + vrp->demand[p2]);
	} else { // p1のみのルートを作成
		rdata->route[rdata->num_customers*i] = p1;
		rdata->route_size[i] = 1;
		rdata->route_cap[i] = vrp->demand[p1];
	}
	return 0;
}

//TODO: デザインとしてよくない。route_customerがinteriorでないことが条件となっている。
//! 顧客route_customerが属しているルートnum_routeにadding_customerをroute_customerの隣に加える
/*!
 * route_customerはnum_route番目のルートに属している顧客、adding_customerは今から追加する顧客を表す
 */
void add_route(RDATA *rdata, int route_customer, int adding_customer) {
	int idx_rc, route_num; // idx_rc is route_customer's index

	if (search_customer(rdata, route_customer, &route_num, &idx_rc) != 0) {
		printf("customer %d do not exist\n", route_customer);
		exit(1);
	}
	if (idx_rc == 0) { // ルートの先頭にadding_customerを加える
		replace_order_of_route(rdata, route_num);
		rdata->route[rdata->route_size[route_num] + rdata->num_customers * route_num] = adding_customer;
		rdata->route_cap[route_num] += vrp->demand[adding_customer];
		rdata->route_size[route_num]++;
	} else if (rdata->route[rdata->num_customers*route_num+(idx_rc+1)] == 0) { // route[CUSTOMER_SIZE * num_route + idx_rc]はnum_route番目の車体のルートの末尾を指しているはず
		rdata->route[rdata->num_customers*route_num+(idx_rc+1)] = adding_customer;
		rdata->route_cap[route_num] += vrp->demand[adding_customer];
		rdata->route_size[route_num]++;
	} else {
		printf("second argument is interior\n");
		exit(1);
	}
}

// TODO: 読みやすくかけないか
void shift_right_route(RDATA *rdata, int route_num, int shift_size) {
	int tail; // route size of route_num

	if (shift_size <= 0) {
		printf("needs shift_size >= 0\n");
		exit(1);
	}

	tail = rdata->route_size[route_num];
	if (tail + shift_size >= rdata->num_customers) {
		printf("over max_length\n");
		exit(1);
	}

	while (tail >= 0) {
		rdata->route[rdata->num_customers*route_num+(tail+shift_size)] = rdata->route[rdata->num_customers*route_num+tail];
		tail--;
	}
	while (shift_size >= 1) {
		rdata->route[rdata->num_customers*route_num+(shift_size-1)] = 0;
		shift_size--;
	}
}

//! ルートnum_routeの顧客の順番を逆にする
void replace_order_of_route(RDATA *rdata, int route_num) {
	int i, tmp;

	for (i=0; i < rdata->route_size[route_num]/2; i++) {
		tmp = rdata->route[rdata->num_customers*route_num+i];
		rdata->route[rdata->num_customers*route_num+i] = rdata->route[rdata->num_customers*route_num+(rdata->route_size[route_num]-(i+1))];
		rdata->route[rdata->num_customers*route_num+(rdata->route_size[route_num]-(i+1))] = tmp;
	}
}


//! nr1番目のルートの直後にnr2番目のルートを結合する (tested)
// nr1, nr2が負の値だったり、ルートサイズより大きいなどおかしな値のときどうする？
void route_cat(RDATA *rdata, int nr1, int nr2) {
	int i, j;

	if (rdata->route_size[nr1] + rdata->route_size[nr2] > rdata->num_customers) {
		printf("ERROR: In route_cat(), size over\n");
		exit(1);
	}

	for (i=0; i < rdata->route_size[nr2]; i++) {
		rdata->route[rdata->num_customers*nr1+(rdata->route_size[nr1]+i)] = rdata->route[rdata->num_customers*nr2+i];
	}
	rdata->route_size[nr1] += rdata->route_size[nr2];
}

//! (tested)
void merge_route(RDATA *rdata, int p1, int p2) {
	int p1_idx, p2_idx;
	int nrp1, nrp2;

	search_customer(rdata, p1, &nrp1, &p1_idx);
	search_customer(rdata, p2, &nrp2, &p2_idx);

	if (p1_idx == 0 && p2_idx == 0) {
		replace_order_of_route(rdata, nrp1);
		route_cat(rdata, nrp1, nrp2);
		memset(&rdata->route[rdata->num_customers*nrp2], 0, rdata->num_customers*sizeof(int));
		rdata->route_cap[nrp1] += rdata->route_cap[nrp2];
		rdata->route_size[nrp2] = 0;
		rdata->route_cap[nrp2] = 0;
	} else if (p1_idx != 0 && p2_idx != 0) {
		replace_order_of_route(rdata, nrp2);
		route_cat(rdata, nrp1, nrp2);
		memset(&rdata->route[rdata->num_customers*nrp2], 0, rdata->num_customers*sizeof(int));
		rdata->route_cap[nrp1] += rdata->route_cap[nrp2];
		rdata->route_size[nrp2] = 0;
		rdata->route_cap[nrp2] = 0;
	} else {
		if (p1_idx == 0) {
			route_cat(rdata, nrp2, nrp1);
			memset(&rdata->route[rdata->num_customers*nrp1], 0, rdata->num_customers*sizeof(int));
			rdata->route_cap[nrp2] += rdata->route_cap[nrp1];
			rdata->route_size[nrp1] = 0;
			rdata->route_cap[nrp1] = 0;
		} else if (p2_idx == 0) {
			route_cat(rdata, nrp1, nrp2);
			memset(&rdata->route[rdata->num_customers * nrp2], 0, rdata->num_customers * sizeof(int));
			rdata->route_cap[nrp1] += rdata->route_cap[nrp2];
			rdata->route_size[nrp2] = 0;
			rdata->route_cap[nrp2] = 0;
		} else {
			printf("error in merge\n");
			exit(1);
		}
	}
}

//! Clark and Wright's methodのアルゴリズムの分岐部分
void branch_cws_method(S_LIST *sl, int edge_idx, RDATA *rdata) {
	int p1, p2;
	int nrp1, nrp2;
	int i;
	int *belong;
	EDGE *e;

	belong = (int *)ec_calloc(rdata->num_customers, sizeof(int));
	for (i=0; i < rdata->num_customers; i++) {
		belong[i] = -1;
	}
	e = &sl->edge[edge_idx];
	p1 = e->from;
	p2 = e->to;
	search_route_belonging_to_a(belong, rdata);
	nrp1 = belong[p1];
	nrp2 = belong[p2];
	free(belong);

	if ((nrp1 == -1) && (nrp2 == -1)) { // どちらの顧客(customer)もルートに属していない
		if (vrp->demand[p1] + vrp->demand[p2] <= rdata->capacity) {
			create_new_route(p1, p2, rdata);
		}
	} else if ((nrp1 == -1) && (nrp2 != -1) && (is_interior(rdata, p2) == 0)) { // 顧客p1のみがルートに属しておらず、p2がinteriorでない
		if (rdata->route_cap[nrp2] + vrp->demand[p1] <= rdata->capacity) {
			add_route(rdata, p2, p1);
		}
	} else if ((nrp2 == -1) && (nrp1 != -1) && (is_interior(rdata, p1) == 0)) { // 顧客p2のみがルートに属しておらず、p1がinteriorでない
		if (rdata->route_cap[nrp1] + vrp->demand[p2] <= rdata->capacity) {
			add_route(rdata, p1, p2);
		}
	} else if ((nrp1 != -1) && (nrp2 != -1) && (nrp1 != nrp2) && (is_interior(rdata, p1) || is_interior(rdata, p2)) == 0) { // どちらの顧客も互いに異なるルートに属しており
		if (rdata->route_cap[nrp1] + rdata->route_cap[nrp2] <= rdata->capacity) {
			merge_route(rdata, p1, p2);
		}
	}
	sl->edge[edge_idx].from = 0;
	sl->edge[edge_idx].to   = 0;
	sl->num_enable_sav--;
}
