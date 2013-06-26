#ifndef _CWS_H
#define _CWS_H

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

#include "uct_types.h"


typedef struct edge {
	int from;
	int to;
} EDGE;

// 型名をSAVINGSにしたかったがSAVINGSにするとgccでエラーが発生するためにSAVSにした
typedef struct savings {
	EDGE edge;
	int  sav;
} SAVS;

typedef struct s_list {
	int num_sav; // savingsの数
	int num_enable_sav; // 使用されていないsavingsの数
	EDGE *edge;
} S_LIST;


void print_savings_list(const S_LIST *sl);

void swap_savings(SAVS *x, SAVS *y);
void insert_heap_savings(SAVS *s, int from, int to, int sav);
EDGE delete_max_savings(SAVS *s);

void slist_init(S_LIST **sl, vrp_problem *vrp);
void create_savings_list(S_LIST *sl);
void delete_savings(S_LIST *sl, int index);

int search_route_belonging_to(int customer, RDATA *rdata);
void search_route_belonging_to_a(int *belong, RDATA *rdata);
int search_customer(RDATA *rdata, int customer, int *route_num, int *idx);

int is_interior(RDATA *rdata, int customer);
int unvisited_customer(RDATA *rdata);

int create_new_route(int p1, int p2, RDATA *rdata);
void add_route(RDATA *rdata, int route_customer, int adding_customer);

void shift_right_route(RDATA *rdata, int route_num, int shift_size);
void replace_order_of_route(RDATA *rdata, int route_num);
void route_cat(RDATA *rdata, int nr1, int nr2);
void merge_route(RDATA *rdata, int p1, int p2);

void branch_cws_method(S_LIST *sl, int edge_idx, RDATA *rdata);

#endif
