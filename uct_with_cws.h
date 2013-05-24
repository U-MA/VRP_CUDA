#ifndef _UCT_WITH_CWS_H
#define _UCT_WITH_CWS_H

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
#include "cws.h"


void rdata_init(RDATA *rdata, vrp_problem *vrp);
void rdata_free(RDATA *rdata);

void print_route_tsplib_format(RDATA *rdata);
void print_customer_count(CWS_NODE *pNode, S_LIST *sl);

int isVisited(RDATA *rdata, int customer_num);
int allVisit(RDATA *rdata);
int isAddCws(RDATA *rdata, S_LIST *sl, int sl_idx);

int calc_distance(RDATA *rdata);

int binary_cws_simulation(RDATA *rdata, S_LIST *sl, int probability);

int uct_with_cws(int num_node, S_LIST *sl, RDATA *rdata);
int search_best_savings(S_LIST *sl, RDATA *rdata);

#endif
