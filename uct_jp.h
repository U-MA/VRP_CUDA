#ifndef _UCT_JP_H
#define _UCT_JP_H


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


void print_route();
void print_customer_count();

int isVisited(int customer_num);
int allVisit();

int calc_distance();

int move(int customer_num, int *route);
int simulation();

void create_child(NODE *pNode, int customer_num);
int create_node();

int uct();

int search_best_customer();

void initialize_globa_var(vrp_problem *vrp);

void free_global_var();

void tolower_s(char *str);

int get_opt_from_file(char *optfile);

#endif
