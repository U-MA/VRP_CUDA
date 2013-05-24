#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char **argv) {
	vrp_problem *vrp;
	int i, j;
	int num_customer;
	char infile_name[LENGTH];
	distances *dist;

	if (argc == 2) {
		strcpy(infile_name, argv[1]);
	} else {
		printf("usage: %s vrp_filename\n", argv[0]);
		exit(1);
	}

	vrp = (vrp_problem *) malloc(sizeof(vrp_problem));

	vrp_io(vrp, infile_name);
	num_customer = vrp->vertnum;

	dist = &vrp->dist;
	for (i=0; i < num_customer; i++) {
		for (j=0; j < num_customer; j++) {
			if (i != j) {
				printf("%2d-%2d:%3d\n", i, j, dist->cost[INDEX(i, j)]);
			}
		}
	}
	return 0;
}
