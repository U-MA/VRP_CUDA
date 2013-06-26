#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// from SYMPHONY's headers
#include "../compute_cost.h"
#include "../vrp_common_types.h"
#include "../vrp_const.h"
#include "../vrp_io.h"
#include "../vrp_macros.h"
#include "../vrp_types.h"
#include "../sym_macros.h"
#include "../sym_proto.h"
#include "../sym_types.h"
#include "../vrp_lp_params.h"
#include "../vrp_cg_params.h"

#include "../mt19937.h" // メルセンヌツイスターのため
#include "../uct_types.h"
#include "../uct_const.h"
#include "../uct_with_cws.h"
#include "../my_lib.h"


#define SAMPLE 100000

int main(int argc, char **argv) {
	char infile_name[LENGTH] = "../Vrp-All/E/E-n13-k4.vrp";

	vrp_problem *vrp;
	vrp = (vrp_problem *)ec_malloc(sizeof(vrp_problem));
	vrp_io(vrp, infile_name);
	if (vrp->numroutes == 0) {
		vrp->numroutes = atoi(strchr(vrp->name, 'k')+1);
	}

	RDATA *rdata;
	S_LIST *sl;

	rdata = (RDATA *)ec_malloc(sizeof(RDATA));
	rdata_init(rdata, vrp);

	sl = (S_LIST *)ec_malloc(sizeof(S_LIST));
	slist_init(&sl, vrp);

	unsigned long init[4] = { 0x123, 0x234, 0x345, 0x456 }, length = 4; // mt用
	init_by_array(init, length); // mt用

	int *dev_sl;
	cudaMalloc((void **)&dev_sl, sizeof(S_LIST));
	int *dev_data;
	cudaMalloc((void **)&dev_data, sizeof(RDATA));
	int *dev_distance;
	cudaMalloc((void **)&dev_distance, SAMPLE * sizeof(int));

	cudaMemcpy(dev_sl, sl, sizeof(S_LIST), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_sl->edge, sl->edge, vrp->edgenum*sizeof(EDGE), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data, rdata, sizeof(RDATA), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data->route, rdata->route, vrp->vertnum*vrp->numroutes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data->route_num, rdata->route_num, vrp->numroutes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_data->route_size, rdata->route_size, vrp->numroutes, cudaMemcpyHostToDevice);

	bcws_simulation<<<(SAMPLE + 216)/216, 216>>>(dev_sl, dev_data, dev_distance, 5, 40);
	reduction_sum<<<(SAMPLE + 216)/216, 216>>>(dev_distance, SAMPLE);
	
	cudaFree(dev_sl);
	cudaFree(dev_data);

	int dmin;
	cudaMemcpy(&dmin, dev_distance[0], sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_distance);
	printf("best_distance:%d\n", dmin);

	rdata_free(rdata);
	free(sl->edge);
	free(sl);
	free(vrp);
	return 0;
	}
