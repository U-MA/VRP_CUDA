#include <stdio.h>
#include <stdlib.h>

int penarty;

//! 次に進む顧客をルートに追加する関数
/*!
 * customer_numはcapacityの制限を超過しないことを仮定している
 */
__device__ void move(RDATA *rdata, PDATA *pdata, int customer_num) {
	if (customer_num == 0) {
		printf("CUSTOMER NUMBER IS 0 in move().\n");
		exit(1);
	} else {
		int offset = rdata->cur_vehicle * rdata->nc + rdata->idx;
		rdata->route[offset] = customer_num;
		rdata->route_cap[rdata->cur_vehicle] += pdata->demands[customer_num];
		rdata->idx++;
	}
}

//! ルートの合計距離を返す関数
/*!
 * 倉庫と最初の車体(vehicle_num)が最初に訪問した顧客との距離、その最初に訪問した顧客と2番目に訪問した顧客との距離、2番目に訪問した顧客と3番目に訪問した顧客との距離...
 * 最後に訪問した顧客と倉庫との距離、これらを全て合計する。これを全ての車体に対して行う。そうして全てを合計した距離を返す。\n
 */
int calc_distance(RDATA *rdata, PDATA *pdata) {
	int sum_distance;
	int route_num;
	int i;
	int c1, c2;

	sum_distance = 0;
	for (route_num=0; route_num < rdata->nr; route_num++) {
		c2 = DEPOT_NUM;
		for (i=0; i < rdata->nc; i++) {
			c1 = c2;
			c2 = rdata->route[route_num*rdata->nc+i];
			if (c2 == DEPOT_NUM) {
				sum_distance += pdata->cost[INDEX(c1, DEPOT_NUM)];
				break; // 次のルートへ
			} else {
				sum_distance += pdata->cost[INDEX(c1, c2)];
			}
		}
		if (i == rdata->nc) {
			sum_distance += pdata->cost[INDEX(c2, DEPOT_NUM)];
		}
	}
	return sum_distance;
}

__global__ void random_simulation(RDATA *d_rdata, PDATA *d_pdata, int *d_distance) {
	__shared__ int candidates[d_pdata->nc]; // これで定義出来なければ動的に定義する
	__shared__ int num_candidates;

	int tid = threadIdx.y * blockDim.x + threadIdx.x;

	int cv = d_rdata->cur_vehicle; // current working vehicle
	while (cv < d_rdata->nr && !allVisit()) {
		if (tid == 0)
			num_candidates = 0;

		if (tid < d_rdata->nc) {
			if (!isVisit(tid)) {
				if (d_rdata->route_cap[cv] + vrp->demand[tid] <= d_pdata->capacity) {
					candidates[num_candidates] = i;
					atomicAdd(&num_candidates, 1);
				}
			}
		}

		__syncthreads();

		if (num_candidates == 0) {
			d_rdata->cur_vehicle++; // change to next vehicle
			d_rdata->idx = 0;
		} else {
			i = rand() % num_candidates;
			move(d_rdata, d_pdata, candidates[i], d_rdata);
		}
		__syncthreads();
	}
	d_distance[tid] = allVisit() ? calc_distance() : penarty;
}


void capsule_random_simulation(RDATA *rdata, PDATA *pdata) {
}


int main(int argc, char **argv) {
	return 0;
}
