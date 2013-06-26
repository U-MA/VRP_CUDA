#include <stdio.h>
#include <stdlib.h>

#include "uct_with_cws.h"
#include "cws.h"


//! 要素の中で一番小さいものを求める
/*!
 * dev_in[0]に格納される
 */
__global__ void reduction_less(int *dev_in, size_t size) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int tid = x + y * gridDim.x;

	for (int i=1; i < size; i *= 2) {
		if (tid % (2*i) == 0) {
			dev_in[tid] = (dev_in[tid] <= dev_in[tid+i]) ? dev_in[tid] : dev_in[tid+i];
		}
	}
}

//! 顧客customer_numを既に訪れているか確認する関数
/*!
 * route配列を最初から見て行き、customer_numを既に訪れている場合は1、そうでない場合は0を返す。
 */

int isVisited(RDATA *rdata, int customer_num) {
	int i;
	int route_max;

	route_max = rdata->nc * rdata->nr;
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

	route_max = rdata->nc * rdata->nr;
	for (customer_num=1; customer_num < rdata->nc; customer_num++) {
		for (i=0; i < route_max; i++) {
			if (rdata->route[i] == customer_num) // find
				break;
			else if (i == route_max-1) // didn't find
				return 0;
		}
	}
	return 1;
}


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


__global__ void random_simulation(RDATA *d_rdata, PDATA *d_pdata, int *d_distance) {
	int cv; // current working vehicle
	__shared__ int candidates[d_pdata->nc]; // これで定義出来なければ動的に定義する
	__shared__ int cum_candidates;

	int tid = threadIdx.x;

	cv = d_rdata->cur_vehicle;
	while (cv < d_rdata->nr && !allVisit()) {
		if (tid < d_rdata->nc) {
			if (!isVisit()) {
				if (d_rdata->route_cap[cv] + vrp->demand[tid] <= d_pdata->capacity) {
					candidates[num_candidates] = i;
					atomicAdd(&num_candidates, 1);
				}
			}
		}

		__symcthreads();

		if (num_candidates == 0) {
			d_rdata->cur_vehicle++; // change to next vehicle
			d_rdata->idx = 0;
		} else {
			i = rand() % num_candidates;
			move(d_rdata, d_pdata, candidates[i], d_rdata);
		}
	}
	dev_distance[tid] = allVisit() ? calc_distance() : 100000;
}





//! binary cws simulationを行う
/*!
 * savings listを使用しない確率を[prob_left, prob_right]の範囲で設定し、その確率を用いてBinary-CWSを行う。
 * 全スレッドで並列にシミュレーションを行い、それぞれの結果をdev_distance配列の対応する要素に格納
 */
__global__ void bcws_simulation(S_LIST *dev_sl, RDATA *dev_data, int *dev_distance, int prob_left, int prob_right) {

	int probability = -1;
	while (probability <= prob_left || prob_right <= probability) { // savings listを使用しない確率の設定
		probability = genrand_int32() % 101;
	}

	int p, i = 0;
	while (sl->num_enable_sav > 0) { // dev_slを全て使い切るまで繰り返す
		while (dev_sl->edge[i].from == 0) { // 使用されていないsavingsを探す
			if (++i >= dev_sl->num_sav) {
				i = 0;
			}
		}

		p = genrand_int32() % 101;
		if (p >= probability) {
			branch_cws_method(dev_sl, i, dev_data);
		} else {
			i++;
		}
	}

	int rest; // 未訪問の点
	while ((rest = unvisited_customer(dev_data)) != 0) {
		if (create_new_route(rest, 0, rdata) == -1) {
			break;
		}
	}

	int x, y, tid;
	x = threadIdx.x + blockIdx.x * blockDim.x;
	y = threadIdx.y + blockIdx.y * blockDim.y;
	tid = x + y * gridDim.x;
	dev_distance[tid] = calc_distance(dev_data);
}


//! random simulationを行う
/*!
 * 現在のルートの状況から次に訪問可能な点のうち一つをランダムに選び、それを繰り返してルートを作成する。
 */
