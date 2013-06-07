#include <stdio.h>
#include <stdlib.h>

#include "uct_with_cws.h"
#include "cws.h"


void slist_init(S_LIST *sl) {
	sl = (S_LIST *)malloc(sizeof(S_LIST));
	sl->edge = (EDGE *)calloc(vrp->edgenum, sizeof(EDGE));
}

//! S_LIST構造体のメンバedgeの中で使われていない要素の個数を求める関数
/*!
 * sl->edgeの中で使われた要素は０になっているため、０になっていない要素の個数を数える。
 * 数える方法にはリダクションを使用する
 */
__global__ void count_sl(S_LIST *dev_sl, int *dev_cnt) {
	int tid;
	__shared__ int temp[dev_sl->num_sav];

	tid = threadIdx.x + threadIdx.y * blockDim.x;
	if (tid < sl->num_sav) { // get nonzero elements of sl->edge
		if (sl->edge[tid].from != 0) { // 使用されていない
			temp[tid] = 1;
		} else {
			temp[tid] = 0;
		}
	}
	__syncthreads();

	// parallel sum
	for (int i=1; i < sl->num_sav; i *=2) {
		if (tid % (2*i) == 0) {
			if (tid+i < sl->num_sav) {
				temp[tid] += temp[tid+i];
			}
		}
		__syncthreads();
	}
	*dev_cnt = temp[0];
}


//! binary cws simulationを行う
/*!
 * savings listを使用しない確率を[prob_left, prob_right]の範囲で設定し、その確率を用いてBinary-CWSを行う。
 * 全スレッドで並列にシミュレーションを行い、それぞれの結果をdev_distance配列の対応する要素に格納
 */
__global__ void bcws_simulation(S_LIST *dev_sl, int *cnt_sl, RDATA *dev_data, int *dev_distance, int prob_left, int prob_right) {

	int probability = -1;
	while (probability <= prob_left || prob_right <= probability) { // savings listを使用しない確率の設定
		probability = genrand_int32() % 101;
	}

	int p, i = 0;
	while (*cnt_sl > 0) { // dev_slを全て使い切るまで繰り返す
		while (dev_sl->edge[i].from == 0) { // 使用されていないsavingsを探す
			if (++i >= dev_sl->num_sav) {
				i = 0;
			}
		}

		p = genrand_int32() % 101;
		if (p >= probability) {
			branch_cws_method(dev_sl, i, dev_data);
			(*cnt_sl)--;
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
