#include <stdio.h>

void swap_int(int *x, int *y) {
	int tmp;
	tmp = *x;
	*x = *y;
	*y = tmp;
}

void insert_heap(int x, int *h) {
	int n, tmp;

	h[0]++;
	n = h[0];
	h[n] = x;
	while ((n != 1) && (h[n/2] < h[n])) {
		swap_int(&h[n], &h[n/2]);
		n = n/2;
	}
}

int delete_max(int *h) {
	int ret, n, i, tmp;

	n = h[0];
	i = 1;
	ret = h[1];
	h[1] = h[n];
	h[0]--;
	if (n == 0) {
		printf("heap is empty\n");
		return -1;
	}

	i = 2*1;
	while (i < n) {
		if (i < n && h[i] < h[i+1])
			i++;
		if (h[i/2] < h[i]) {
			swap_int(&h[i/2], &h[i]);
		} else {
			break;
		}
		i = 2*i;
	}
	return ret;
}

int main() {
	int test_array[10] = {
		9, 0, 1, 4, 3, 6, 7, 2, 5, 8
	;
	int i, h[11];

	h[0] = 0;
	for (i=0; i < 10; i++) {
		insert_heap(test_array[i], h);
	}
	for (i=0; i < 10; i++) {
		printf("No.%d = %d\n", i, delete_max(h));
	}
	return 0;
}
