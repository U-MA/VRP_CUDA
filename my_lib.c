#include <stdio.h>
#include <stdlib.h>

#include "my_lib.h"

//! エラーチェック付きcalloc関数
/*!
 * callocにエラーが発生した場合、exit(1)を行う
 */
void *ec_calloc(size_t count, size_t size) {
	void *ptr;
	ptr = calloc(count, size);
	if (ptr == NULL) {
		printf("ERROR: ec_calloc error\n");
		exit(1);
	}
	return ptr;
}

void *ec_malloc(size_t size) {
	void *ptr;
	ptr = malloc(size);
	if (ptr == NULL) {
		printf("ERROR: ec_malloc error\n");
		exit(1);
	}
	return ptr;
}
