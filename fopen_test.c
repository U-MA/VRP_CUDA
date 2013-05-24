#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
	FILE *fp;

	printf("fopen test problem\n");
	fp = fopen("Vrp-All/A/A-n32-k5.vrp", "r");
	if(fp == NULL) {
		printf("can't open this file\n");
		exit(1);
	} else {
		printf("file open!!\n");
		fclose(fp);
	}
	return 0;
}
