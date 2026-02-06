#ifndef NeuN_H_
#define NeuN_H_
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#define MAT_AT(dst, i, j) (dst).es[(i)*(dst).stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m)

typedef struct {
	size_t rows;
	size_t cols;
	size_t stride;
	float *es;
} MAT;

float sigmoidf(float x);
float rand_float();

MAT MAT_ALLOC(size_t rows, size_t columns);
void MAT_DOT(MAT dst, MAT a, MAT b);
void MAT_SUM(MAT dst, MAT sum); 
void mat_print(MAT dst, const char* name); 
void MAT_RAND(MAT dst, float low, float high);
void MAT_FILL(MAT dst, float fill);
void MAT_FREE(MAT dst);
void MAT_SIG(MAT dst);
void MAT_COPY(MAT dst, MAT copy);
MAT MAT_ROW(MAT m, size_t row);

#endif // NeuN_H_

#ifdef NeuN_IMPLEMENTATION

float rand_float() {
	return (float)rand()/(float)RAND_MAX;
}

MAT MAT_ALLOC(size_t rows, size_t columns) {
	MAT newMAT;
	newMAT.rows = rows;
	newMAT.cols = columns;
	newMAT.stride = columns;
	newMAT.es = (float*)malloc(sizeof(float)*rows*columns);
	return newMAT;
}

void MAT_DOT(MAT dst, MAT a, MAT b) {
	assert(a.cols == b.rows);		
	assert(a.rows == dst.rows);		
	assert(b.cols == dst.cols);

	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
			MAT_AT(dst, i, j) = 0;
		}
	}

	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
			for (size_t k = 0; k < a.cols; k++) {
				MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j); 
			}
		}
	}
	
}

void MAT_SUM(MAT dst, MAT sum) {
	assert(dst.rows == sum.rows);		
	assert(dst.cols == sum.cols);		

	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
			MAT_AT(dst, i, j) += MAT_AT(sum, i, j);
		}
	}
}

void mat_print(MAT dst, const char* name) {
	printf("%s = [\n", name);
	for (size_t i = 0; i < dst.rows; i++) {
		printf("\t");
		for (size_t j = 0; j < dst.cols; j++) {
			printf("%f\t", MAT_AT(dst, i, j));
		}
		printf("\n");
	}
	printf("]\n");
}

// 这是宽度 [0, 1]*5=[0, 5]
// [2, 4] = [0, 1]*2 + 2 = [0, 2] + 2 = [2, 4]
// [15, 18] = [0, 1] * (18 - 15) + 15 = [0, 3] + 15 = [15, 18]
void MAT_RAND(MAT dst, float low, float high) {
	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
			MAT_AT(dst, i, j) = rand_float()*(high-low) + low;
		}
	}
}

void MAT_FILL(MAT dst, float fill) {
	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
			MAT_AT(dst, i, j) = fill;
		}
	}
}

float sigmoidf(float x) {
	return 1.0f/(1.0f + expf(-x));
}

void MAT_SIG(MAT dst) {
	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
			MAT_AT(dst, i, j) = sigmoidf(MAT_AT(dst, i, j));
		}
	}
}

void MAT_FREE(MAT dst) {
	free(dst.es);
}

void MAT_COPY(MAT dst, MAT copy) {
	assert(dst.rows == copy.rows);
	assert(dst.cols == copy.cols);

	for (size_t i = 0; i < dst.rows; i++) {
		for (size_t j = 0; j < dst.cols; j++) {
			MAT_AT(dst, i, j) = MAT_AT(copy, i, j);
		}
	}
}

MAT MAT_ROW(MAT m, size_t row) {
	return (MAT) {
		.rows = 1,
		.cols = m.cols,
		.stride = m.stride,
		.es = &MAT_AT(m, row, 0)
	};
}

#endif
