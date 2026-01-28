#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float train_set[][3] = {
	{0.0f, 0.0f, 0.0f},
	{1.0f, 0.0f, 1.0f},
	{0.0f, 1.0f, 1.0f},
	{1.0f, 1.0f, 1.0f},
};

float sigmoid(float x) {
	return 1.0f/(1.0f + expf(-x));
}

#define train_count sizeof(train_set)/sizeof(train_set[0])

float cost(float w1, float w2, float bias) {
	float cost_total = 0.0f;
	for (size_t i = 0; i < train_count; i++) {
		float y = train_set[i][0]*w1 + train_set[i][1]*w2 + bias;
		y = sigmoid(y);
		float costd = y - train_set[i][2]; 
		cost_total += costd*costd;
	}	
	return cost_total;	
}

float rand_float() {
	return (float)rand() / (float)RAND_MAX;
}

int main() {
	srand(time(0));

	float w1 = rand_float();
	float w2 = rand_float();
	float bias = rand_float();
	float esp = 1e-3;
	float rrate = 1e-2;
	size_t count = 10000;

	for (size_t i = 0; i < count; i++) {
		float costd_w1 = ((cost(w1+esp, w2, bias)-cost(w1, w2, bias))/esp);
		float costd_w2 = ((cost(w1, w2+esp, bias)-cost(w1, w2, bias))/esp);
		float costd_bias = ((cost(w1, w2, bias+esp)-cost(w1, w2, bias))/esp);
		w1 -= rrate*costd_w1;
		w2 -= rrate*costd_w2;
		bias -= rrate*costd_bias;
		// printf("%f, %f, %f: %f\n", w1, w2, bias, cost(w1, w2, bias));
		printf("%f\n", cost(w1, w2, bias));
	}

	for (size_t i = 0; i < train_count; i++) {
		float y = train_set[i][0] * w1 + train_set[i][1] * w2 + bias;
		float sigy = sigmoid(y);
		// printf("%f*%f + %f*%f + %f = %f, sig=%f\n", train_set[i][0], w1, train_set[i][1], w2, bias, y, sigy);
	}


	return 0;
}
