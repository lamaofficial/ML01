#define NeuN_IMPLEMENTATION
#include "ml08.h"

//  a0是输入端，有n个输入，那就是1*n的矩阵
//  w1是第一层对输入a0的权重矩阵，
// 		因为每一层可以有m个神经元，
// 		所以先思考，要把输入乘以权重，
// 		那矩阵大小至少有n*1大小，换句话说
// 		n*1是一个神经元的w1大小，
// 		n*m就是m个神经元的w1大小
// 		b1就对应了a0，行列都得一样，1*n
// 	运算过程是
// 		乘法需要有一个矩阵承载，即dst
// 		我们把每一层的这个承载，叫做a
// 		同时也是下一层的输入端
// 		a1就是第一层的承载，
// 		首先把a0*w1，放到a1，
// 		再把a1+b1（自动会放到a1）
// 		再把a1的每个元素，套上激活函数
// 	往往最后一层都只有一个神经元（即一个输出）
// 		所以最后一层的
// 			w，都是n*1
// 			b，都是1*1
// 			a，自然也是1*1
//
typedef struct {
	MAT a0;
	MAT w1, b1, a1;
	MAT w2, b2, a2;
} XOR;

void forward(XOR m) {
	MAT_DOT(m.a1, m.a0, m.w1); 
	MAT_SUM(m.a1, m.b1); 
	MAT_SIG(m.a1);

	MAT_DOT(m.a2, m.a1, m.w2);
	MAT_SUM(m.a2, m.b2);
	MAT_SIG(m.a2);
}
 
float td[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0,
};

// ti, training input, to, training output, td, traning data
float cost(XOR m, MAT ti, MAT to) {
	assert(ti.rows == to.rows);
	assert(to.cols == m.a2.cols);

	float costs = 0;
	for (size_t i = 0; i < ti.rows; i++) {
		// 把输入集合的第i行，直接复制到XOR的a0，然后forward
		MAT x = MAT_ROW(ti, i);
		MAT y = MAT_ROW(to, i);
		MAT_COPY(m.a0, x);
		forward(m);
		
		for (size_t j = 0; j < y.cols; j++) {
			float d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j); 
			costs += d*d;	
		}
	}
	return costs/ti.rows;
}


int main() {
	srand(time(0));

	size_t stride = 3;
	size_t n = sizeof(td)/sizeof(td[0])/stride;
	MAT ti = {
		.rows = n,
		.cols = 2,
		.stride = stride, 
		.es = td
	};
	
	MAT to = {
		.rows = n,
		.cols = 1, 
		.stride = stride, 
		.es = td + 2
	};

	MAT_PRINT(ti);
	MAT_PRINT(to);

	XOR xor;

	xor.a0 = MAT_ALLOC(1, 2);
	xor.w1 = MAT_ALLOC(2, 2);
	xor.b1 = MAT_ALLOC(1, 2);
	xor.a1 = MAT_ALLOC(1, 2);

	xor.w2 = MAT_ALLOC(2, 1);
	xor.b2 = MAT_ALLOC(1, 1);
	xor.a2 = MAT_ALLOC(1, 1);

	MAT_RAND(xor.w1, 0.0f, 1.0f);
	MAT_RAND(xor.b1, 0.0f, 1.0f);
	MAT_RAND(xor.a1, 0.0f, 1.0f);
	MAT_RAND(xor.w2, 0.0f, 1.0f);
	MAT_RAND(xor.b2, 0.0f, 1.0f);
	MAT_RAND(xor.a2, 0.0f, 1.0f);

	for (size_t i = 0; i < 2; i++) {
		for (size_t j = 0; j < 2; j++) {
			MAT_AT(xor.a0, 0, 1) = i;
			MAT_AT(xor.a0, 0, 2) = j;

			forward(xor);

			float y = *xor.a2.es;

			printf("%zu ^ %zu = %f\n", i, j, y);
		}
	}
	

	return 0;
} 
