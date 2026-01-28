// 小优化方向：一个是计算时间，当cost小于某个阈值时就停止训练，对于n次训练完成，的平均训练时间，以及尝试其他train_set
// 一个是尝试用矩阵表示和计算参数
// 2个隐藏神经元（a和b）+ 1个输出神经元（c），正好可以学习XOR这样的非线性问题
// 还有什么其他非线性问题，能够用这三个神经元解决呢？
// 将损失函数随训练次数的减少做成图表展示（中间其实有一段，减少的突然会慢下来）
// 能否通过损失函数值的变化（也就是每一点斜率的变化，找到/画出这次训练的曲线？）
// 试一试其他的随机值（比如说，乘十倍？）
// 将输出格式进行优化（print_xor）
// forward函数值的变化，用点表示，预测值（一个数）在数轴上的变动方向，向固定的期望值接近的过程
// 能不能就是根据每个神经元的，三个参数
// （也就是w1,w2,bias, 画一条直线，也就是一共三条，然后在更新参数后存一帧，看这条线的变化过程）
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// 1 只想“最初的输入”，和想要的“最终的输出”
float train_set[][3] = {
	{0, 0, 0},
	{0, 1, 1},
	{1, 0, 1},
	{1, 1, 0},
};
// 想要的输出总状态数
#define train_count 4

// 2 随机函数 和 激活函数
// 记得主函数要加入srand(time(0));或是固定
float rand_float() {
	return (float)rand() / (float)RAND_MAX;
}
float sigmoidf(float x) {
	return 1/(1+expf(-x));
}

// 3 定义和初始化参数矩阵
typedef struct {
	float a_w1;
	float a_w2;
	float a_bias;
	float b_w1;
	float b_w2;
	float b_bias;
	float c_w1;
	float c_w2;
	float c_bias;
} xor;

void init_xor(xor *d) {
	d->a_w1 = rand_float();
	d->a_w2 = rand_float();
	d->a_bias = rand_float();
	d->b_w1 = rand_float();
	d->b_w2 = rand_float();
	d->b_bias = rand_float();
	d->c_w1 = rand_float();
	d->c_w2 = rand_float();
	d->c_bias = rand_float();
}

// 4 打印参数矩阵
// 打印目前矩阵的参数
void print_xor(xor *d) {
	printf("%f\n", d->a_w1);
	printf("%f\n", d->a_w2);
	printf("%f\n", d->a_bias);
	printf("%f\n", d->b_w1);
	printf("%f\n", d->b_w2);
	printf("%f\n", d->b_bias);
	printf("%f\n", d->c_w1);
	printf("%f\n", d->c_w2);
	printf("%f\n", d->c_bias);
}

// 5 前向传播函数
// 对于一个矩阵，给输入的两个值，根据当前矩阵内的参数，输出其预测值（result_pred）
float forward(xor *d, float x, float y) {
	float a = sigmoidf(x*d->a_w1 + y*d->a_w2 + d->a_bias);
	float b = sigmoidf(x*d->b_w1 + y*d->b_w2 + d->b_bias);
	return sigmoidf(a*d->c_w1 + b*d->c_w2 + d->c_bias);
}

// 6 打印测试（列举：对最初的输入，和最后的预测值）
void print_test(xor *d) {
	for (size_t i = 0; i < train_count; i++) {
		printf("%f %f : %f\n", train_set[i][0], train_set[i][1], forward(d, train_set[i][0], train_set[i][1]));
	}
}

// 7 损失函数（loss）
// 对于一个矩阵，根据其输出的值，对每个train_set的输入进行预测，将预测值减去期望值，做均方误差的累加，然后返回最终的平均总损失
float cost(xor *d) {
	float costs = 0;
	for (size_t i = 0; i < train_count; i++) {
		float result_pred = forward(d, train_set[i][0], train_set[i][1]);
		float delta = result_pred - train_set[i][2];
		costs += delta*delta;
	}
	// 注意返回的是平均每一次的损失值
	return costs/train_count;
}

// 8 损失斜率矩阵
// 对每个参数：
// 保存这个参数，加esp，将这个参数改变后的“损失斜率”，赋值给新的一个矩阵的对应元素
// 然后还原原来的矩阵，重复九次（轮流对每个参数）
xor* finite_d(xor *d, float esp) {
	xor *e = malloc(sizeof(xor));
	float cost_bf = cost(d);
	float saved = 0;

	saved = d->a_w1;
	d->a_w1 += esp;
	e->a_w1 = (cost(d) - cost_bf) / esp;
	d->a_w1 = saved;  
	
	saved = d->a_w2;
	d->a_w2 += esp;
	e->a_w2 = (cost(d) - cost_bf) / esp;
	d->a_w2 = saved;  
	
	saved = d->a_bias;
	d->a_bias += esp;
	e->a_bias = (cost(d) - cost_bf) / esp;
	d->a_bias = saved;  
	
	saved = d->b_w1;
	d->b_w1 += esp;
	e->b_w1 = (cost(d) - cost_bf) / esp;
	d->b_w1 = saved;  
	
	saved = d->b_w2;
	d->b_w2 += esp;
	e->b_w2 = (cost(d) - cost_bf) / esp;
	d->b_w2 = saved;  
	
	saved = d->b_bias;
	d->b_bias += esp;
	e->b_bias = (cost(d) - cost_bf) / esp;
	d->b_bias = saved;  
	
	saved = d->c_w1;
	d->c_w1 += esp;
	e->c_w1 = (cost(d) - cost_bf) / esp;
	d->c_w1 = saved;  
	
	saved = d->c_w2;
	d->c_w2 += esp;
	e->c_w2 = (cost(d) - cost_bf) / esp;
	d->c_w2 = saved;  
	
	saved = d->c_bias;
	d->c_bias += esp;
	e->c_bias = (cost(d) - cost_bf) / esp;
	d->c_bias = saved;  
	
	return e;
}

// 9 学习函数
// 已知一个参数矩阵和损失斜率矩阵
// 对每个参数向有利于减少损失的方向，单独进行更新
void learn(xor *d, xor* e, float lrate) {
	d->a_w1 -= e->a_w1*lrate;
	d->a_w2 -= e->a_w2*lrate;
	d->a_bias -= e->a_bias*lrate;
	d->b_w1 -= e->b_w1*lrate;
	d->b_w2 -= e->b_w2*lrate;
	d->b_bias -= e->b_bias*lrate;
	d->c_w1 -= e->c_w1*lrate;
	d->c_w2 -= e->c_w2*lrate;
	d->c_bias -= e->c_bias*lrate;
}

// 10 主函数
int main() {
	// 初始化随机参数、最小量、学习率和参数矩阵，并打印参数矩阵
	srand(time(0));
	
	float esp = 1e-3f;
	float lrate = 5e-2f;

	xor *a = malloc(sizeof(xor));
	init_xor(a);
	print_xor(a);
	
	// 初始化损失斜率矩阵
	xor *e;
	size_t count = 99999;
	for (size_t i = 0; i < count; i++) {
		// 更新损失斜率矩阵
		e = finite_d(a, esp);
		// 学习：更新参数矩阵
		learn(a, e, lrate);
		
		// 释放e
		free(e);

		// 打印当前损失值
		printf("costs: %f\n", cost(a));
	}

	// 打印参数矩阵，打印测试情况
	print_xor(a);
	print_test(a);
	free(a);

	return 0;
}
