#include <stdio.h>
#include <math.h>
#include <stdbool.h>

// 0 0 0 0 0 每个数字有两种可能，2^5=32, 同样的16+8+4+2+1+1（0）=32
// 遍历从0到31，输出每个数字代表的二进制数就行了
// 这里有个技巧就是，2的n次方能用1 << n表示，表示讲1往箭头方向，位移动
// 比如 1 << 5那就是00000，100000自然就是2^5了
// 还有一个技巧，给一个数字a和n(总共多少位)
// 要求输出对应的二进制，那就将这个数字进行右移j次，取最低位输出
// 比如5，5，那就遍历从4到0，输出i右移4到0次的最低位
// 也就是，00101（i），第一次要0，那就取右移4次，然后取最后一位（和00001与）
// 0：三次，1：两次；0一次，1零次
// 0：与；1：或；2：非；3：与非；4：异或；5：同或；

void exhaust(int n, int cal) {
	for (int i = 0; i < 1 << n; i++) {
		bool result = 0;
		for (int j = n-1; j >= 0; j--) {
			bool temp = (i >> j) & 1;
			printf("%d ", temp);
			if (j==n-1)
				result = temp; 
			else {
				switch (cal) {
				case 0:
					result = result && temp; 
					break;
				case 1:
					result = result || temp;
					break;
				case 2:
					result = !(result && temp);
					break;
				case 3:
					result = result != temp;
					break;
				case 4:
					result = result == temp;
					break;
				}
			}
		}
		if (cal != -1) {
			printf("%d", result);
		}
		printf("\n");
	}
	printf("\n");
}

int main() {
	// 我要输出n个点的xy坐标，然后用gnu plot出来
	// 用来认识与或非、异或、同或……等等操作哪些可以被线性分割

	exhaust(2, 4);

	return 0;
}
