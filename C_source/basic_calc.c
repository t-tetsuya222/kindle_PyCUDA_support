#include <stdio.h>
#include <math.h>

void basic_calculation(float num1, float num2) {
	// 四則演算 + - x /
	printf("a + b = %f\n", num1 + num2);
	printf("a - b = %f\n", num1 - num2);
	printf("a x b = %f\n", num1 * num2);
	printf("a / b = %f\n", num1 / num2);
	// べき乗・平方根の計算
	printf("a ** 2 = %f\n", powf(num1, 2.0));
	printf("a ** 0.5 = %f\n", sqrtf(num1));
}