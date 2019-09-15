#include <stdio.h>

void print_odd_even(int num){
    for (int i = 0; i < num; i++){
        if (i % 2 == 0){
            printf("%d is an even number.\n", i);
        } else if (!(i % 2 == 0)){
	        printf("%d is an odd number.\n", i);
        } else {
            printf("Something wrong...\n");
        }
    }
}