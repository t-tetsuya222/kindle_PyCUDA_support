#include <thrust/sort.h>
#include <thrust/execution_policy.h>

extern "C" {

__global__ void sort_thrust(int num_component, int *arr){
thrust::sort(thrust::device, arr, (arr + num_component));
}

__global__ void sort_by_key_thrust( int num_component, int *key, int *value){
thrust::sort_by_key(thrust::device, key, (key + num_component), value);
}
}