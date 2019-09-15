__global__ void sum_atomic(int nx, int *sum, int *data){
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < nx){
        atomicAdd(sum, data[x]);
    }
}