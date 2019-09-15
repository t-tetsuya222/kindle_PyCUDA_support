template <class T>
__device__ T add_two_vector(T x, T y){
    return (x + y);
}

extern "C" {
__global__ void add_two_vector_kernel(int nx, int *a, int *b, int *res){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x < nx){
        res[x] = add_two_vector<int>(a[x], b[x]);
    }
}
}