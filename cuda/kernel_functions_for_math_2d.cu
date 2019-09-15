__global__ void add_two_array_kernel(int nx, int ny, float *output, float *arr1, float *arr2){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    int ij = nx * y + x;
    if (x < nx && y < ny){
        output[ij] = arr1[ij] + arr2[ij];
    }
}
