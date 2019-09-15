__global__ void calc_grad_x_3d(int nx, int ny, int nz, float dx, float *arr_grad, float *arr){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    int ijk = nx * ny * z + nx * y + x;
    int ijk_f = nx * ny * z + nx * y + (x + 1);
    int ijk_b = nx * ny * z + nx * y + (x - 1);
    if (x < nx && y < ny && z < nz){
        // calc gradient of x direction
        if (x == 0){
            arr_grad[ijk] = (arr[ijk_f] - arr[ijk]) / dx;
        } else if (x == (nx - 1)){
            arr_grad[ijk] = (arr[ijk] - arr[ijk_b]) / dx;
        } else {
            arr_grad[ijk] = (arr[ijk_f] - arr[ijk_b]) / (2.0 * dx);
        }
    }
}

