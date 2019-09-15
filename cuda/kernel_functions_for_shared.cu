# define NUM_THREADS 6
# define X_DIRECTION 0
# define Y_DIRECTION 1
# define Z_DIRECTION 2
# define NUM_HALO 2
__global__ void calc_grad_shared_3d(int nx, int ny, int nz, float dx, float *arr_grad, float *arr){
    __shared__ float arr_s[NUM_THREADS+NUM_HALO][NUM_THREADS+NUM_HALO][NUM_THREADS+NUM_HALO];
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;
    const int tz = threadIdx.z + 1;
    const int nxyz = nx * ny * nz;
    if (x < nx && y < ny && z < nz){
        int ijk = nx * ny * z + nx * y + x;
        //
        //  copy global memory to shared memory
        //
        int ijk_f;
        int ijk_b;
        arr_s[tz][ty][tx] = arr[ijk];
        // halo area
        if (!(x == 0) && (tx == 1)){
            ijk_b = nx * ny * z + nx * y + (x - 1);
            arr_s[tz][ty][tx-1] = arr[ijk_b];
        } else if (!(x == 0) && (tx == NUM_THREADS)){
            ijk_f = nx * ny * z + nx * y + (x + 1);
            arr_s[tz][ty][tx+1] = arr[ijk_f];
        }

        // halo area
        if (!(y == 0) && (ty == 1)){
            ijk_b = nx * ny * z + nx * (y - 1) + x;
            arr_s[tz][ty-1][tx] = arr[ijk_b];
        } else if (!(y == 0) && (ty == NUM_THREADS)){
            ijk_f = nx * ny * z + nx * (y + 1) + x;
            arr_s[tz][ty+1][tx] = arr[ijk_f];
        }

        // halo area
        if (!(z == 0) && (tz == 1)){
            ijk_b = nx * ny * (z - 1) + nx * y + x;
            arr_s[tz-1][ty][tx] = arr[ijk_b];
        } else if (!(z == 0) && (tz == NUM_THREADS)){
            ijk_f = nx * ny * (z + 1) + nx * y + x;
            arr_s[tz+1][ty][tx] = arr[ijk_f];
        }
        __syncthreads();
        //
        // x direction
        //
        // calc gradient of x direction
        if (x == 0){
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr_s[tz][ty][tx+1] - arr_s[tz][ty][tx]) / dx;
        } else if (x == (nx - 1)){
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr_s[tz][ty][tx] - arr_s[tz][ty][tx-1]) / dx;
        } else {
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr_s[tz][ty][tx+1] - arr_s[tz][ty][tx-1]) / (2.0 * dx);
        }
        //
        // y direction
        //
        // calc gradient of y direction
        if (y == 0){
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr_s[tz][ty+1][tx] - arr_s[tz][ty][tx]) / dx;
        } else if (y == (ny - 1)){
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr_s[tz][ty][tx] - arr_s[tz][ty-1][tx]) / dx;
        } else {
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr_s[tz][ty+1][tx] - arr_s[tz][ty-1][tx]) / (2.0 * dx);
        }
        //
        // z direction
        //
        // calc gradient of z direction
        if (z == 0){
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr_s[tz+1][ty][tx] - arr_s[tz][ty][tx]) / dx;
        } else if (z == (nz - 1)){
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr_s[tz][ty][tx] - arr_s[tz-1][ty][tx]) / dx;
        } else {
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr_s[tz+1][ty][tx] - arr_s[tz-1][ty][tx]) / (2.0 * dx);
        }
    }
}

__global__ void calc_grad_global_3d(int nx, int ny, int nz, float dx, float *arr_grad, float *arr){
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;
    const int nxyz = nx * ny * nz;
    int ijk = nx * ny * z + nx * y + x;
    if (x < nx && y < ny && z < nz){
        int ijk_f;
        int ijk_b;
        //
        // x direction
        //
        // calc gradient of x direction
        ijk_f = nx * ny * z + nx * y + (x + 1);
        ijk_b = nx * ny * z + nx * y + (x - 1);
        if (x == 0){
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk]) / dx;
        } else if (x == (nx - 1)){
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr[ijk] - arr[ijk_b]) / dx;
        } else {
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk_b]) / (2.0 * dx);
        }
        //
        // y direction
        //
        // calc gradient of y direction
        ijk_f = nx * ny * z + nx * (y + 1) + x;
        ijk_b = nx * ny * z + nx * (y - 1) + x;
        if (y == 0){
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk]) / dx;
        } else if (y == (ny - 1)){
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr[ijk] - arr[ijk_b]) / dx;
        } else {
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk_b]) / (2.0 * dx);
        }
        //
        // z direction
        //
        // calc gradient of z direction
        ijk_f = nx * ny * (z + 1) + nx * y + x;
        ijk_b = nx * ny * (z - 1) + nx * y + x;
        if (z == 0){
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk]) / dx;
        } else if (z == (nz - 1)){
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr[ijk] - arr[ijk_b]) / dx;
        } else {
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk_b]) / (2.0 * dx);
        }
    }
}

