__global__ void add_two_vector(int nx, float *arr1, float *arr2, float *res){
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   if (x < nx){
       res[x] = arr1[x] + arr2[x];
   }
}

__global__ void add_two_vector_dynamic(int *grid, int *block, int nx, float *arr1, float *arr2, float *res){
dim3 grid_ = dim3(grid[0], grid[1], grid[2]);
dim3 block_ = dim3(block[0], block[1], block[2]);
// カーネルを呼び出す
add_two_vector<<<grid_, block_>>>(nx, arr1, arr2, res);
}
