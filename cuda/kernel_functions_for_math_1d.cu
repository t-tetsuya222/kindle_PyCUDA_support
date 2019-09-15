__global__ void plus_one_kernel(int num_comp, int *y, int *x){
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i < num_comp){
       y[i] = x[i] + 1;
   }
}
