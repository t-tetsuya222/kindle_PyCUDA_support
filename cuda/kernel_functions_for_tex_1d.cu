texture<int, 1, cudaReadModeElementType> tex_1d;
__global__ void read_texture_1d(int nx){
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   if (x < nx){
        int value = tex1Dfetch(tex_1d, x);
        printf("my id is %d, my value is %d\n", x, value);
   }
}
