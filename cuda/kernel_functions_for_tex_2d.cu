texture<int, 2, cudaReadModeElementType> tex_2d;
__global__ void read_texture_2d(int nx, int ny){
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;
   if (x < nx && y < ny){
        int value = tex2D(tex_2d, x, y);
        printf("x: %d, y: %d, my value is %d\n", x, y, value);
   }
}
