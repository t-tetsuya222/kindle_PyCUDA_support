texture<int, 3, cudaReadModeElementType> tex_3d;
__global__ void read_texture_3d(int nx, int ny, int nz){
   int x = threadIdx.x + blockDim.x * blockIdx.x;
   int y = threadIdx.y + blockDim.y * blockIdx.y;
   int z = threadIdx.z + blockDim.z * blockIdx.z;
   if (x < nx && y < ny && z < nz){
        int value = tex3D(tex_3d, x, y, z);
        printf("x: %d, y: %d, z: %d, my value is %d\n", x, y, z, value);
   }
}
